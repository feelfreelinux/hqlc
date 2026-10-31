#include "hqlc_container.h"

#include <string.h>

// sync + cfg + frame index
#define PAGE_HEADER_BYTES 7

// total_samples + last_page_offset + magic
#define TRAILER_BYTES 20

// channels - 1 in low bits, rest bitstream version
#define CFG_CHANNELS_MASK 0x0F

#define MAX_PAD 511

// Window tail kept on sequential refills, so a header can be re-read
#define WINDOW_KEEP_BYTES 16

static const uint8_t page_sync[2] = {0x68, 0x71}; // "hq"

// Writer state
struct hqlc_container_writer {
  uint64_t offset; // bytes emitted so far
  uint64_t frames_written;
  uint64_t page_offset; // start of the current page
  uint16_t pre_skip;
  uint16_t prev_len; // previous frame length, for the in-page delta coding
  uint16_t crc;      // running page CRC
  uint8_t cfg;
  uint8_t page_frame_count; // frames in the current page
};

// Reader state
struct hqlc_container_reader {
  hqlc_container_io io;
  hqlc_container_info info;
  uint64_t last_page_offset; // page holding the end marker, from the trailer
  uint64_t read_offset;      // next byte to parse
  uint64_t output_start;     // untrimmed sample index where output begins
  // Read window over the reader
  uint64_t window_offset;
  size_t window_len;
  uint32_t last_page_index;  // frame_index of the last page
  uint32_t page_frame_index; // frame_index of the current page
  uint16_t crc;              // running page CRC
  uint16_t prev_len;         // previous frame length, for the in-page delta coding
  uint16_t pending_len;      // next frame length, read ahead
  uint16_t pad;              // set once the look-ahead has found the end marker
  uint8_t cfg;
  uint8_t page_frame_count; // frames consumed from the current page
  bool page_open;
  bool continuous; // the current page is known to follow the previous one
  bool ended;
  bool reset;
  bool damaged;
  bool io_error;
  uint8_t window[2048];
};

size_t hqlc_container_writer_size(void) {
  return sizeof(hqlc_container_writer);
}

size_t hqlc_container_reader_size(void) {
  return sizeof(hqlc_container_reader);
}

// CRC-16 over a page, matching FLAC, 0x8005 poly
static uint16_t page_crc16(uint16_t crc, const uint8_t *data, size_t len) {
  while (len--) {
    crc ^= (uint16_t)(*data++ << 8);
    for (int bit = 0; bit < 8; bit++) {
      bool top_bit = crc & 0x8000;
      crc = (uint16_t)(crc << 1);
      if (top_bit) {
        crc ^= 0x8005;
      }
    }
  }
  return crc;
}

// low endian write helper
static void write_le(uint8_t *dst, uint64_t value, int bytes) {
  for (int i = 0; i < bytes; i++) {
    dst[i] = (uint8_t)(value >> (8 * i));
  }
}

// low endian read helper
static uint64_t read_le(const uint8_t *src, int bytes) {
  uint64_t value = 0;
  while (bytes--) {
    value = value << 8 | src[bytes];
  }
  return value;
}

// encode a signed value with zigzag
static uint32_t zigzag_encode(int32_t val) {
  if (val >= 0) {
    return (uint32_t)val << 1;
  }
  return ((uint32_t)-val << 1) - 1;
}

// decode the zigzag value
static int32_t zigzag_decode(uint32_t code) {
  if (code & 1) {
    return -(int32_t)((code + 1) >> 1);
  }
  return (int32_t)(code >> 1);
}

hqlc_container_error hqlc_container_writer_init(hqlc_container_writer *writer,
                                                uint8_t channels,
                                                uint16_t pre_skip,
                                                uint8_t *out) {
  if (!writer || !out || channels < 1 || channels > HQLC_CONTAINER_MAX_CHANNELS) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }

  // Write the header
  memset(writer, 0, sizeof(*writer));
  writer->cfg = (uint8_t)((HQLC_BITSTREAM_VERSION << 4) | (channels - 1));
  writer->pre_skip = pre_skip;
  writer->offset = HQLC_CONTAINER_HEADER_BYTES;
  memcpy(out, "HQLC", 4);
  out[4] = HQLC_CONTAINER_VERSION;
  out[5] = 0;
  write_le(out + 6, pre_skip, 2);
  return HQLC_CONTAINER_OK;
}

// emit entry len, write first page if needed
static size_t
write_entry_length(hqlc_container_writer *writer, uint16_t len, uint8_t *out) {
  size_t size = 0;
  if (writer->page_frame_count == 0) {
    writer->page_offset = writer->offset;
    writer->crc = 0;
    memcpy(out, page_sync, 2);
    out[2] = writer->cfg;
    write_le(out + 3, writer->frames_written, 4);
    size = PAGE_HEADER_BYTES;
  }
  // The first entry of a page stores the length, later ones the delta
  uint32_t code = zigzag_encode((int32_t)len - writer->prev_len);
  if (writer->page_frame_count == 0) {
    code = len;
  }
  // Values fit 14 bits, so the varint is 1 or 2 bytes
  if (code > 0x7F) {
    out[size++] = (uint8_t)(code | 0x80);
    out[size++] = (uint8_t)(code >> 7);
  } else {
    out[size++] = (uint8_t)code;
  }
  writer->crc = page_crc16(writer->crc, out, size);
  writer->prev_len = len;
  return size;
}

hqlc_container_error hqlc_container_write_frame(hqlc_container_writer *writer,
                                                const uint8_t *frame,
                                                size_t len,
                                                uint8_t *out,
                                                size_t *out_len) {
  // frame_index is 32 bits, with room left for the end marker page
  if (!writer || !frame || !out || !out_len || len < 1 || len > HQLC_MAX_FRAME_BYTES ||
      writer->frames_written >= UINT32_MAX - HQLC_CONTAINER_PAGE_FRAMES) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  size_t size = write_entry_length(writer, (uint16_t)len, out);
  memcpy(out + size, frame, len);
  writer->crc = page_crc16(writer->crc, frame, len);
  size += len;
  writer->frames_written++;
  if (++writer->page_frame_count == HQLC_CONTAINER_PAGE_FRAMES) {
    write_le(out + size, writer->crc, 2);
    size += 2;
    writer->page_frame_count = 0;
  }
  writer->offset += size;
  *out_len = size;
  return HQLC_CONTAINER_OK;
}

hqlc_container_error hqlc_container_write_end(hqlc_container_writer *writer,
                                              uint64_t input_samples,
                                              uint8_t *out,
                                              size_t *out_len) {
  if (!writer || !out || !out_len) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  // The frames must cover pre_skip plus the input
  uint64_t untrimmed = writer->frames_written * HQLC_FRAME_SAMPLES;
  uint64_t needed = writer->pre_skip + input_samples;
  if (untrimmed < needed || untrimmed - needed > MAX_PAD) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  uint16_t pad = (uint16_t)(untrimmed - needed);
  size_t size = write_entry_length(writer, 0, out);
  write_le(out + size, pad, 2);
  writer->crc = page_crc16(writer->crc, out + size, 2);
  write_le(out + size + 2, writer->crc, 2);
  size += 4;
  write_le(out + size, input_samples, 8);
  write_le(out + size + 8, writer->page_offset, 8);
  memcpy(out + size + 16, "HQLT", 4);
  size += TRAILER_BYTES;
  writer->page_frame_count = 0;
  writer->offset += size;
  *out_len = size;
  return HQLC_CONTAINER_OK;
}

// Pointer into the read window for offset, refilled as needed, NULL at end of data
static const uint8_t *
bytes_at(hqlc_container_reader *reader, uint64_t offset, size_t *avail) {
  uint64_t end = reader->window_offset + reader->window_len;
  if (offset < reader->window_offset || offset >= end) {
    size_t keep = 0;
    if (offset == end && reader->window_len > WINDOW_KEEP_BYTES) {
      // Sequential refill keeps a short tail for header look-backs
      keep = WINDOW_KEEP_BYTES;
      memmove(reader->window, reader->window + reader->window_len - keep, keep);
      reader->window_offset = end - keep;
    } else {
      reader->window_offset = offset;
    }
    size_t got = 0;
    if (reader->io.read_at(reader->io.ctx,
                           reader->window_offset + keep,
                           reader->window + keep,
                           sizeof(reader->window) - keep,
                           &got)) {
      reader->io_error = true;
      got = 0;
    }
    reader->window_len = keep + got;
    if (offset >= reader->window_offset + reader->window_len) {
      return NULL;
    }
  }
  *avail = (size_t)(reader->window_offset + reader->window_len - offset);
  return reader->window + (offset - reader->window_offset);
}

// Copy bytes without consuming them, returns the count available
static size_t
peek_bytes(hqlc_container_reader *reader, uint64_t offset, uint8_t *dst, size_t len) {
  size_t done = 0;
  size_t available;
  const uint8_t *src;
  while (done < len && (src = bytes_at(reader, offset + done, &available))) {
    size_t chunk = len - done;
    if (chunk > available) {
      chunk = available;
    }
    memcpy(dst + done, src, chunk);
    done += chunk;
  }
  return done;
}

// Consume bytes into dst, updating the page crc on the go
static int read_bytes(hqlc_container_reader *reader, uint8_t *dst, size_t len) {
  size_t avail;
  const uint8_t *src;
  while (len) {
    if (!(src = bytes_at(reader, reader->read_offset, &avail))) {
      return -1;
    }
    size_t chunk = len;
    if (chunk > avail) {
      chunk = avail;
    }
    if (dst) {
      memcpy(dst, src, chunk);
      dst += chunk;
    }
    reader->crc = page_crc16(reader->crc, src, chunk);
    reader->read_offset += chunk;
    len -= chunk;
  }
  return 0;
}

// LEB128, at most 2 bytes per format
static int read_varint(hqlc_container_reader *reader, uint32_t *value) {
  uint8_t bytes[2] = {0, 0};
  if (read_bytes(reader, bytes, 1) ||
      ((bytes[0] & 0x80) && (read_bytes(reader, bytes + 1, 1) || (bytes[1] & 0x80)))) {
    return -1;
  }
  *value = (bytes[0] & 0x7Fu) | (uint32_t)bytes[1] << 7;
  return 0;
}

// Validate a page header, returns 0 and its frame_index when valid
static int parse_page_header(const hqlc_container_reader *reader,
                             const uint8_t *header,
                             uint32_t *page_index) {
  *page_index = (uint32_t)read_le(header + 3, 4);
  if (memcmp(header, page_sync, 2) != 0 || header[2] != reader->cfg ||
      *page_index % HQLC_CONTAINER_PAGE_FRAMES != 0) {
    return -1;
  }
  return 0;
}

/**
 * @brief Scan forward for the first page header in a byte range
 *
 * @param reader Open reader, its window is moved by the scan.
 * @param from_offset First byte offset to try, from the start of the stream.
 * @param to_offset Byte offset where the scan stops, exclusive.
 * @param min_index Lowest acceptable frame_index of the page, inclusive.
 * @param max_index Highest acceptable frame_index of the page, inclusive.
 * @param page_offset Receives the byte offset of the page found.
 * @param page_index Receives the frame_index of the page found.
 * @return 0 when a page was found, -1 when the range or the data ran out.
 */
static int scan_for_page(hqlc_container_reader *reader,
                         uint64_t from_offset,
                         uint64_t to_offset,
                         uint32_t min_index,
                         uint32_t max_index,
                         uint64_t *page_offset,
                         uint32_t *page_index) {
  for (uint64_t offset = from_offset; offset < to_offset; offset++) {
    size_t available;
    const uint8_t *bytes = bytes_at(reader, offset, &available);
    if (!bytes) {
      return -1;
    }
    if (bytes[0] != page_sync[0]) {
      continue;
    }
    uint8_t header[PAGE_HEADER_BYTES];
    if (peek_bytes(reader, offset, header, PAGE_HEADER_BYTES) == PAGE_HEADER_BYTES &&
        parse_page_header(reader, header, page_index) == 0 && *page_index >= min_index &&
        *page_index <= max_index) {
      *page_offset = offset;
      return 0;
    }
  }
  return -1;
}

// marks the reader as damaged
static void mark_damaged(hqlc_container_reader *reader) {
  reader->reset = true;
  reader->damaged = true;
}

// Check the page CRC
static void close_page(hqlc_container_reader *reader) {
  uint16_t expected = reader->crc;
  uint8_t stored[2];
  reader->page_open = false;
  if (read_bytes(reader, stored, 2) || read_le(stored, 2) != expected) {
    mark_damaged(reader);
  }
}

// Open the page at the read position, -1 at end of data or on a bad header
static int open_page(hqlc_container_reader *reader) {
  uint8_t header[PAGE_HEADER_BYTES];
  uint32_t page_index;
  reader->crc = 0;
  if (read_bytes(reader, header, PAGE_HEADER_BYTES) ||
      parse_page_header(reader, header, &page_index)) {
    return -1;
  }
  if (reader->continuous &&
      page_index != reader->page_frame_index + HQLC_CONTAINER_PAGE_FRAMES) {
    mark_damaged(reader); // pages went missing
  }
  reader->page_frame_index = page_index;
  reader->page_frame_count = 0;
  reader->page_open = true;
  reader->continuous = true;
  return 0;
}

// Read one entry length into pending, or the end marker and its pad
static int read_entry_length(hqlc_container_reader *reader) {
  uint32_t code;
  uint8_t pad[2];
  if (read_varint(reader, &code)) {
    return -1;
  }
  int32_t len = reader->prev_len + zigzag_decode(code);
  if (reader->page_frame_count == 0) {
    len = (int32_t)code; // the first entry of a page stores the length itself
  }
  if (len > 0 && len <= HQLC_MAX_FRAME_BYTES) {
    reader->prev_len = (uint16_t)len;
    reader->pending_len = (uint16_t)len;
    return 0;
  }
  if (len != 0 || read_bytes(reader, pad, 2) || read_le(pad, 2) > MAX_PAD) {
    return -1;
  }
  reader->pad = (uint16_t)read_le(pad, 2);
  reader->ended = true;
  close_page(reader); // a missing CRC here loses no audio
  return 0;
}

// Read the next entry into pending, crossing pages and resyncing past damage
// Sets ended at the end marker or at end of data
static void read_next_entry(hqlc_container_reader *reader) {
  for (;;) {
    if (reader->page_open && reader->page_frame_count == HQLC_CONTAINER_PAGE_FRAMES) {
      close_page(reader);
    }
    if ((reader->page_open || open_page(reader) == 0) && read_entry_length(reader) == 0) {
      return;
    }
    uint32_t page_index;
    if (scan_for_page(reader,
                      reader->read_offset,
                      UINT64_MAX,
                      0,
                      UINT32_MAX,
                      &reader->read_offset,
                      &page_index)) {
      reader->ended = true;
      return;
    }
    reader->page_open = false;
    reader->continuous = false;
    mark_damaged(reader);
  }
}

// Consume the pending frame into buf (NULL skips), then read one entry ahead
static int consume_frame(hqlc_container_reader *reader, uint8_t *buf) {
  if (read_bytes(reader, buf, reader->pending_len)) {
    reader->ended = true;
    return -1;
  }
  reader->page_frame_count++;
  read_next_entry(reader);
  return 0;
}

// Locate the APEv2 tag and the trailer, working back from end of file
static void parse_tail(hqlc_container_reader *reader) {
  uint64_t end = reader->io.size;
  uint8_t tail[32];
  uint8_t header[PAGE_HEADER_BYTES];
  uint32_t page_index;
  if (end >= HQLC_CONTAINER_HEADER_BYTES + 32 &&
      peek_bytes(reader, end - 32, tail, 32) == 32 && memcmp(tail, "APETAGEX", 8) == 0) {
    // The size field excludes the optional header
    uint64_t tag_size = read_le(tail + 12, 4);
    if (tail[23] & 0x80) {
      tag_size += 32; // the size field excludes the optional header
    }
    if (tag_size >= 32 && tag_size <= end - HQLC_CONTAINER_HEADER_BYTES) {
      end -= tag_size;
      reader->info.tag_offset = end;
      reader->info.tag_size = (uint32_t)tag_size;
    }
  }
  if (end < HQLC_CONTAINER_HEADER_BYTES + TRAILER_BYTES ||
      peek_bytes(reader, end - TRAILER_BYTES, tail, TRAILER_BYTES) != TRAILER_BYTES ||
      memcmp(tail + 16, "HQLT", 4) != 0) {
    return;
  }
  uint64_t last_page = read_le(tail + 8, 8);
  uint64_t total = read_le(tail, 8);
  if (last_page < HQLC_CONTAINER_HEADER_BYTES || last_page >= end - TRAILER_BYTES ||
      peek_bytes(reader, last_page, header, PAGE_HEADER_BYTES) != PAGE_HEADER_BYTES ||
      parse_page_header(reader, header, &page_index) ||
      total > ((uint64_t)page_index + HQLC_CONTAINER_PAGE_FRAMES) * HQLC_FRAME_SAMPLES) {
    return;
  }
  reader->last_page_offset = last_page;
  reader->last_page_index = page_index;
  reader->info.total_samples = total;
  reader->info.seekable = true;
}

hqlc_container_error hqlc_container_reader_open(hqlc_container_reader *reader,
                                                const hqlc_container_io *io) {
  if (!reader || !io || !io->read_at) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  memset(reader, 0, sizeof(*reader));
  reader->io = *io;
  reader->info.total_samples = HQLC_CONTAINER_UNKNOWN_SAMPLES;

  // File header, then the first page header, which fixes the stream cfg
  uint8_t head[HQLC_CONTAINER_HEADER_BYTES + PAGE_HEADER_BYTES];
  uint32_t index;
  size_t got = peek_bytes(reader, 0, head, sizeof(head));
  if (got < HQLC_CONTAINER_HEADER_BYTES || memcmp(head, "HQLC", 4) != 0) {
    return reader->io_error ? HQLC_CONTAINER_ERR_IO : HQLC_CONTAINER_ERR_FORMAT;
  }
  if (head[4] != HQLC_CONTAINER_VERSION) {
    return HQLC_CONTAINER_ERR_VERSION;
  }
  reader->cfg = head[HQLC_CONTAINER_HEADER_BYTES + 2];
  if (got < sizeof(head) ||
      parse_page_header(reader, head + HQLC_CONTAINER_HEADER_BYTES, &index)) {
    return HQLC_CONTAINER_ERR_FORMAT;
  }
  reader->info.channels = (uint8_t)((reader->cfg & CFG_CHANNELS_MASK) + 1);
  reader->info.bitstream_version = reader->cfg >> 4;
  reader->info.pre_skip = (uint16_t)read_le(head + 6, 2);
  reader->read_offset = HQLC_CONTAINER_HEADER_BYTES;
  reader->output_start = reader->info.pre_skip;
  reader->reset = true;
  if (io->size) {
    parse_tail(reader);
  }
  read_next_entry(reader);
  return HQLC_CONTAINER_OK;
}

const hqlc_container_info *
hqlc_container_reader_info(const hqlc_container_reader *reader) {
  return &reader->info;
}

hqlc_container_error hqlc_container_read_frame(hqlc_container_reader *reader,
                                               uint8_t *buf,
                                               size_t cap,
                                               hqlc_container_frame *frame) {
  if (!reader || !buf || !frame) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  if (reader->ended) {
    return reader->io_error ? HQLC_CONTAINER_ERR_IO : HQLC_CONTAINER_END;
  }
  if (reader->pending_len > cap) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  frame->index = (uint64_t)reader->page_frame_index + reader->page_frame_count;
  frame->len = reader->pending_len;
  frame->reset = reader->reset;
  frame->damaged = reader->damaged;
  reader->reset = false;
  reader->damaged = false;
  if (consume_frame(reader, buf)) {
    return reader->io_error ? HQLC_CONTAINER_ERR_IO : HQLC_CONTAINER_END;
  }
  // pad is nonzero only once the look-ahead has found the end marker
  uint64_t first = frame->index * HQLC_FRAME_SAMPLES;
  uint64_t skip = 0;
  if (reader->output_start > first) {
    skip = reader->output_start - first;
  }
  if (skip > HQLC_FRAME_SAMPLES) {
    skip = HQLC_FRAME_SAMPLES;
  }
  uint64_t keep = 0;
  if (skip + reader->pad < HQLC_FRAME_SAMPLES) {
    keep = HQLC_FRAME_SAMPLES - reader->pad - skip;
  }
  frame->skip = (uint16_t)skip;
  frame->keep = (uint16_t)keep;
  return HQLC_CONTAINER_OK;
}

hqlc_container_error hqlc_container_seek(hqlc_container_reader *reader, uint64_t sample) {
  if (!reader) {
    return HQLC_CONTAINER_ERR_INVALID_ARG;
  }
  if (!reader->info.seekable) {
    return HQLC_CONTAINER_ERR_NOT_SEEKABLE;
  }
  if (sample > reader->info.total_samples) {
    sample = reader->info.total_samples;
  }
  uint64_t target_sample = sample + reader->info.pre_skip;
  uint64_t target_frame = target_sample / HQLC_FRAME_SAMPLES;
  if (target_frame > 0) {
    target_frame--; // one frame of pre-roll, the MDCT overlap comes from frame N-1
  }

  // target page in the page units (frame * 64)
  uint32_t target_page =
      (uint32_t)(target_frame - target_frame % HQLC_CONTAINER_PAGE_FRAMES);

  // Find a page at most one page before the target page
  uint64_t lo_offset = HQLC_CONTAINER_HEADER_BYTES;
  uint64_t hi_offset = reader->last_page_offset;
  uint32_t lo_index = 0;
  uint32_t hi_index = reader->last_page_index;

  if (target_page >= hi_index) {
    lo_offset = hi_offset;
    lo_index = hi_index;
  }
  bool bisect = false;

  // Do bisection
  while (target_page - lo_index > HQLC_CONTAINER_PAGE_FRAMES &&
         hi_offset - lo_offset > 1) {
    uint64_t span = hi_offset - lo_offset;
    uint64_t found_offset;
    uint32_t found_index;

    // interpolate
    float frac = (float)(target_page - HQLC_CONTAINER_PAGE_FRAMES / 2 - lo_index) /
                 (float)(hi_index - lo_index);
    uint64_t guess = lo_offset + (uint64_t)(frac * (float)span);
    if (bisect) {
      guess = lo_offset + span / 2;
    }
    if (guess <= lo_offset) {
      guess = lo_offset + 1;
    } else if (guess >= hi_offset) {
      guess = hi_offset - 1;
    }
    if (scan_for_page(reader,
                      guess,
                      hi_offset,
                      lo_index + HQLC_CONTAINER_PAGE_FRAMES,
                      hi_index - HQLC_CONTAINER_PAGE_FRAMES,
                      &found_offset,
                      &found_index)) {
      hi_offset = guess; // no page between guess and hi_offset
    } else if (found_index <= target_page) {
      lo_offset = found_offset;
      lo_index = found_index;
    } else {
      hi_offset = found_offset;
      hi_index = found_index;
    }
    bisect = hi_offset - lo_offset > span / 2;
  }

  // Walk to the pre-roll frame, continuing the last probe's read
  reader->read_offset = lo_offset;
  reader->page_open = false;
  reader->ended = false;
  reader->pad = 0;
  reader->continuous = false;
  reader->output_start = target_sample;
  reader->reset = true;
  read_next_entry(reader);
  while (!reader->ended &&
         (uint64_t)reader->page_frame_index + reader->page_frame_count < target_frame) {
    consume_frame(reader, NULL);
  }
  return reader->io_error ? HQLC_CONTAINER_ERR_IO : HQLC_CONTAINER_OK;
}
