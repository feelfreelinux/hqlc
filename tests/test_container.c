#include "unity.h"
#include "hqlc.h"
#include "hqlc_container.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* Memory sink and source */

typedef struct {
  uint8_t *data;
  size_t len, cap;
} buf_t;

static void buf_put(buf_t *file, const void *data, size_t count) {
  if (file->len + count > file->cap) {
    file->cap = (file->len + count) * 2;
    file->data = (uint8_t *)realloc(file->data, file->cap);
  }
  memcpy(file->data + file->len, data, count);
  file->len += count;
}

typedef struct {
  const uint8_t *data;
  size_t len;
  int sequential;
  uint64_t next;
  int backwards; // reads that went backwards on a sequential source
} src_t;

static int src_read(void *ctx, uint64_t off, void *dst, size_t count, size_t *got) {
  src_t *source = (src_t *)ctx;
  if (source->sequential && off < source->next) {
    source->backwards++;
  }
  size_t size = off >= source->len
                    ? 0
                    : (source->len - off < count ? (size_t)(source->len - off) : count);
  if (size) {
    memcpy(dst, source->data + off, size);
  }
  *got = size;
  source->next = off + size;
  return 0;
}

static hqlc_container_error open_mem(hqlc_container_reader *reader,
                                     src_t *source,
                                     const buf_t *file,
                                     int sequential) {
  memset(source, 0, sizeof(*source));
  source->data = file->data;
  source->len = file->len;
  source->sequential = sequential;
  hqlc_container_io io = {src_read, source, sequential ? 0 : file->len};
  return hqlc_container_reader_open(reader, &io);
}

/* Reference CRCs, straight from the spec */

static uint16_t ref_crc16(const uint8_t *data, size_t size) {
  uint16_t value = 0;
  while (size--) {
    value ^= (uint16_t)(*data++ << 8);
    for (int i = 0; i < 8; i++) {
      value =
          (value & 0x8000) ? (uint16_t)((value << 1) ^ 0x8005) : (uint16_t)(value << 1);
    }
  }
  return value;
}

/* Synthetic streams */

// Rate-control-like sizes with occasional extremes, to exercise 2-byte deltas
static size_t synth_len(uint32_t i) {
  if (i % 97 == 5) {
    return HQLC_MAX_FRAME_BYTES;
  }
  if (i % 89 == 7) {
    return 1;
  }
  return 120 + (i * 37) % 90;
}

static void synth_frame(uint32_t i, uint8_t *data, size_t len) {
  for (size_t j = 0; j < len; j++) {
    data[j] = (uint8_t)(i * 31 + j * 7);
  }
}

// Write count synthetic frames, entry_off[i] gets the offset of frame i's length
static void synth_stream(
    buf_t *out, uint32_t count, uint16_t pre_skip, uint16_t pad, uint64_t *entry_off) {
  hqlc_container_writer *writer =
      (hqlc_container_writer *)malloc(hqlc_container_writer_size());
  uint8_t chunk[HQLC_CONTAINER_MAX_WRITE_BYTES];
  uint8_t frame[HQLC_MAX_FRAME_BYTES];
  size_t size;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_writer_init(writer, 2, pre_skip, chunk));
  buf_put(out, chunk, HQLC_CONTAINER_HEADER_BYTES);
  for (uint32_t i = 0; i < count; i++) {
    size_t len = synth_len(i);
    synth_frame(i, frame, len);
    if (entry_off) {
      entry_off[i] = out->len + (i % HQLC_CONTAINER_PAGE_FRAMES == 0 ? 7 : 0);
    }
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                      hqlc_container_write_frame(writer, frame, len, chunk, &size));
    buf_put(out, chunk, size);
  }
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_write_end(
                        writer, (uint64_t)count * 512 - pre_skip - pad, chunk, &size));
  buf_put(out, chunk, size);
  free(writer);
}

// Read every frame and check contents except frame corrupt.
// Returns the frame count and the kept samples.
static uint32_t
read_all(hqlc_container_reader *reader, uint64_t *kept, int *damaged_at, int corrupt) {
  uint8_t buf[HQLC_MAX_FRAME_BYTES], want[HQLC_MAX_FRAME_BYTES];
  hqlc_container_frame frame_info;
  uint32_t count = 0;
  *kept = 0;
  *damaged_at = -1;
  while (hqlc_container_read_frame(reader, buf, sizeof(buf), &frame_info) ==
         HQLC_CONTAINER_OK) {
    TEST_ASSERT_EQUAL(synth_len((uint32_t)frame_info.index), frame_info.len);
    synth_frame((uint32_t)frame_info.index, want, frame_info.len);
    if ((int)frame_info.index != corrupt) {
      TEST_ASSERT_EQUAL_MEMORY(want, buf, frame_info.len);
    }
    if (frame_info.damaged && *damaged_at < 0) {
      *damaged_at = (int)frame_info.index;
    }
    *kept += frame_info.keep;
    count++;
  }
  return count;
}

/* Tests */

void test_crc_and_layout(void) {
  TEST_ASSERT_EQUAL_HEX16(0xFEE8, ref_crc16((const uint8_t *)"123456789", 9));

  buf_t file = {0};
  synth_stream(&file, 2, 512, 7, NULL);
  const uint8_t *data = file.data;
  TEST_ASSERT_EQUAL_MEMORY("HQLC", data, 4);
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_VERSION, data[4]);
  TEST_ASSERT_EQUAL(512, data[6] | (data[7] << 8));

  // Page: header, plain first length, delta second length, end marker, CRC
  const uint8_t *page = data + 8;
  TEST_ASSERT_EQUAL_HEX8(0x68, page[0]);
  TEST_ASSERT_EQUAL_HEX8(0x71, page[1]);
  TEST_ASSERT_EQUAL_HEX8((HQLC_BITSTREAM_VERSION << 4) | 1, page[2]);
  size_t len0 = synth_len(0), len1 = synth_len(1);
  TEST_ASSERT_EQUAL(len0, page[7]); // < 128, one byte
  size_t second_entry = 8 + len0;
  int32_t delta = (int32_t)len1 - (int32_t)len0;
  TEST_ASSERT_EQUAL(delta >= 0 ? 2 * delta : -2 * delta - 1, page[second_entry]);
  size_t page_len =
      second_entry + 1 + len1 + 2 + 2; // ... frame, 2-byte marker delta, pad
  uint16_t crc = (uint16_t)(page[page_len] | (page[page_len + 1] << 8));
  TEST_ASSERT_EQUAL_HEX16(ref_crc16(page, page_len), crc);

  // Trailer
  const uint8_t *trailer = page + page_len + 2;
  TEST_ASSERT_EQUAL(file.len, (size_t)(trailer + 20 - data));
  TEST_ASSERT_EQUAL_MEMORY("HQLT", trailer + 16, 4);
  TEST_ASSERT_EQUAL(2 * 512 - 512 - 7, trailer[0] | (trailer[1] << 8));
  free(file.data);
}

void test_roundtrip_frame_counts(void) {
  static const uint32_t counts[] = {1, 2, 63, 64, 65, 127, 128, 129, 300};
  for (size_t idx = 0; idx < sizeof(counts) / sizeof(counts[0]); idx++) {
    uint32_t count = counts[idx];
    uint16_t pad = count > 1 ? 100 : 0;
    uint64_t total = (uint64_t)count * 512 - 512 - pad;
    buf_t file = {0};
    synth_stream(&file, count, 512, pad, NULL);

    for (int seq = 0; seq < 2; seq++) {
      hqlc_container_reader *reader =
          (hqlc_container_reader *)malloc(hqlc_container_reader_size());
      src_t source;
      TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &file, seq));
      const hqlc_container_info *info = hqlc_container_reader_info(reader);
      TEST_ASSERT_EQUAL(2, info->channels);
      TEST_ASSERT_EQUAL(HQLC_BITSTREAM_VERSION, info->bitstream_version);
      TEST_ASSERT_EQUAL(512, info->pre_skip);
      TEST_ASSERT_EQUAL(!seq, info->seekable);
      TEST_ASSERT_TRUE(info->total_samples ==
                       (seq ? HQLC_CONTAINER_UNKNOWN_SAMPLES : total));

      uint64_t kept;
      int damaged_at;
      TEST_ASSERT_EQUAL(count, read_all(reader, &kept, &damaged_at, -1));
      TEST_ASSERT_TRUE(kept == total);
      TEST_ASSERT_EQUAL(-1, damaged_at);
      TEST_ASSERT_EQUAL(0, source.backwards);
      free(reader);
    }
    free(file.data);
  }
}

void test_writer_rejects_bad_args(void) {
  hqlc_container_writer *writer =
      (hqlc_container_writer *)malloc(hqlc_container_writer_size());
  uint8_t chunk[HQLC_CONTAINER_MAX_WRITE_BYTES], f[1] = {0};
  size_t size;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_INVALID_ARG,
                    hqlc_container_writer_init(writer, 0, 512, chunk));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_INVALID_ARG,
                    hqlc_container_writer_init(writer, 17, 512, chunk));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, hqlc_container_writer_init(writer, 1, 512, chunk));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_INVALID_ARG,
                    hqlc_container_write_frame(writer, f, 0, chunk, &size));
  TEST_ASSERT_EQUAL(
      HQLC_CONTAINER_ERR_INVALID_ARG,
      hqlc_container_write_frame(writer, f, HQLC_MAX_FRAME_BYTES + 1, chunk, &size));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_write_frame(writer, f, 1, chunk, &size));
  // One frame covers pre_skip and nothing more
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_INVALID_ARG,
                    hqlc_container_write_end(writer, 1, chunk, &size));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_write_frame(writer, f, 1, chunk, &size));
  // Two frames leave a whole frame of padding for an empty input
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_INVALID_ARG,
                    hqlc_container_write_end(writer, 0, chunk, &size));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, hqlc_container_write_end(writer, 1, chunk, &size));
  free(writer);
}

static void put_le32(uint8_t *data, uint32_t v) {
  for (int i = 0; i < 4; i++) {
    data[i] = (uint8_t)(v >> (8 * i));
  }
}

// APEv2 header or footer
static void ape_hf(uint8_t *data, uint32_t size, uint32_t items, int header) {
  memset(data, 0, 32);
  memcpy(data, "APETAGEX", 8);
  put_le32(data + 8, 2000);
  put_le32(data + 12, size);
  put_le32(data + 16, items);
  put_le32(data + 20, (1u << 31) | (header ? 1u << 29 : 0));
}

void test_ape_tag_is_skipped(void) {
  buf_t audio = {0}, file = {0};
  synth_stream(&audio, 100, 512, 33, NULL);
  buf_put(&file, audio.data, audio.len);

  // APEv2 with one item
  uint8_t item[] = {1, 0, 0, 0, 0, 0, 0, 0, 'T', 'i', 't', 'l', 'e', 0, 'x'};
  uint8_t ape_frame[32];
  uint32_t size = (uint32_t)sizeof(item) + 32;
  uint64_t tag_at = file.len;
  ape_hf(ape_frame, size, 1, 1);
  buf_put(&file, ape_frame, 32);
  buf_put(&file, item, sizeof(item));
  ape_hf(ape_frame, size, 1, 0);
  buf_put(&file, ape_frame, 32);

  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  src_t source;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &file, 0));
  const hqlc_container_info *info = hqlc_container_reader_info(reader);
  TEST_ASSERT_EQUAL(1, info->seekable);
  TEST_ASSERT_TRUE(info->tag_offset == tag_at);
  TEST_ASSERT_EQUAL(size + 32, info->tag_size);
  uint64_t kept;
  int damaged_at;
  TEST_ASSERT_EQUAL(100, read_all(reader, &kept, &damaged_at, -1));
  TEST_ASSERT_TRUE(kept == 100 * 512 - 512 - 33);
  free(reader);
  free(audio.data);
  free(file.data);
}

void test_truncated_stream(void) {
  buf_t file = {0};
  synth_stream(&file, 300, 512, 0, NULL);
  size_t cuts[] = {file.len / 2, file.len - 21, file.len - 1};
  for (size_t idx = 0; idx < 3; idx++) {
    buf_t copy = {file.data, cuts[idx], cuts[idx]};
    hqlc_container_reader *reader =
        (hqlc_container_reader *)malloc(hqlc_container_reader_size());
    src_t source;
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &copy, 0));
    TEST_ASSERT_EQUAL(0, hqlc_container_reader_info(reader)->seekable);
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_NOT_SEEKABLE, hqlc_container_seek(reader, 1000));
    uint64_t kept;
    int damaged_at;
    uint32_t count = read_all(reader, &kept, &damaged_at, -1);
    // A cut inside the trailer still leaves every frame readable
    TEST_ASSERT_TRUE(idx == 0 ? count > 100 && count < 300 : count == 300);
    free(reader);
  }
  free(file.data);
}

void test_damage_recovery(void) {
  buf_t file = {0};
  uint64_t off[300];
  synth_stream(&file, 300, 512, 0, off);

  // Payload byte flipped in page 1: structure intact, CRC catches it at page end
  buf_t copy = {(uint8_t *)malloc(file.len), file.len, file.len};
  memcpy(copy.data, file.data, file.len);
  copy.data[off[70] + 5] ^= 0x40;
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  src_t source;
  uint64_t kept;
  int damaged_at;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &copy, 0));
  TEST_ASSERT_EQUAL(300, read_all(reader, &kept, &damaged_at, 70));
  TEST_ASSERT_EQUAL(128, damaged_at);

  // Broken length field in page 1: resync at page 2, frames 70..127 lost
  memcpy(copy.data, file.data, file.len);
  copy.data[off[70]] = 0xFF;
  copy.data[off[70] + 1] = 0xFF;
  for (int seq = 0; seq < 2; seq++) {
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &copy, seq));
    TEST_ASSERT_EQUAL(300 - 58, read_all(reader, &kept, &damaged_at, -1));
    TEST_ASSERT_EQUAL(128, damaged_at);
    TEST_ASSERT_EQUAL(0, source.backwards);
  }
  free(reader);
  free(copy.data);
  free(file.data);
}

void test_rejects_other_formats(void) {
  uint8_t legacy[64] = {'H', 'Q', 'L', 'C', 7, 2};
  uint8_t junk[64] = {'R', 'I', 'F', 'F'};
  buf_t b1 = {legacy, sizeof(legacy), sizeof(legacy)};
  buf_t b2 = {junk, sizeof(junk), sizeof(junk)};
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  src_t source;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_VERSION, open_mem(reader, &source, &b1, 0));
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_ERR_FORMAT, open_mem(reader, &source, &b2, 0));
  free(reader);
}

void test_seek_long_stream(void) {
  // 20size frames = 313 pages, so the page search actually runs
  const uint32_t count = 20000;
  buf_t file = {0};
  synth_stream(&file, count, 512, 200, NULL);
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  src_t source;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &file, 0));
  uint64_t total = hqlc_container_reader_info(reader)->total_samples;
  uint8_t buf[HQLC_MAX_FRAME_BYTES];
  hqlc_container_frame frame_info;
  uint32_t rng = 7;
  for (int i = 0; i < 200; i++) {
    rng = rng * 1664525u + 1013904223u;
    uint64_t pos = i == 0 ? total - 1 : rng % total;
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, hqlc_container_seek(reader, pos));
    uint64_t ds = pos + 512;
    // Pre-roll frame, then the target frame with the in-frame offset
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                      hqlc_container_read_frame(reader, buf, sizeof(buf), &frame_info));
    TEST_ASSERT_TRUE(frame_info.index == ds / 512 - 1);
    TEST_ASSERT_EQUAL(1, frame_info.reset);
    TEST_ASSERT_EQUAL(0, frame_info.keep);
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                      hqlc_container_read_frame(reader, buf, sizeof(buf), &frame_info));
    TEST_ASSERT_TRUE(frame_info.index == ds / 512);
    TEST_ASSERT_EQUAL(ds % 512, frame_info.skip);
    TEST_ASSERT_EQUAL(0, frame_info.reset);
  }
  free(reader);
  free(file.data);
}

/* Seeking with the real codec */

typedef struct {
  hqlc_decoder *dec;
  void *scratch;
} dec_t;

// Decode up to max output samples from the current position
static size_t decode_run(
    hqlc_container_reader *reader, dec_t *decoder, int16_t *out, size_t max, int ch) {
  uint8_t buf[HQLC_MAX_FRAME_BYTES];
  int16_t pcm[HQLC_FRAME_SAMPLES * 2];
  hqlc_container_frame frame_info;
  size_t count = 0;
  while (count < max && hqlc_container_read_frame(
                            reader, buf, sizeof(buf), &frame_info) == HQLC_CONTAINER_OK) {
    TEST_ASSERT_EQUAL(0, frame_info.damaged);
    if (frame_info.reset) {
      hqlc_decoder_reset(decoder->dec);
    }
    TEST_ASSERT_EQUAL(HQLC_OK,
                      hqlc_decode_frame(decoder->dec,
                                        buf,
                                        frame_info.len,
                                        (uint8_t *)pcm,
                                        HQLC_PCM16,
                                        decoder->scratch));
    size_t size = frame_info.keep < max - count ? frame_info.keep : max - count;
    memcpy(out + count * ch, pcm + frame_info.skip * ch, size * ch * sizeof(int16_t));
    count += size;
  }
  return count;
}

void test_seek_is_sample_exact(void) {
  const int ch = 2;
  const size_t n_in = 3 * 48000 + 123;
  const size_t n_frames = (n_in + 512 + 511) / 512;

  // Tone plus noise, zero-padded to whole frames
  int16_t *pcm = (int16_t *)calloc(n_frames * 512 * ch, sizeof(int16_t));
  uint32_t rng = 1;
  for (size_t i = 0; i < n_in; i++) {
    rng = rng * 1664525u + 1013904223u;
    float v =
        0.3f * sinf(0.05f * (float)i) + 0.1f * ((float)(rng >> 16) / 65536.0f - 0.5f);
    pcm[i * ch] = (int16_t)(v * 32767.0f);
    pcm[i * ch + 1] = (int16_t)(-v * 20000.0f);
  }

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  hqlc_encoder_config cfg = {0};
  cfg.channels = (uint8_t)ch;
  cfg.sample_rate = HQLC_SAMPLE_RATE;
  cfg.mode = HQLC_MODE_RC;
  cfg.bitrate = 128000;
  TEST_ASSERT_EQUAL(HQLC_OK, hqlc_encoder_init(enc, &cfg));

  buf_t file = {0};
  hqlc_container_writer *writer =
      (hqlc_container_writer *)malloc(hqlc_container_writer_size());
  uint8_t chunk[HQLC_CONTAINER_MAX_WRITE_BYTES], frame[HQLC_MAX_FRAME_BYTES];
  size_t size, len;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_writer_init(writer, (uint8_t)ch, 512, chunk));
  buf_put(&file, chunk, HQLC_CONTAINER_HEADER_BYTES);
  for (size_t f = 0; f < n_frames; f++) {
    TEST_ASSERT_EQUAL(HQLC_OK,
                      hqlc_encode_frame(enc,
                                        (const uint8_t *)(pcm + f * 512 * ch),
                                        HQLC_PCM16,
                                        frame,
                                        sizeof(frame),
                                        &len,
                                        enc_scratch));
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                      hqlc_container_write_frame(writer, frame, len, chunk, &size));
    buf_put(&file, chunk, size);
  }
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK,
                    hqlc_container_write_end(writer, n_in, chunk, &size));
  buf_put(&file, chunk, size);
  free(writer);

  dec_t decoder = {(hqlc_decoder *)calloc(1, hqlc_decoder_size()),
                   calloc(1, hqlc_decoder_scratch_size())};
  TEST_ASSERT_EQUAL(HQLC_OK,
                    hqlc_decoder_init(decoder.dec, (uint8_t)ch, HQLC_SAMPLE_RATE));

  // Linear reference decode
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  src_t source;
  TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, open_mem(reader, &source, &file, 0));
  TEST_ASSERT_TRUE(hqlc_container_reader_info(reader)->total_samples == n_in);
  int16_t *ref = (int16_t *)calloc(n_in * ch, sizeof(int16_t));
  TEST_ASSERT_EQUAL(n_in, decode_run(reader, &decoder, ref, n_in, ch));

  // Seek test
  uint64_t pos[40] = {0,
                      1,
                      511,
                      512,
                      513,
                      1023,
                      1024,
                      64 * 512 - 513,
                      64 * 512 - 512,
                      64 * 512,
                      128 * 512 + 7,
                      n_in - 1,
                      n_in};
  size_t n_pos = 13;
  while (n_pos < 40) {
    rng = rng * 1664525u + 1013904223u; // pseudo random positions
    pos[n_pos++] = rng % n_in;
  }
  const size_t span = 3000;
  int16_t *got = (int16_t *)calloc(span * ch, sizeof(int16_t));
  for (size_t i = 0; i < n_pos; i++) {
    TEST_ASSERT_EQUAL(HQLC_CONTAINER_OK, hqlc_container_seek(reader, pos[i]));
    size_t want = n_in - pos[i] < span ? n_in - pos[i] : span;
    TEST_ASSERT_EQUAL(want, decode_run(reader, &decoder, got, span, ch));
    if (want) {
      TEST_ASSERT_EQUAL_MEMORY(ref + pos[i] * ch, got, want * ch * sizeof(int16_t));
    }
  }

  free(reader);
  free(got);
  free(ref);
  free(decoder.dec);
  free(decoder.scratch);
  free(file.data);
  free(enc);
  free(enc_scratch);
  free(pcm);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_crc_and_layout);
  RUN_TEST(test_roundtrip_frame_counts);
  RUN_TEST(test_writer_rejects_bad_args);
  RUN_TEST(test_ape_tag_is_skipped);
  RUN_TEST(test_truncated_stream);
  RUN_TEST(test_damage_recovery);
  RUN_TEST(test_rejects_other_formats);
  RUN_TEST(test_seek_long_stream);
  RUN_TEST(test_seek_is_sample_exact);
  return UNITY_END();
}
