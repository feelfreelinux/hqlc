// hqlc - HQLC codec CLI tool.
//
// Usage:
//   hqlc [options] input output
//
// Modes:
//   (default)  Roundtrip: encode+decode WAV for quality evaluation
//   -e         Encode WAV to .hqlc bitstream
//   -d         Decode .hqlc bitstream to WAV
//
// Encoder options:
//   -b <bps>   Rate-controlled mode (default: 128000)
//   -g <gain>  Fixed-gain mode (e.g. 2.0)
//
// Use "-" for stdin/stdout. Stats are printed to stderr.
//
// Audio is streamed one block at a time in every mode, so peak memory is a
// fixed ~100 KB no matter how long the input is.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "entropy.h"
#include "hqlc.h"

#ifdef HQLC_TRAIN_TABLES
// Optional path for dumping the rANS table-training histogram (-T <file>).
static const char *g_train_path = NULL;
#endif

/* HQLC container format */
//
// Header (16 bytes):
//   [0..3]   magic "HQLC"
//   [4]      version (6)
//   [5]      channels (1 or 2)
//   [6..7]   zero padding in the last audio frame (LE16, 0..511)
//   [8..11]  sample_rate (LE32)
//   [12..15] (LE32), including the trailing flush frame
//
// Per frame:
//   [0..1]   payload_len (LE16)
//   [2..]    payload

#define HQLC_FILE_VERSION  6
#define HQLC_FILE_HDR_SIZE 16

// Buffered frames per block, 32 x 512 x 2ch x 2B = 64 KB
#define PCM_BLOCK_FRAMES 32

/* Helpers */

static void put_le16(uint8_t *p, uint16_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
}

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
  p[2] = (v >> 16) & 0xFF;
  p[3] = (v >> 24) & 0xFF;
}

static uint16_t get_le16(const uint8_t *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static int is_stdio(const char *path) {
  return path[0] == '-' && path[1] == '\0';
}

static void set_binary_mode(FILE *f) {
#ifdef _WIN32
  _setmode(_fileno(f), _O_BINARY);
#else
  (void)f;
#endif
}

// Open a raw byte stream (the .hqlc side), or stdin/stdout for "-".
static FILE *raw_open(const char *path, int for_write) {
  FILE *f;
  if (is_stdio(path)) {
    f = for_write ? stdout : stdin;
    set_binary_mode(f);
  } else {
    f = fopen(path, for_write ? "wb" : "rb");
  }
  if (f) {
    // Adjust the buffer
    setvbuf(f, NULL, _IOFBF, 1 << 18);
  }
  return f;
}

static void raw_close(FILE *f, const char *path) {
  if (f && !is_stdio(path)) {
    fclose(f);
  }
}

/* Streaming byte source/sink for dr_wav */
typedef struct {
  FILE *f;
  const char *path;
  int seekable;
  int64_t pos;
  uint8_t *tee;
  size_t tee_len;
  size_t tee_cap;
  int tee_on;
} byte_stream;

static int byte_stream_open(byte_stream *s, const char *path, int for_write) {
  s->f = raw_open(path, for_write);
  s->path = path;
  if (!s->f) {
    return -1;
  }
  s->seekable = (fseek(s->f, 0, SEEK_CUR) == 0);
  s->pos = s->seekable ? ftell(s->f) : 0;
  return 0;
}

static void byte_stream_close(byte_stream *s) {
  raw_close(s->f, s->path);
  s->f = NULL;
  free(s->tee);
  s->tee = NULL;
  s->tee_len = s->tee_cap = 0;
  s->tee_on = 0;
}

static void byte_stream_tee(byte_stream *s, const void *data, size_t n) {
  if (s->tee_len + n > s->tee_cap) {
    size_t cap = s->tee_cap ? s->tee_cap : (size_t)1 << 16;
    while (cap < s->tee_len + n) {
      cap *= 2;
    }
    uint8_t *p = (uint8_t *)realloc(s->tee, cap);
    if (!p) {
      s->tee_on = 0; // give up on the fallback rather than on the stream
      return;
    }
    s->tee = p;
    s->tee_cap = cap;
  }
  memcpy(s->tee + s->tee_len, data, n);
  s->tee_len += n;
}

static size_t byte_stream_read(void *ud, void *out, size_t n) {
  byte_stream *s = (byte_stream *)ud;
  size_t got = fread(out, 1, n, s->f);
  s->pos += (int64_t)got;
  if (s->tee_on && got > 0) {
    byte_stream_tee(s, out, got);
  }
  return got;
}

// Pull whatever is left of the stream into the tee buffer.
static int byte_stream_slurp(byte_stream *s) {
  uint8_t chunk[1 << 16];
  while (s->tee_on) {
    size_t got = byte_stream_read(s, chunk, sizeof(chunk));
    if (got < sizeof(chunk)) {
      break;
    }
  }
  return s->tee_on ? 0 : -1;
}

static size_t byte_stream_write(void *ud, const void *data, size_t n) {
  byte_stream *s = (byte_stream *)ud;
  size_t put = fwrite(data, 1, n, s->f);
  s->pos += (int64_t)put;
  return put;
}

static drwav_bool32 byte_stream_seek(void *ud, int offset, drwav_seek_origin origin) {
  byte_stream *s = (byte_stream *)ud;
  if (s->seekable) {
    int whence = SEEK_SET;
    if (origin == DRWAV_SEEK_CUR) {
      whence = SEEK_CUR;
    } else if (origin == DRWAV_SEEK_END) {
      whence = SEEK_END;
    }
    if (fseek(s->f, offset, whence) != 0) {
      return DRWAV_FALSE;
    }
    s->pos = ftell(s->f);
    return DRWAV_TRUE;
  }

  // Forward-only, skip by reading and discarding
  if (origin == DRWAV_SEEK_END) {
    return DRWAV_FALSE;
  }
  int64_t target = (origin == DRWAV_SEEK_SET) ? offset : s->pos + offset;
  if (target < s->pos) {
    return DRWAV_FALSE;
  }
  uint8_t sink[4096];
  while (s->pos < target) {
    int64_t want = target - s->pos;
    if (want > (int64_t)sizeof(sink)) {
      want = (int64_t)sizeof(sink);
    }
    // Through byte_stream_read so skipped chunks still reach the tee
    if (byte_stream_read(s, sink, (size_t)want) != (size_t)want) {
      return DRWAV_FALSE;
    }
  }
  return DRWAV_TRUE;
}

static drwav_bool32 byte_stream_tell(void *ud, drwav_int64 *cursor) {
  *cursor = (drwav_int64)((byte_stream *)ud)->pos;
  return DRWAV_TRUE;
}

/* Block-streamed WAV input */
typedef struct {
  byte_stream io;
  drwav wav;
  int ch;
  int64_t total_pcm; // PCM frames declared by the WAV header
  int32_t buf_pos;   // next codec frame to hand out of buf
  int32_t buf_frames;
  int16_t buf[(size_t)PCM_BLOCK_FRAMES * HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS];
} wav_reader;

static int wav_reader_open(wav_reader *r, const char *path) {
  memset(r, 0, sizeof(*r));
  if (byte_stream_open(&r->io, path, 0) != 0) {
    return -1;
  }
  // Tee the header while parsing so an unsized WAV on a pipe can be retried from memory
  r->io.tee_on = !r->io.seekable;
  int ok = drwav_init(
      &r->wav, byte_stream_read, byte_stream_seek, byte_stream_tell, &r->io, NULL);
  if (!ok && r->io.tee_on && byte_stream_slurp(&r->io) == 0) {
    ok = drwav_init_memory(&r->wav, r->io.tee, r->io.tee_len, NULL);
  }
  r->io.tee_on = 0;
  if (!ok) {
    byte_stream_close(&r->io);
    return -1;
  }
  if (r->wav.sampleRate != HQLC_SAMPLE_RATE) {
    fprintf(stderr,
            "error: sample rate must be %d Hz (got %u Hz)\n"
            "  hint: resample with  ffmpeg -i input -ar 48000 output.wav\n",
            HQLC_SAMPLE_RATE,
            r->wav.sampleRate);
    drwav_uninit(&r->wav);
    byte_stream_close(&r->io);
    return -1;
  }
  if (r->wav.channels < 1 || r->wav.channels > HQLC_MAX_CHANNELS) {
    fprintf(stderr, "error: unsupported channel count %u\n", r->wav.channels);
    drwav_uninit(&r->wav);
    byte_stream_close(&r->io);
    return -1;
  }
  r->ch = (int)r->wav.channels;
  r->total_pcm = (int64_t)r->wav.totalPCMFrameCount;
  return 0;
}

static void wav_reader_close(wav_reader *r) {
  drwav_uninit(&r->wav);
  byte_stream_close(&r->io);
}

// Hand out the next codec frame of interleaved s16, zero-filled past the end
// of the input. That zero fill is also what feeds the encoder's flush frame.
static const int16_t *wav_reader_next(wav_reader *r) {
  if (r->buf_pos == r->buf_frames) {
    size_t want = (size_t)PCM_BLOCK_FRAMES * HQLC_FRAME_SAMPLES;
    drwav_uint64 got = drwav_read_pcm_frames_s16(&r->wav, want, r->buf);
    if (got < want) {
      size_t filled = (size_t)got * (size_t)r->ch;
      memset(r->buf + filled, 0, (want * (size_t)r->ch - filled) * sizeof(int16_t));
    }
    r->buf_frames = PCM_BLOCK_FRAMES;
    r->buf_pos = 0;
  }
  return &r->buf[(size_t)(r->buf_pos++) * HQLC_FRAME_SAMPLES * r->ch];
}

/* Streamed WAV output */

typedef struct {
  byte_stream io;
  drwav wav;
  int64_t remaining; // PCM frames still wanted; the rest is trailing padding
} wav_writer;

static int wav_writer_open(wav_writer *w, const char *path, int ch, int64_t n_pcm) {
  memset(w, 0, sizeof(*w));
  if (byte_stream_open(&w->io, path, 1) != 0) {
    return -1;
  }
  drwav_data_format fmt = {0};

  // Use RF64 over 4GB
  fmt.container = (uint64_t)n_pcm * (uint64_t)ch * 2 > 0xFFFFFF00u ? drwav_container_rf64
                                                                   : drwav_container_riff;
  fmt.format = DR_WAVE_FORMAT_PCM;
  fmt.channels = (drwav_uint32)ch;
  fmt.sampleRate = HQLC_SAMPLE_RATE;
  fmt.bitsPerSample = 16;

  // Sequential: the frame count is known upfront, so the header never needs
  // to be backpatched and stdout works like any other sink.
  if (!drwav_init_write_sequential_pcm_frames(
          &w->wav, &fmt, (drwav_uint64)n_pcm, byte_stream_write, &w->io, NULL)) {
    byte_stream_close(&w->io);
    return -1;
  }
  w->remaining = n_pcm;
  return 0;
}

static int wav_writer_push(wav_writer *w, const int16_t *pcm, int64_t n) {
  if (n > w->remaining) {
    n = w->remaining;
  }
  if (n <= 0) {
    return 0;
  }
  if (drwav_write_pcm_frames(&w->wav, (drwav_uint64)n, pcm) != (drwav_uint64)n) {
    return -1;
  }
  w->remaining -= n;
  return 0;
}

static void wav_writer_close(wav_writer *w) {
  drwav_uninit(&w->wav);
  if (w->io.f) {
    fflush(w->io.f);
  }
  byte_stream_close(&w->io);
}

/* Codec state, allocated as one small bundle */

typedef struct {
  hqlc_encoder *enc;
  hqlc_decoder *dec;
  void *enc_scratch;
  void *dec_scratch;
} codec_mem;

static void codec_mem_free(codec_mem *m) {
  free(m->enc);
  free(m->dec);
  free(m->enc_scratch);
  free(m->dec_scratch);
  memset(m, 0, sizeof(*m));
}

static int codec_mem_alloc(codec_mem *m, int want_enc, int want_dec) {
  memset(m, 0, sizeof(*m));
  if (want_enc) {
    m->enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
    m->enc_scratch = calloc(1, hqlc_encoder_scratch_size());
    if (!m->enc || !m->enc_scratch) {
      goto oom;
    }
  }
  if (want_dec) {
    m->dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());
    m->dec_scratch = calloc(1, hqlc_decoder_scratch_size());
    if (!m->dec || !m->dec_scratch) {
      goto oom;
    }
  }
  return 0;
oom:
  fprintf(stderr, "error: out of memory\n");
  codec_mem_free(m);
  return -1;
}

static void fill_encoder_config(
    hqlc_encoder_config *cfg, int ch, hqlc_mode mode, uint32_t bitrate, float gain) {
  memset(cfg, 0, sizeof(*cfg));
  cfg->channels = (uint8_t)ch;
  cfg->sample_rate = HQLC_SAMPLE_RATE;
  cfg->mode = mode;
  if (mode == HQLC_MODE_RC) {
    cfg->bitrate = bitrate;
  } else {
    cfg->gain = gain;
  }
}

// Frame counts land in an LE32 container field.
static int frame_count_fits(int64_t n_enc) {
  if (n_enc > (int64_t)UINT32_MAX) {
    fprintf(stderr, "error: input too long\n");
    return 0;
  }
  return 1;
}

/* Encode: WAV to .hqlc */

static int
do_encode(const char *in, const char *out, hqlc_mode mode, uint32_t bitrate, float gain) {
  wav_reader *r = (wav_reader *)malloc(sizeof(*r));
  codec_mem mem = {0};
  FILE *fout = NULL;
  int ret = 1;

  if (!r) {
    fprintf(stderr, "error: out of memory\n");
    return 1;
  }
  if (wav_reader_open(r, in) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    free(r);
    return 1;
  }

  int ch = r->ch;
  int64_t total_pcm = r->total_pcm;
  int64_t n_frames = (total_pcm + HQLC_FRAME_SAMPLES - 1) / HQLC_FRAME_SAMPLES;
  if (n_frames < 1) {
    fprintf(stderr, "error: empty input\n");
    goto cleanup;
  }
  int64_t n_enc = n_frames + 1;
  if (!frame_count_fits(n_enc)) {
    goto cleanup;
  }
  int pad = (int)(n_frames * HQLC_FRAME_SAMPLES - total_pcm);

  if (codec_mem_alloc(&mem, 1, 0) != 0) {
    goto cleanup;
  }
  hqlc_encoder_config cfg;
  fill_encoder_config(&cfg, ch, mode, bitrate, gain);
  if (hqlc_encoder_init(mem.enc, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    goto cleanup;
  }

#ifdef HQLC_TRAIN_TABLES
  if (g_train_path) {
    rans_train_reset();
  }
#endif

  // Open output and write header (n_frames is known upfront)
  fout = raw_open(out, 1);
  if (!fout) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    goto cleanup;
  }

  uint8_t hdr[HQLC_FILE_HDR_SIZE] = {0};
  memcpy(hdr, "HQLC", 4);
  hdr[4] = HQLC_FILE_VERSION;
  hdr[5] = (uint8_t)ch;
  put_le16(hdr + 6, (uint16_t)pad);
  put_le32(hdr + 8, HQLC_SAMPLE_RATE);
  put_le32(hdr + 12, (uint32_t)n_enc);
  fwrite(hdr, 1, HQLC_FILE_HDR_SIZE, fout);

  // Encode block by block, streaming both sides
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];
  uint64_t total_bytes = 0;
  for (int64_t f = 0; f < n_enc; f++) {
    const int16_t *fp = wav_reader_next(r);
    size_t comp_len = 0;
    hqlc_error err = hqlc_encode_frame(mem.enc,
                                       (const uint8_t *)fp,
                                       HQLC_PCM16,
                                       compressed,
                                       HQLC_MAX_FRAME_BYTES,
                                       &comp_len,
                                       mem.enc_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: encode failed at frame %lld\n", (long long)f);
      goto cleanup;
    }
    total_bytes += comp_len;
    uint8_t frame_hdr[2];
    put_le16(frame_hdr, (uint16_t)comp_len);
    if (fwrite(frame_hdr, 1, 2, fout) != 2 ||
        fwrite(compressed, 1, comp_len, fout) != comp_len) {
      fprintf(stderr, "error: cannot write '%s'\n", out);
      goto cleanup;
    }
  }
  fflush(fout);

#ifdef HQLC_TRAIN_TABLES
  // Dump the per-table symbol histogram for table training
  if (g_train_path) {
    FILE *th = fopen(g_train_path, "w");
    if (th) {
      for (int t = 0; t < RANS_COEF_NTABLES; t++) {
        for (int sym = 0; sym < RANS_MAX_SYM; sym++) {
          fprintf(th,
                  "%llu%c",
                  (unsigned long long)rans_train_hist[t][sym],
                  sym == RANS_MAX_SYM - 1 ? '\n' : ' ');
        }
      }
      fclose(th);
    } else {
      fprintf(stderr, "warning: cannot write training histogram '%s'\n", g_train_path);
    }
  }
#endif

  float duration = (float)total_pcm / HQLC_SAMPLE_RATE;
  float avg_bps = (float)(total_bytes * 8) / duration;
  float raw_bps = (float)(HQLC_SAMPLE_RATE * ch * 16);
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %lld frames, %.2fs, %dch\n", (long long)n_enc, duration, ch);
  fprintf(stderr, "  avg bitrate: %.0f bps (%.1f:1)\n", avg_bps, raw_bps / avg_bps);
  ret = 0;

cleanup:
  raw_close(fout, out);
  codec_mem_free(&mem);
  wav_reader_close(r);
  free(r);
  return ret;
}

/* Decode: .hqlc to WAV */

static int do_decode(const char *in, const char *out) {
  FILE *fin = raw_open(in, 0);
  codec_mem mem = {0};
  wav_writer w = {0};
  int have_writer = 0;
  int ret = 1;

  if (!fin) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    return 1;
  }

  uint8_t hdr[HQLC_FILE_HDR_SIZE];
  if (fread(hdr, 1, HQLC_FILE_HDR_SIZE, fin) != HQLC_FILE_HDR_SIZE ||
      memcmp(hdr, "HQLC", 4) != 0) {
    fprintf(stderr, "error: '%s' is not a valid .hqlc file\n", in);
    goto cleanup;
  }
  if (hdr[4] != HQLC_FILE_VERSION) {
    fprintf(stderr, "error: unsupported .hqlc version %d\n", hdr[4]);
    goto cleanup;
  }

  int ch = hdr[5];
  uint32_t pad = get_le16(hdr + 6);
  uint32_t sample_rate = get_le32(hdr + 8);
  uint32_t n_frames = get_le32(hdr + 12);

  if (sample_rate != HQLC_SAMPLE_RATE) {
    fprintf(stderr, "error: unexpected sample rate %u in .hqlc file\n", sample_rate);
    goto cleanup;
  }
  if (ch < 1 || ch > HQLC_MAX_CHANNELS) {
    fprintf(stderr, "error: invalid channel count %d in .hqlc file\n", ch);
    goto cleanup;
  }
  // n_frames counts the trailing flush frame, so audio needs at least two
  if (n_frames < 2 || pad >= HQLC_FRAME_SAMPLES) {
    fprintf(stderr, "error: empty or malformed .hqlc file\n");
    goto cleanup;
  }

  if (codec_mem_alloc(&mem, 0, 1) != 0) {
    goto cleanup;
  }
  if (hqlc_decoder_init(mem.dec, (uint8_t)ch, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    goto cleanup;
  }

  // Trim the 1-frame decoder latency at the head and the encoder's zero
  // padding at the tail; the flush frame carries the last real audio frame
  int64_t out_pcm = (int64_t)(n_frames - 1) * HQLC_FRAME_SAMPLES - pad;
  if (wav_writer_open(&w, out, ch, out_pcm) != 0) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    goto cleanup;
  }
  have_writer = 1;

  uint8_t payload[HQLC_MAX_FRAME_BYTES];
  int16_t pcm[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS];
  uint64_t total_bytes = 0;
  for (uint32_t f = 0; f < n_frames; f++) {
    uint8_t frame_hdr[2];
    if (fread(frame_hdr, 1, 2, fin) != 2) {
      fprintf(stderr, "error: truncated .hqlc at frame %u\n", f);
      goto cleanup;
    }
    uint16_t frame_len = get_le16(frame_hdr);
    if (frame_len > HQLC_MAX_FRAME_BYTES) {
      fprintf(stderr, "error: oversized frame %u in .hqlc file\n", f);
      goto cleanup;
    }
    if (fread(payload, 1, frame_len, fin) != frame_len) {
      fprintf(stderr, "error: truncated .hqlc at frame %u\n", f);
      goto cleanup;
    }

    hqlc_error err = hqlc_decode_frame(
        mem.dec, payload, frame_len, (uint8_t *)pcm, HQLC_PCM16, mem.dec_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: decode failed at frame %u\n", f);
      goto cleanup;
    }
    total_bytes += frame_len;

    // Frame 0 is the decoder's priming output, not audio
    if (f > 0 && wav_writer_push(&w, pcm, HQLC_FRAME_SAMPLES) != 0) {
      fprintf(stderr, "error: cannot write '%s'\n", out);
      goto cleanup;
    }
  }

  float duration = (float)out_pcm / HQLC_SAMPLE_RATE;
  float avg_bps = duration > 0 ? (float)(total_bytes * 8) / duration : 0;
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %u frames, %.2fs, %dch\n", n_frames, duration, ch);
  fprintf(stderr, "  avg bitrate: %.0f bps\n", avg_bps);
  ret = 0;

cleanup:
  if (have_writer) {
    wav_writer_close(&w);
  }
  codec_mem_free(&mem);
  raw_close(fin, in);
  return ret;
}

/* Roundtrip: encode then decode a WAV */

static int do_roundtrip(
    const char *in, const char *out, hqlc_mode mode, uint32_t bitrate, float gain) {
  wav_reader *r = (wav_reader *)malloc(sizeof(*r));
  codec_mem mem = {0};
  wav_writer w = {0};
  int have_writer = 0;
  int ret = 1;

  if (!r) {
    fprintf(stderr, "error: out of memory\n");
    return 1;
  }
  if (wav_reader_open(r, in) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    free(r);
    return 1;
  }

  int ch = r->ch;
  int64_t total_pcm = r->total_pcm;
  // Round up: a partial trailing frame is zero-padded rather than dropped
  int64_t n_frames = (total_pcm + HQLC_FRAME_SAMPLES - 1) / HQLC_FRAME_SAMPLES;
  if (n_frames < 1) {
    fprintf(stderr, "error: empty input\n");
    goto cleanup;
  }
  int64_t n_enc = n_frames + 1;
  if (!frame_count_fits(n_enc)) {
    goto cleanup;
  }

  if (codec_mem_alloc(&mem, 1, 1) != 0) {
    goto cleanup;
  }
  hqlc_encoder_config cfg;
  fill_encoder_config(&cfg, ch, mode, bitrate, gain);
  if (hqlc_encoder_init(mem.enc, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    goto cleanup;
  }
  if (hqlc_decoder_init(mem.dec, (uint8_t)ch, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    goto cleanup;
  }

  // Trim 1-frame latency: decoded[1..n_enc) ~= orig[0..n_frames), and the
  // flush frame brings the last real frame out, so this is the whole input
  if (wav_writer_open(&w, out, ch, total_pcm) != 0) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    goto cleanup;
  }
  have_writer = 1;

  uint8_t compressed[HQLC_MAX_FRAME_BYTES];
  int16_t pcm_out[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS];
  uint64_t total_bytes = 0;
  for (int64_t f = 0; f < n_enc; f++) {
    const int16_t *fp = wav_reader_next(r);
    size_t comp_len = 0;

    hqlc_error err = hqlc_encode_frame(mem.enc,
                                       (const uint8_t *)fp,
                                       HQLC_PCM16,
                                       compressed,
                                       HQLC_MAX_FRAME_BYTES,
                                       &comp_len,
                                       mem.enc_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: encode failed at frame %lld\n", (long long)f);
      goto cleanup;
    }
    total_bytes += comp_len;

    err = hqlc_decode_frame(
        mem.dec, compressed, comp_len, (uint8_t *)pcm_out, HQLC_PCM16, mem.dec_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: decode failed at frame %lld\n", (long long)f);
      goto cleanup;
    }

    if (f > 0 && wav_writer_push(&w, pcm_out, HQLC_FRAME_SAMPLES) != 0) {
      fprintf(stderr, "error: cannot write '%s'\n", out);
      goto cleanup;
    }
  }

  float duration = (float)total_pcm / HQLC_SAMPLE_RATE;
  float avg_bps = (float)(total_bytes * 8) / duration;
  float raw_bps = (float)(HQLC_SAMPLE_RATE * ch * 16);
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %lld frames, %.2fs, %dch\n", (long long)n_enc, duration, ch);
  fprintf(stderr, "  mode: %s", mode == HQLC_MODE_RC ? "RC" : "fixed");
  if (mode == HQLC_MODE_RC) {
    fprintf(stderr, " (target %u bps)", bitrate);
  } else {
    fprintf(stderr, " (gain %.2f)", gain);
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "  avg bitrate: %.0f bps (%.1f:1)\n", avg_bps, raw_bps / avg_bps);
  ret = 0;

cleanup:
  if (have_writer) {
    wav_writer_close(&w);
  }
  codec_mem_free(&mem);
  wav_reader_close(r);
  free(r);
  return ret;
}

/* Main */

enum op { OP_ROUNDTRIP, OP_ENCODE, OP_DECODE };

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s [options] input output\n"
          "\n"
          "Modes:\n"
          "  (default)  Roundtrip: encode+decode WAV for quality evaluation\n"
          "  -e         Encode WAV to .hqlc bitstream\n"
          "  -d         Decode .hqlc bitstream to WAV\n"
          "\n"
          "Encoder options:\n"
          "  -b <bps>      Rate-controlled mode (default: 128000)\n"
          "  -g <gain>     Fixed-gain mode\n"
#ifdef HQLC_TRAIN_TABLES
          "  -T <file>     Dump rANS table-training histogram while encoding\n"
#endif
          "\n"
          "Use \"-\" for stdin/stdout.\n",
          argv0);
}

int main(int argc, char **argv) {
  enum op op = OP_ROUNDTRIP;
  hqlc_mode mode = HQLC_MODE_RC;
  uint32_t bitrate = 128000;
  float gain = 0.0f;
  const char *input_path = NULL;
  const char *output_path = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-e") == 0) {
      op = OP_ENCODE;
    } else if (strcmp(argv[i], "-d") == 0) {
      op = OP_DECODE;
    } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
      mode = HQLC_MODE_RC;
      bitrate = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
      mode = HQLC_MODE_FIXED;
      gain = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
#ifdef HQLC_TRAIN_TABLES
      op = OP_ENCODE;
      g_train_path = argv[++i];
#else
      fprintf(stderr, "error: -T requires a build with -DHQLC_TRAIN_TABLES=ON\n");
      return 1;
#endif
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else if (!input_path) {
      input_path = argv[i];
    } else if (!output_path) {
      output_path = argv[i];
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  if (!input_path || !output_path) {
    usage(argv[0]);
    return 1;
  }

  switch (op) {
  case OP_ENCODE:
    return do_encode(input_path, output_path, mode, bitrate, gain);
  case OP_DECODE:
    return do_decode(input_path, output_path);
  case OP_ROUNDTRIP:
    return do_roundtrip(input_path, output_path, mode, bitrate, gain);
  }
  return 1;
}
