// hqlc — HQLC codec CLI tool.
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

#include "hqlc.h"

// ── HQLC container format ────────────────────────────────────────────────
//
// Header (16 bytes):
//   [0..3]   magic "HQLC"
//   [4]      version (1)
//   [5]      channels (1 or 2)
//   [6..7]   reserved
//   [8..11]  sample_rate (LE32)
//   [12..15] n_frames (LE32)
//
// Per frame:
//   [0..1]   payload_len (LE16)
//   [2..]    payload

#define HQLC_FILE_VERSION 1
#define HQLC_FILE_HDR_SIZE 16

// ── Helpers ──────────────────────────────────────────────────────────────

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
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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

// Read entire file or stdin into a malloc'd buffer.
static uint8_t *read_all(const char *path, size_t *out_len) {
  FILE *f;
  if (is_stdio(path)) {
    set_binary_mode(stdin);
    f = stdin;
  } else {
    f = fopen(path, "rb");
  }
  if (!f) return NULL;

  size_t cap = 1 << 20, len = 0;
  uint8_t *buf = (uint8_t *)malloc(cap);
  if (!buf) { if (!is_stdio(path)) fclose(f); return NULL; }

  while (1) {
    if (len == cap) {
      cap *= 2;
      uint8_t *tmp = (uint8_t *)realloc(buf, cap);
      if (!tmp) { free(buf); if (!is_stdio(path)) fclose(f); return NULL; }
      buf = tmp;
    }
    size_t n = fread(buf + len, 1, cap - len, f);
    len += n;
    if (n == 0) break;
  }

  if (!is_stdio(path)) fclose(f);
  *out_len = len;
  return buf;
}

// Read WAV from file or stdin, converting to interleaved s16.
static int read_wav(const char *path, int *out_ch, int32_t *out_n_frames, int16_t **out_data) {
  uint8_t *mem = NULL;
  size_t mem_len = 0;
  drwav wav;
  int ok;

  if (is_stdio(path)) {
    mem = read_all(path, &mem_len);
    if (!mem) return -1;
    ok = drwav_init_memory(&wav, mem, mem_len, NULL);
  } else {
    ok = drwav_init_file(&wav, path, NULL);
  }
  if (!ok) { free(mem); return -1; }

  if (wav.sampleRate != HQLC_SAMPLE_RATE) {
    fprintf(stderr,
            "error: sample rate must be %d Hz (got %u Hz)\n"
            "  hint: resample with  ffmpeg -i input -ar 48000 output.wav\n",
            HQLC_SAMPLE_RATE, wav.sampleRate);
    drwav_uninit(&wav);
    free(mem);
    return -1;
  }
  if (wav.channels < 1 || wav.channels > HQLC_MAX_CHANNELS) {
    fprintf(stderr, "error: unsupported channel count %u\n", wav.channels);
    drwav_uninit(&wav);
    free(mem);
    return -1;
  }

  *out_ch = (int)wav.channels;
  *out_n_frames = (int32_t)wav.totalPCMFrameCount;
  *out_data = (int16_t *)malloc((size_t)*out_n_frames * *out_ch * sizeof(int16_t));
  if (!*out_data) { drwav_uninit(&wav); free(mem); return -1; }

  drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, *out_data);
  drwav_uninit(&wav);
  free(mem);
  return 0;
}

// Write interleaved s16 PCM to WAV file or stdout.
static int write_wav(const char *path, const int16_t *data, int32_t n_pcm_frames, int ch) {
  drwav_data_format fmt = {0};
  fmt.container = drwav_container_riff;
  fmt.format = DR_WAVE_FORMAT_PCM;
  fmt.channels = (drwav_uint32)ch;
  fmt.sampleRate = HQLC_SAMPLE_RATE;
  fmt.bitsPerSample = 16;

  drwav wav;
  if (is_stdio(path)) {
    // Write to memory buffer, then dump to stdout
    void *wav_buf = NULL;
    size_t wav_len = 0;
    if (!drwav_init_memory_write(&wav, &wav_buf, &wav_len, &fmt, NULL)) return -1;
    drwav_write_pcm_frames(&wav, (drwav_uint64)n_pcm_frames, data);
    drwav_uninit(&wav);
    set_binary_mode(stdout);
    fwrite(wav_buf, 1, wav_len, stdout);
    fflush(stdout);
    drwav_free(wav_buf, NULL);
  } else {
    if (!drwav_init_file_write(&wav, path, &fmt, NULL)) return -1;
    drwav_write_pcm_frames(&wav, (drwav_uint64)n_pcm_frames, data);
    drwav_uninit(&wav);
  }
  return 0;
}

// ── Encode: WAV → .hqlc ─────────────────────────────────────────────────

static int do_encode(const char *in, const char *out,
                     hqlc_mode mode, uint32_t bitrate, float gain) {
  int ch;
  int32_t total_pcm;
  int16_t *pcm;
  if (read_wav(in, &ch, &total_pcm, &pcm) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    return 1;
  }

  int n_frames = total_pcm / HQLC_FRAME_SAMPLES;
  if (n_frames < 1) {
    fprintf(stderr, "error: file too short (need >= %d samples)\n", HQLC_FRAME_SAMPLES);
    free(pcm);
    return 1;
  }

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_encoder_config cfg = {
      .channels = (uint8_t)ch, .sample_rate = HQLC_SAMPLE_RATE, .mode = mode,
  };
  if (mode == HQLC_MODE_RC) cfg.bitrate = bitrate;
  else cfg.gain = gain;

  if (hqlc_encoder_init(enc, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    free(pcm); free(enc);
    return 1;
  }
  void *scratch = calloc(1, hqlc_encoder_scratch_size());

  // Open output and write header (n_frames is known upfront)
  FILE *fout;
  if (is_stdio(out)) {
    set_binary_mode(stdout);
    fout = stdout;
  } else {
    fout = fopen(out, "wb");
  }
  if (!fout) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    free(pcm); free(enc); free(scratch);
    return 1;
  }

  uint8_t hdr[HQLC_FILE_HDR_SIZE] = {0};
  memcpy(hdr, "HQLC", 4);
  hdr[4] = HQLC_FILE_VERSION;
  hdr[5] = (uint8_t)ch;
  put_le32(hdr + 8, HQLC_SAMPLE_RATE);
  put_le32(hdr + 12, (uint32_t)n_frames);
  fwrite(hdr, 1, HQLC_FILE_HDR_SIZE, fout);

  // Encode frame by frame, streaming to output
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];
  size_t total_bytes = 0;
  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&pcm[f * HQLC_FRAME_SAMPLES * ch];
    size_t comp_len = 0;
    hqlc_error err = hqlc_encode_frame(
        enc, fp, HQLC_PCM16, compressed, HQLC_MAX_FRAME_BYTES, &comp_len, scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: encode failed at frame %d\n", f);
      if (!is_stdio(out)) fclose(fout);
      free(pcm); free(enc); free(scratch);
      return 1;
    }
    total_bytes += comp_len;
    uint8_t frame_hdr[2];
    put_le16(frame_hdr, (uint16_t)comp_len);
    fwrite(frame_hdr, 1, 2, fout);
    fwrite(compressed, 1, comp_len, fout);
  }
  if (!is_stdio(out)) fclose(fout);

  float duration = (float)(n_frames * HQLC_FRAME_SAMPLES) / HQLC_SAMPLE_RATE;
  float avg_bps = (float)(total_bytes * 8) / duration;
  float raw_bps = (float)(HQLC_SAMPLE_RATE * ch * 16);
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %d frames, %.2fs, %dch\n", n_frames, duration, ch);
  fprintf(stderr, "  avg bitrate: %.0f bps (%.1f:1)\n", avg_bps, raw_bps / avg_bps);

  free(pcm); free(enc); free(scratch);
  return 0;
}

// ── Decode: .hqlc → WAV ─────────────────────────────────────────────────

static int do_decode(const char *in, const char *out) {
  size_t file_len;
  uint8_t *file_data = read_all(in, &file_len);
  if (!file_data) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    return 1;
  }

  if (file_len < HQLC_FILE_HDR_SIZE || memcmp(file_data, "HQLC", 4) != 0) {
    fprintf(stderr, "error: '%s' is not a valid .hqlc file\n", in);
    free(file_data);
    return 1;
  }
  if (file_data[4] != HQLC_FILE_VERSION) {
    fprintf(stderr, "error: unsupported .hqlc version %d\n", file_data[4]);
    free(file_data);
    return 1;
  }

  int ch = file_data[5];
  uint32_t sample_rate = get_le32(file_data + 8);
  uint32_t n_frames = get_le32(file_data + 12);

  if (sample_rate != HQLC_SAMPLE_RATE) {
    fprintf(stderr, "error: unexpected sample rate %u in .hqlc file\n", sample_rate);
    free(file_data);
    return 1;
  }
  if (ch < 1 || ch > HQLC_MAX_CHANNELS) {
    fprintf(stderr, "error: invalid channel count %d in .hqlc file\n", ch);
    free(file_data);
    return 1;
  }
  if (n_frames < 1) {
    fprintf(stderr, "error: empty .hqlc file\n");
    free(file_data);
    return 1;
  }

  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());
  if (hqlc_decoder_init(dec, (uint8_t)ch, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    free(file_data); free(dec);
    return 1;
  }
  void *scratch = calloc(1, hqlc_decoder_scratch_size());
  int16_t *pcm_out =
      (int16_t *)calloc((size_t)n_frames * HQLC_FRAME_SAMPLES * ch, sizeof(int16_t));

  size_t pos = HQLC_FILE_HDR_SIZE;
  size_t total_bytes = 0;
  for (uint32_t f = 0; f < n_frames; f++) {
    if (pos + 2 > file_len) {
      fprintf(stderr, "error: truncated .hqlc at frame %u\n", f);
      free(file_data); free(dec); free(scratch); free(pcm_out);
      return 1;
    }
    uint16_t frame_len = get_le16(file_data + pos);
    pos += 2;
    if (pos + frame_len > file_len) {
      fprintf(stderr, "error: truncated .hqlc at frame %u\n", f);
      free(file_data); free(dec); free(scratch); free(pcm_out);
      return 1;
    }

    uint8_t *dp = (uint8_t *)&pcm_out[f * HQLC_FRAME_SAMPLES * ch];
    hqlc_error err =
        hqlc_decode_frame(dec, file_data + pos, frame_len, dp, HQLC_PCM16, scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: decode failed at frame %u\n", f);
      free(file_data); free(dec); free(scratch); free(pcm_out);
      return 1;
    }
    total_bytes += frame_len;
    pos += frame_len;
  }

  // Trim 1-frame decoder latency
  uint32_t out_frames = n_frames - 1;
  int16_t *trimmed = &pcm_out[HQLC_FRAME_SAMPLES * ch];

  if (write_wav(out, trimmed, (int32_t)(out_frames * HQLC_FRAME_SAMPLES), ch) != 0) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    free(file_data); free(dec); free(scratch); free(pcm_out);
    return 1;
  }

  float duration = (float)(out_frames * HQLC_FRAME_SAMPLES) / HQLC_SAMPLE_RATE;
  float avg_bps = duration > 0 ? (float)(total_bytes * 8) / duration : 0;
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %u frames, %.2fs, %dch\n", n_frames, duration, ch);
  fprintf(stderr, "  avg bitrate: %.0f bps\n", avg_bps);

  free(file_data); free(dec); free(scratch); free(pcm_out);
  return 0;
}

// ── Roundtrip: WAV → encode → decode → WAV ──────────────────────────────

// Write one JSONL line of diagnostic data for a frame
static void diag_write_frame(FILE *f, const hqlc_frame_diag *d, int frame, int ch) {
  fprintf(f, "{\"f\":%d,\"gc\":%d,\"bits\":%d,\"tgt\":%d,\"res\":%d,\"quiet\":%s,",
          frame, d->gain_code, d->frame_bits, d->target_bpf, (int)d->res_bits,
          d->quiet_frame ? "true" : "false");
  fprintf(f, "\"exp\":[");
  for (int c = 0; c < ch; c++) {
    fprintf(f, "[");
    for (int b = 0; b < HQLC_DIAG_MAX_BANDS; b++) {
      fprintf(f, "%d%s", (int)d->exp_indices[c * HQLC_DIAG_MAX_BANDS + b],
              b < HQLC_DIAG_MAX_BANDS - 1 ? "," : "");
    }
    fprintf(f, "]%s", c < ch - 1 ? "," : "");
  }
  fprintf(f, "],\"nf\":[");
  for (int c = 0; c < ch; c++)
    fprintf(f, "%d%s", d->noise_factors[c], c < ch - 1 ? "," : "");
  fprintf(f, "],\"tns\":[");
  for (int c = 0; c < ch; c++)
    fprintf(f, "%d%s", d->tns_orders[c], c < ch - 1 ? "," : "");
  fprintf(f, "]}\n");
}

static int do_roundtrip(const char *in, const char *out,
                        hqlc_mode mode, uint32_t bitrate, float gain,
                        const char *diag_path) {
  int ch;
  int32_t total_pcm;
  int16_t *pcm;
  if (read_wav(in, &ch, &total_pcm, &pcm) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", in);
    return 1;
  }

  int n_frames = total_pcm / HQLC_FRAME_SAMPLES;
  if (n_frames < 2) {
    fprintf(stderr, "error: file too short (need >= %d samples)\n", HQLC_FRAME_SAMPLES * 2);
    free(pcm);
    return 1;
  }

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());

  hqlc_encoder_config cfg = {
      .channels = (uint8_t)ch, .sample_rate = HQLC_SAMPLE_RATE, .mode = mode,
  };
  if (mode == HQLC_MODE_RC) cfg.bitrate = bitrate;
  else cfg.gain = gain;

  if (hqlc_encoder_init(enc, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    free(pcm); free(enc); free(dec);
    return 1;
  }
  if (hqlc_decoder_init(dec, (uint8_t)ch, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    free(pcm); free(enc); free(dec);
    return 1;
  }

  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  void *dec_scratch = calloc(1, hqlc_decoder_scratch_size());
  int16_t *pcm_out =
      (int16_t *)calloc((size_t)n_frames * HQLC_FRAME_SAMPLES * ch, sizeof(int16_t));
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];

  FILE *diag_f = NULL;
  if (diag_path) {
    diag_f = fopen(diag_path, "w");
    if (!diag_f) {
      fprintf(stderr, "error: cannot open diag file '%s'\n", diag_path);
      free(pcm); free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(pcm_out);
      return 1;
    }
    fprintf(diag_f, "{\"v\":1,\"codec\":\"hqlc-c\",\"br\":%u,\"ch\":%d,\"n\":%d,\"sr\":%d}\n",
            bitrate, ch, n_frames, HQLC_SAMPLE_RATE);
  }

  size_t total_bytes = 0;
  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&pcm[f * HQLC_FRAME_SAMPLES * ch];
    size_t comp_len = 0;

    hqlc_error err = hqlc_encode_frame(
        enc, fp, HQLC_PCM16, compressed, HQLC_MAX_FRAME_BYTES, &comp_len, enc_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: encode failed at frame %d\n", f);
      if (diag_f) fclose(diag_f);
      free(pcm); free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(pcm_out);
      return 1;
    }
    total_bytes += comp_len;

    if (diag_f) {
      diag_write_frame(diag_f, hqlc_encoder_get_diag(enc), f, ch);
    }

    uint8_t *dp = (uint8_t *)&pcm_out[f * HQLC_FRAME_SAMPLES * ch];
    err = hqlc_decode_frame(dec, compressed, comp_len, dp, HQLC_PCM16, dec_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: decode failed at frame %d\n", f);
      if (diag_f) fclose(diag_f);
      free(pcm); free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(pcm_out);
      return 1;
    }
  }
  if (diag_f) fclose(diag_f);

  // Trim 1-frame latency: decoded[1..n_frames) ≈ orig[0..n_frames-1)
  int out_frames = n_frames - 1;
  int16_t *trimmed = &pcm_out[HQLC_FRAME_SAMPLES * ch];

  if (write_wav(out, trimmed, out_frames * HQLC_FRAME_SAMPLES, ch) != 0) {
    fprintf(stderr, "error: cannot write '%s'\n", out);
    free(pcm); free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(pcm_out);
    return 1;
  }

  float duration = (float)(out_frames * HQLC_FRAME_SAMPLES) / HQLC_SAMPLE_RATE;
  float avg_bps = (float)(total_bytes * 8) / duration;
  float raw_bps = (float)(HQLC_SAMPLE_RATE * ch * 16);
  fprintf(stderr, "%s -> %s\n", in, out);
  fprintf(stderr, "  %d frames, %.2fs, %dch\n", n_frames, duration, ch);
  fprintf(stderr, "  mode: %s", mode == HQLC_MODE_RC ? "RC" : "fixed");
  if (mode == HQLC_MODE_RC) fprintf(stderr, " (target %u bps)", bitrate);
  else fprintf(stderr, " (gain %.2f)", gain);
  fprintf(stderr, "\n");
  fprintf(stderr, "  avg bitrate: %.0f bps (%.1f:1)\n", avg_bps, raw_bps / avg_bps);

  free(pcm); free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(pcm_out);
  return 0;
}

// ── Main ─────────────────────────────────────────────────────────────────

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
          "  --diag <path> Write per-frame diagnostic trace (JSONL)\n"
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
  const char *diag_path = NULL;

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
    } else if (strcmp(argv[i], "--diag") == 0 && i + 1 < argc) {
      diag_path = argv[++i];
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
    case OP_ENCODE:    return do_encode(input_path, output_path, mode, bitrate, gain);
    case OP_DECODE:    return do_decode(input_path, output_path);
    case OP_ROUNDTRIP: return do_roundtrip(input_path, output_path, mode, bitrate, gain, diag_path);
  }
  return 1;
}
