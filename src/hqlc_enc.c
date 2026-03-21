// hqlc_enc — Encode and decode a WAV file through the HQLC codec.
//
// Usage:
//   hqlc_enc input.wav output.wav [-b bitrate] [-g gain]
//
// Modes:
//   -b <bps>   Rate-controlled mode at the given bitrate (e.g. 128000)
//   -g <gain>  Fixed-gain mode (e.g. 2.0)
//
// If neither -b nor -g is given, defaults to -b 128000.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hqlc.h"

// ── Minimal WAV I/O (PCM16 only) ──────────────────────────────────────────

typedef struct {
  int channels;
  int sample_rate;
  int32_t n_frames; // total sample frames
  int16_t *data;    // interleaved
} wav_file;

static int wav_read(const char *path, wav_file *wf) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    return -1;
  }

  uint8_t riff[12];
  if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 ||
      memcmp(riff + 8, "WAVE", 4) != 0) {
    fclose(f);
    return -1;
  }

  int got_fmt = 0;
  int32_t data_size = 0;
  memset(wf, 0, sizeof(*wf));

  while (1) {
    uint8_t hdr[8];
    if (fread(hdr, 1, 8, f) != 8) {
      break;
    }

    uint32_t sz = (uint32_t)hdr[4] | ((uint32_t)hdr[5] << 8) | ((uint32_t)hdr[6] << 16) |
                  ((uint32_t)hdr[7] << 24);

    if (memcmp(hdr, "fmt ", 4) == 0) {
      uint8_t fmt[18];
      size_t rd = sz < 18 ? sz : 18;
      if (fread(fmt, 1, rd, f) != rd) {
        fclose(f);
        return -1;
      }
      wf->channels = fmt[2] | (fmt[3] << 8);
      wf->sample_rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
      int bps = fmt[14] | (fmt[15] << 8);
      if (bps != 16) {
        fprintf(stderr, "error: only 16-bit WAV supported (got %d-bit)\n", bps);
        fclose(f);
        return -1;
      }
      got_fmt = 1;
      if (sz > rd) {
        fseek(f, (long)(sz - rd), SEEK_CUR);
      }
    } else if (memcmp(hdr, "data", 4) == 0) {
      data_size = (int32_t)sz;
      break;
    } else {
      fseek(f, (long)sz, SEEK_CUR);
    }
  }

  if (!got_fmt || data_size <= 0) {
    fclose(f);
    return -1;
  }

  wf->n_frames = data_size / (wf->channels * 2);
  wf->data = (int16_t *)malloc((size_t)data_size);
  if (!wf->data) {
    fclose(f);
    return -1;
  }

  size_t got = fread(wf->data, 1, (size_t)data_size, f);
  fclose(f);
  if ((int32_t)got < data_size) {
    wf->n_frames = (int32_t)got / (wf->channels * 2);
  }

  return 0;
}

static int wav_write(
    const char *path, const int16_t *data, int n_frames, int channels, int sample_rate) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    return -1;
  }

  int32_t data_size = n_frames * channels * 2;
  int32_t file_size = 36 + data_size;
  int32_t byte_rate = sample_rate * channels * 2;
  int16_t block_align = (int16_t)(channels * 2);

  uint8_t hdr[44] = {0};
  memcpy(hdr, "RIFF", 4);
  hdr[4] = file_size & 0xFF;
  hdr[5] = (file_size >> 8) & 0xFF;
  hdr[6] = (file_size >> 16) & 0xFF;
  hdr[7] = (file_size >> 24) & 0xFF;
  memcpy(hdr + 8, "WAVE", 4);
  memcpy(hdr + 12, "fmt ", 4);
  hdr[16] = 16;
  hdr[20] = 1; // PCM
  hdr[22] = channels & 0xFF;
  hdr[23] = (channels >> 8) & 0xFF;
  hdr[24] = sample_rate & 0xFF;
  hdr[25] = (sample_rate >> 8) & 0xFF;
  hdr[26] = (sample_rate >> 16) & 0xFF;
  hdr[27] = (sample_rate >> 24) & 0xFF;
  hdr[28] = byte_rate & 0xFF;
  hdr[29] = (byte_rate >> 8) & 0xFF;
  hdr[30] = (byte_rate >> 16) & 0xFF;
  hdr[31] = (byte_rate >> 24) & 0xFF;
  hdr[32] = block_align & 0xFF;
  hdr[33] = (block_align >> 8) & 0xFF;
  hdr[34] = 16; // bits per sample
  memcpy(hdr + 36, "data", 4);
  hdr[40] = data_size & 0xFF;
  hdr[41] = (data_size >> 8) & 0xFF;
  hdr[42] = (data_size >> 16) & 0xFF;
  hdr[43] = (data_size >> 24) & 0xFF;

  fwrite(hdr, 1, 44, f);
  fwrite(data, 1, (size_t)data_size, f);
  fclose(f);
  return 0;
}

// ── Main ──────────────────────────────────────────────────────────────────

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s input.wav output.wav [-b bitrate] [-g gain]\n"
          "\n"
          "  -b <bps>   Rate-controlled mode (default: 128000)\n"
          "  -g <gain>  Fixed-gain mode (e.g. 2.0)\n",
          argv0);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[2];

  hqlc_mode mode = HQLC_MODE_RC;
  float gain = 0.0f;
  uint32_t bitrate = 128000;

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
      mode = HQLC_MODE_RC;
      bitrate = (uint32_t)atoi(argv[++i]);
    } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
      mode = HQLC_MODE_FIXED;
      gain = (float)atof(argv[++i]);
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  // Read input
  wav_file wf;
  if (wav_read(input_path, &wf) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", input_path);
    return 1;
  }
  if (wf.sample_rate != HQLC_SAMPLE_RATE) {
    fprintf(stderr,
            "error: sample rate must be %d (got %d)\n",
            HQLC_SAMPLE_RATE,
            wf.sample_rate);
    free(wf.data);
    return 1;
  }
  if (wf.channels < 1 || wf.channels > HQLC_MAX_CHANNELS) {
    fprintf(stderr, "error: unsupported channel count %d\n", wf.channels);
    free(wf.data);
    return 1;
  }

  int ch = wf.channels;
  int n_frames = wf.n_frames / HQLC_FRAME_SAMPLES;
  if (n_frames < 2) {
    fprintf(
        stderr, "error: file too short (need >= %d samples)\n", HQLC_FRAME_SAMPLES * 2);
    free(wf.data);
    return 1;
  }

  // Init encoder + decoder
  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());

  hqlc_encoder_config cfg = {
      .channels = (uint8_t)ch,
      .sample_rate = HQLC_SAMPLE_RATE,
      .mode = mode,
  };
  if (mode == HQLC_MODE_RC) {
    cfg.bitrate = bitrate;
  } else {
    cfg.gain = gain;
  }

  if (hqlc_encoder_init(enc, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    return 1;
  }
  if (hqlc_decoder_init(dec, (uint8_t)ch, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    return 1;
  }

  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  void *dec_scratch = calloc(1, hqlc_decoder_scratch_size());
  int16_t *pcm_out =
      (int16_t *)calloc((size_t)n_frames * HQLC_FRAME_SAMPLES * ch, sizeof(int16_t));
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];

  // Encode + decode frame by frame
  size_t total_bytes = 0;
  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&wf.data[f * HQLC_FRAME_SAMPLES * ch];
    size_t comp_len = 0;

    hqlc_error err = hqlc_encode_frame(
        enc, fp, HQLC_PCM16, compressed, HQLC_MAX_FRAME_BYTES, &comp_len, enc_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: encode failed at frame %d\n", f);
      return 1;
    }
    total_bytes += comp_len;

    uint8_t *dp = (uint8_t *)&pcm_out[f * HQLC_FRAME_SAMPLES * ch];
    err = hqlc_decode_frame(dec, compressed, comp_len, dp, HQLC_PCM16, dec_scratch);
    if (err != HQLC_OK) {
      fprintf(stderr, "error: decode failed at frame %d\n", f);
      return 1;
    }
  }

  // Trim 1-frame latency: decoded[1..n_frames) ≈ orig[0..n_frames-1)
  int out_frames = n_frames - 1;
  int16_t *trimmed = &pcm_out[HQLC_FRAME_SAMPLES * ch];

  if (wav_write(
          output_path, trimmed, out_frames * HQLC_FRAME_SAMPLES, ch, HQLC_SAMPLE_RATE) !=
      0) {
    fprintf(stderr, "error: cannot write '%s'\n", output_path);
    return 1;
  }

  // Stats
  float duration = (float)(out_frames * HQLC_FRAME_SAMPLES) / HQLC_SAMPLE_RATE;
  float avg_bitrate = (float)(total_bytes * 8) / duration;
  float input_bitrate = (float)(wf.sample_rate * ch * 16);
  float ratio = input_bitrate / avg_bitrate;

  printf("%s → %s\n", input_path, output_path);
  printf("  %d frames, %.2fs, %dch\n", n_frames, duration, ch);
  printf("  mode: %s", mode == HQLC_MODE_RC ? "RC" : "fixed");
  if (mode == HQLC_MODE_RC) {
    printf(" (target %u bps)", bitrate);
  } else {
    printf(" (gain %.2f)", gain);
  }
  printf("\n");
  printf("  avg bitrate: %.0f bps (%.1f:1)\n", avg_bitrate, ratio);

  free(wf.data);
  free(enc);
  free(dec);
  free(enc_scratch);
  free(dec_scratch);
  free(pcm_out);
  return 0;
}
