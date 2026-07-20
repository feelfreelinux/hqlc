// WASM glue — exposes roundtrip functions for HQLC, Opus, and MP3.
//
// All functions take interleaved int16 PCM at 48 kHz, return the number
// of output sample frames (per channel), or negative on error.

#include <stdlib.h>
#include <string.h>

#include "hqlc.h"
#include <opus.h>
#include <lame.h>

#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

#include <emscripten/emscripten.h>

// ── HQLC roundtrip ───────────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int32_t roundtrip_hqlc(const int16_t *pcm_in, int16_t *pcm_out,
                       int32_t n_samples, int32_t channels, int32_t bitrate) {
  int n_frames = n_samples / HQLC_FRAME_SAMPLES;
  if (n_frames < 2) return -1;

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());
  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  void *dec_scratch = calloc(1, hqlc_decoder_scratch_size());
  int16_t *buf = (int16_t *)calloc((size_t)n_frames * HQLC_FRAME_SAMPLES * channels,
                                   sizeof(int16_t));
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];

  if (!enc || !dec || !enc_scratch || !dec_scratch || !buf) goto fail;

  hqlc_encoder_config cfg = {
      .channels = (uint8_t)channels,
      .sample_rate = HQLC_SAMPLE_RATE,
      .mode = HQLC_MODE_RC,
      .bitrate = (uint32_t)bitrate,
  };
  if (hqlc_encoder_init(enc, &cfg) != HQLC_OK) goto fail;
  if (hqlc_decoder_init(dec, (uint8_t)channels, HQLC_SAMPLE_RATE) != HQLC_OK) goto fail;

  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&pcm_in[f * HQLC_FRAME_SAMPLES * channels];
    size_t comp_len = 0;
    if (hqlc_encode_frame(enc, fp, HQLC_PCM16, compressed, HQLC_MAX_FRAME_BYTES,
                          &comp_len, enc_scratch) != HQLC_OK)
      goto fail;
    uint8_t *dp = (uint8_t *)&buf[f * HQLC_FRAME_SAMPLES * channels];
    if (hqlc_decode_frame(dec, compressed, comp_len, dp, HQLC_PCM16,
                          dec_scratch) != HQLC_OK)
      goto fail;
  }

  // Trim 1-frame latency
  int out_frames = n_frames - 1;
  int32_t out_samples = out_frames * HQLC_FRAME_SAMPLES;
  memcpy(pcm_out, &buf[HQLC_FRAME_SAMPLES * channels],
         (size_t)out_samples * channels * sizeof(int16_t));

  free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(buf);
  return out_samples;

fail:
  free(enc); free(dec); free(enc_scratch); free(dec_scratch); free(buf);
  return -1;
}

// ── Opus roundtrip ───────────────────────────────────────────────────────

#define OPUS_FRAME_SAMPLES 960 // 20ms at 48kHz

EMSCRIPTEN_KEEPALIVE
int32_t roundtrip_opus(const int16_t *pcm_in, int16_t *pcm_out,
                       int32_t n_samples, int32_t channels, int32_t bitrate) {
  int err;
  OpusEncoder *enc = opus_encoder_create(48000, channels, OPUS_APPLICATION_AUDIO, &err);
  if (err != OPUS_OK || !enc) return -1;
  OpusDecoder *dec = opus_decoder_create(48000, channels, &err);
  if (err != OPUS_OK || !dec) { opus_encoder_destroy(enc); return -1; }

  opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));

  // Query encoder lookahead for latency trimming
  int lookahead = 0;
  opus_encoder_ctl(enc, OPUS_GET_LOOKAHEAD(&lookahead));

  int n_frames = n_samples / OPUS_FRAME_SAMPLES;
  if (n_frames < 1) { opus_encoder_destroy(enc); opus_decoder_destroy(dec); return -1; }

  // Decode into temporary buffer, then trim lookahead
  int max_decoded = n_frames * OPUS_FRAME_SAMPLES;
  int16_t *tmp = (int16_t *)malloc((size_t)max_decoded * channels * sizeof(int16_t));
  unsigned char pkt[8192]; // Opus packet buffer
  if (!tmp) { opus_encoder_destroy(enc); opus_decoder_destroy(dec); return -1; }

  int decoded_pos = 0;
  for (int f = 0; f < n_frames; f++) {
    const int16_t *frame_in = &pcm_in[f * OPUS_FRAME_SAMPLES * channels];
    int pkt_len = opus_encode(enc, frame_in, OPUS_FRAME_SAMPLES, pkt, sizeof(pkt));
    if (pkt_len < 0) {
      free(tmp); opus_encoder_destroy(enc); opus_decoder_destroy(dec);
      return -1;
    }
    int dec_samples = opus_decode(dec, pkt, pkt_len,
                                  &tmp[decoded_pos * channels],
                                  OPUS_FRAME_SAMPLES, 0);
    if (dec_samples < 0) {
      free(tmp); opus_encoder_destroy(enc); opus_decoder_destroy(dec);
      return -1;
    }
    decoded_pos += dec_samples;
  }

  // Trim lookahead from the start
  int out_samples = decoded_pos - lookahead;
  if (out_samples <= 0) {
    free(tmp); opus_encoder_destroy(enc); opus_decoder_destroy(dec);
    return -1;
  }
  memcpy(pcm_out, &tmp[lookahead * channels],
         (size_t)out_samples * channels * sizeof(int16_t));

  free(tmp);
  opus_encoder_destroy(enc);
  opus_decoder_destroy(dec);
  return out_samples;
}

// ── MP3 (LAME encode + minimp3 decode) roundtrip ─────────────────────────

EMSCRIPTEN_KEEPALIVE
int32_t roundtrip_mp3(const int16_t *pcm_in, int16_t *pcm_out,
                      int32_t n_samples, int32_t channels, int32_t bitrate) {
  lame_t lame = lame_init();
  if (!lame) return -1;

  lame_set_in_samplerate(lame, 48000);
  lame_set_out_samplerate(lame, 48000);
  lame_set_num_channels(lame, channels);
  if (channels == 1) lame_set_mode(lame, MONO);
  else lame_set_mode(lame, JOINT_STEREO);
  lame_set_brate(lame, bitrate / 1000);
  lame_set_quality(lame, 2); // high quality
  lame_set_bWriteVbrTag(lame, 0);
  if (lame_init_params(lame) < 0) { lame_close(lame); return -1; }

  int encoder_delay = lame_get_encoder_delay(lame);

  // Encode to MP3 buffer
  size_t mp3_cap = (size_t)(1.25 * n_samples + 7200) * 2;
  unsigned char *mp3_buf = (unsigned char *)malloc(mp3_cap);
  if (!mp3_buf) { lame_close(lame); return -1; }

  int mp3_len;
  if (channels == 2) {
    mp3_len = lame_encode_buffer_interleaved(
        lame, (short *)pcm_in, n_samples, mp3_buf, (int)mp3_cap);
  } else {
    mp3_len = lame_encode_buffer(
        lame, pcm_in, pcm_in, n_samples, mp3_buf, (int)mp3_cap);
  }
  if (mp3_len < 0) { free(mp3_buf); lame_close(lame); return -1; }

  int flush_len = lame_encode_flush(lame, mp3_buf + mp3_len, (int)(mp3_cap - mp3_len));
  if (flush_len > 0) mp3_len += flush_len;
  lame_close(lame);

  // Decode with minimp3
  mp3dec_t mp3d;
  mp3dec_init(&mp3d);

  int16_t frame_buf[MINIMP3_MAX_SAMPLES_PER_FRAME];
  int decoded_total = 0;
  int offset = 0;

  // First pass: count output samples
  mp3dec_t mp3d_count;
  mp3dec_init(&mp3d_count);
  int count_total = 0;
  int count_off = 0;
  while (count_off < mp3_len) {
    mp3dec_frame_info_t info;
    int samples = mp3dec_decode_frame(&mp3d_count, mp3_buf + count_off,
                                      mp3_len - count_off, frame_buf, &info);
    if (info.frame_bytes == 0) break;
    count_off += info.frame_bytes;
    count_total += samples;
  }

  // Allocate temp buffer and do real decode
  int16_t *tmp = (int16_t *)malloc((size_t)count_total * channels * sizeof(int16_t));
  if (!tmp) { free(mp3_buf); return -1; }

  while (offset < mp3_len) {
    mp3dec_frame_info_t info;
    int samples = mp3dec_decode_frame(&mp3d, mp3_buf + offset,
                                      mp3_len - offset, frame_buf, &info);
    if (info.frame_bytes == 0) break;
    offset += info.frame_bytes;
    if (samples > 0) {
      memcpy(&tmp[decoded_total * channels], frame_buf,
             (size_t)samples * channels * sizeof(int16_t));
      decoded_total += samples;
    }
  }
  free(mp3_buf);

  // Trim encoder delay from the start
  int out_samples = decoded_total - encoder_delay;
  if (out_samples > n_samples) out_samples = n_samples;
  if (out_samples <= 0) { free(tmp); return -1; }

  memcpy(pcm_out, &tmp[encoder_delay * channels],
         (size_t)out_samples * channels * sizeof(int16_t));
  free(tmp);
  return out_samples;
}
