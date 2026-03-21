#include "unity.h"
#include "hqlc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static void gen_sine_pcm16(
    int16_t *buf, int n_samples, int channels, float freq_hz, float amplitude) {
  for (int i = 0; i < n_samples; i++) {
    float t = (float)i / (float)HQLC_SAMPLE_RATE;
    float val = amplitude * sinf(2.0f * 3.14159265f * freq_hz * t);
    int32_t s = (int32_t)(val * 32767.0f);
    if (s > 32767) {
      s = 32767;
    }
    if (s < -32768) {
      s = -32768;
    }
    for (int ch = 0; ch < channels; ch++) {
      buf[i * channels + ch] = (int16_t)s;
    }
  }
}

// SNR in dB with latency compensation
static float compute_snr(const int16_t *orig,
                         int orig_start,
                         const int16_t *decoded,
                         int dec_start,
                         int n_compare,
                         int channels) {
  double signal_pow = 0.0, noise_pow = 0.0;
  for (int i = 0; i < n_compare; i++) {
    for (int ch = 0; ch < channels; ch++) {
      double s = (double)orig[(orig_start + i) * channels + ch];
      double d = (double)decoded[(dec_start + i) * channels + ch];
      signal_pow += s * s;
      noise_pow += (s - d) * (s - d);
    }
  }
  if (noise_pow < 1.0) {
    noise_pow = 1.0;
  }
  return (float)(10.0 * log10(signal_pow / noise_pow));
}

static int16_t *codec_roundtrip(const int16_t *pcm_orig,
                                int n_frames,
                                int channels,
                                hqlc_mode mode,
                                float gain,
                                uint32_t bitrate) {
  int n_samples = n_frames * HQLC_FRAME_SAMPLES;
  int16_t *pcm_dec = (int16_t *)calloc((size_t)n_samples * channels, sizeof(int16_t));

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());

  hqlc_encoder_config cfg = {
      .channels = (uint8_t)channels,
      .sample_rate = HQLC_SAMPLE_RATE,
      .mode = mode,
  };
  if (mode == HQLC_MODE_RC) {
    cfg.bitrate = bitrate;
  } else {
    cfg.gain = gain;
  }
  hqlc_encoder_init(enc, &cfg);
  hqlc_decoder_init(dec, (uint8_t)channels, HQLC_SAMPLE_RATE);

  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  void *dec_scratch = calloc(1, hqlc_decoder_scratch_size());
  uint8_t compressed[HQLC_MAX_FRAME_BYTES];

  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&pcm_orig[f * HQLC_FRAME_SAMPLES * channels];
    size_t comp_len = 0;
    hqlc_encode_frame(
        enc, fp, HQLC_PCM16, compressed, HQLC_MAX_FRAME_BYTES, &comp_len, enc_scratch);
    uint8_t *dp = (uint8_t *)&pcm_dec[f * HQLC_FRAME_SAMPLES * channels];
    hqlc_decode_frame(dec, compressed, comp_len, dp, HQLC_PCM16, dec_scratch);
  }

  free(enc);
  free(dec);
  free(enc_scratch);
  free(dec_scratch);
  return pcm_dec;
}

void test_roundtrip_mono(void) {
  const int n_frames = 6;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;

  int16_t *pcm_orig = (int16_t *)calloc(n_samples, sizeof(int16_t));
  gen_sine_pcm16(pcm_orig, n_samples, 1, 1000.0f, 0.5f);

  int16_t *pcm_dec = codec_roundtrip(pcm_orig, n_frames, 1, HQLC_MODE_FIXED, 2.0f, 0);

  // Compare settled frames (skip 2 at start, 2 at end, 1-frame latency)
  float snr = compute_snr(pcm_orig,
                          HQLC_FRAME_SAMPLES,
                          pcm_dec,
                          2 * HQLC_FRAME_SAMPLES,
                          2 * HQLC_FRAME_SAMPLES,
                          1);
  TEST_ASSERT_GREATER_THAN_FLOAT(15.0f, snr);

  free(pcm_orig);
  free(pcm_dec);
}

void test_roundtrip_stereo(void) {
  const int n_frames = 6;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;

  int16_t *pcm_orig = (int16_t *)calloc(n_samples * 2, sizeof(int16_t));
  for (int i = 0; i < n_samples; i++) {
    float t = (float)i / (float)HQLC_SAMPLE_RATE;
    pcm_orig[i * 2] = (int16_t)(0.4f * 32767.0f * sinf(2.0f * 3.14159265f * 440.0f * t));
    pcm_orig[i * 2 + 1] =
        (int16_t)(0.4f * 32767.0f * sinf(2.0f * 3.14159265f * 880.0f * t));
  }

  int16_t *pcm_dec = codec_roundtrip(pcm_orig, n_frames, 2, HQLC_MODE_FIXED, 1.5f, 0);

  float snr = compute_snr(pcm_orig,
                          HQLC_FRAME_SAMPLES,
                          pcm_dec,
                          2 * HQLC_FRAME_SAMPLES,
                          2 * HQLC_FRAME_SAMPLES,
                          2);
  TEST_ASSERT_GREATER_THAN_FLOAT(15.0f, snr);

  free(pcm_orig);
  free(pcm_dec);
}

void test_roundtrip_silence(void) {
  const int n_frames = 3;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;

  int16_t *pcm_orig = (int16_t *)calloc(n_samples, sizeof(int16_t));
  int16_t *pcm_dec = codec_roundtrip(pcm_orig, n_frames, 1, HQLC_MODE_FIXED, 1.0f, 0);

  for (int i = 0; i < n_samples; i++) {
    TEST_ASSERT_INT_WITHIN(1, 0, pcm_dec[i]);
  }

  free(pcm_orig);
  free(pcm_dec);
}

void test_roundtrip_onset(void) {
  // Quiet to loud transition: 4 frames at 0.01 amplitude, 8 frames at 0.5
  const int n_frames = 12;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;
  const int quiet_frames = 4;

  int16_t *pcm_orig = (int16_t *)calloc(n_samples, sizeof(int16_t));
  gen_sine_pcm16(pcm_orig, quiet_frames * HQLC_FRAME_SAMPLES, 1, 1000.0f, 0.01f);
  gen_sine_pcm16(pcm_orig + quiet_frames * HQLC_FRAME_SAMPLES,
                 (n_frames - quiet_frames) * HQLC_FRAME_SAMPLES,
                 1,
                 1000.0f,
                 0.5f);

  int16_t *pcm_dec = codec_roundtrip(pcm_orig, n_frames, 1, HQLC_MODE_FIXED, 2.0f, 0);

  // Check steady-state SNR well after the transition
  int steady_dec = (quiet_frames + 4) * HQLC_FRAME_SAMPLES;
  float snr = compute_snr(pcm_orig,
                          steady_dec - HQLC_FRAME_SAMPLES,
                          pcm_dec,
                          steady_dec,
                          3 * HQLC_FRAME_SAMPLES,
                          1);
  TEST_ASSERT_GREATER_THAN_FLOAT(15.0f, snr);

  // Check transition clipping is bounded (< 10%)
  int trans_start = quiet_frames * HQLC_FRAME_SAMPLES;
  int trans_len = 2 * HQLC_FRAME_SAMPLES;
  int clipped = 0;
  for (int i = trans_start; i < trans_start + trans_len; i++) {
    if (pcm_dec[i] >= 32767 || pcm_dec[i] <= -32768) {
      clipped++;
    }
  }
  float clip_pct = 100.0f * (float)clipped / (float)trans_len;
  TEST_ASSERT_MESSAGE(clip_pct < 10.0f, "Excessive onset clipping (>10%)");

  free(pcm_orig);
  free(pcm_dec);
}

void test_roundtrip_rc_mode(void) {
  const int n_frames = 10;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;

  int16_t *pcm_orig = (int16_t *)calloc(n_samples, sizeof(int16_t));
  gen_sine_pcm16(pcm_orig, n_samples, 1, 1000.0f, 0.5f);

  int16_t *pcm_dec = codec_roundtrip(pcm_orig, n_frames, 1, HQLC_MODE_RC, 0, 128000);

  float snr = compute_snr(pcm_orig,
                          2 * HQLC_FRAME_SAMPLES,
                          pcm_dec,
                          3 * HQLC_FRAME_SAMPLES,
                          4 * HQLC_FRAME_SAMPLES,
                          1);
  TEST_ASSERT_GREATER_THAN_FLOAT(15.0f, snr);

  free(pcm_orig);
  free(pcm_dec);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_roundtrip_mono);
  RUN_TEST(test_roundtrip_stereo);
  RUN_TEST(test_roundtrip_silence);
  RUN_TEST(test_roundtrip_onset);
  RUN_TEST(test_roundtrip_rc_mode);
  return UNITY_END();
}
