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

void test_decode_corrupt_frames(void) {
  // Corrupted / truncated frames must decode without memory errors: either
  // HQLC_OK (garbage audio is fine) or HQLC_ERR_BITSTREAM_CORRUPT.
  const int n_frames = 8;
  const int n_samples = n_frames * HQLC_FRAME_SAMPLES;

  int16_t *pcm_orig = (int16_t *)calloc((size_t)n_samples * 2, sizeof(int16_t));
  gen_sine_pcm16(pcm_orig, n_samples, 2, 700.0f, 0.4f);
  // Attack in frame 4 so some frames carry TNS side info
  for (int i = 4 * HQLC_FRAME_SAMPLES; i < 4 * HQLC_FRAME_SAMPLES + 64; i++) {
    pcm_orig[i * 2] = 30000;
    pcm_orig[i * 2 + 1] = -30000;
  }

  hqlc_encoder *enc = (hqlc_encoder *)calloc(1, hqlc_encoder_size());
  hqlc_decoder *dec = (hqlc_decoder *)calloc(1, hqlc_decoder_size());
  hqlc_encoder_config cfg = {
      .channels = 2, .sample_rate = HQLC_SAMPLE_RATE, .mode = HQLC_MODE_RC};
  cfg.bitrate = 96000;
  hqlc_encoder_init(enc, &cfg);
  hqlc_decoder_init(dec, 2, HQLC_SAMPLE_RATE);
  void *enc_scratch = calloc(1, hqlc_encoder_scratch_size());
  void *dec_scratch = calloc(1, hqlc_decoder_scratch_size());

  uint8_t compressed[HQLC_MAX_FRAME_BYTES];
  uint8_t mangled[HQLC_MAX_FRAME_BYTES];
  int16_t pcm_dec[HQLC_FRAME_SAMPLES * 2];
  uint32_t rng = 12345;

  for (int f = 0; f < n_frames; f++) {
    const uint8_t *fp = (const uint8_t *)&pcm_orig[f * HQLC_FRAME_SAMPLES * 2];
    size_t comp_len = 0;
    TEST_ASSERT_EQUAL(HQLC_OK,
                      hqlc_encode_frame(enc,
                                        fp,
                                        HQLC_PCM16,
                                        compressed,
                                        HQLC_MAX_FRAME_BYTES,
                                        &comp_len,
                                        enc_scratch));

    for (int variant = 0; variant < 4; variant++) {
      memcpy(mangled, compressed, comp_len);
      for (int k = 0; k < variant * 8 && comp_len > 0; k++) {
        rng = rng * 1103515245u + 12345u;
        mangled[(rng >> 8) % comp_len] ^= (uint8_t)(rng >> 16);
      }
      size_t use_len = (variant == 3) ? comp_len / 3 : comp_len;
      hqlc_error err = hqlc_decode_frame(
          dec, mangled, use_len, (uint8_t *)pcm_dec, HQLC_PCM16, dec_scratch);
      TEST_ASSERT_TRUE(err == HQLC_OK || err == HQLC_ERR_BITSTREAM_CORRUPT);
    }
  }

  free(pcm_orig);
  free(enc);
  free(dec);
  free(enc_scratch);
  free(dec_scratch);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_roundtrip_mono);
  RUN_TEST(test_roundtrip_stereo);
  RUN_TEST(test_roundtrip_silence);
  RUN_TEST(test_roundtrip_rc_mode);
  RUN_TEST(test_decode_corrupt_frames);
  return UNITY_END();
}
