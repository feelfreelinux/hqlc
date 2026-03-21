#include "unity.h"

#include <math.h>
#include <string.h>

#include "fxp.h"
#include "hqlc.h"
#include "mdct.h"

static uint8_t scratch[MDCT_SCRATCH_BYTES] __attribute__((aligned(8)));

// OLA helper
static hqlc_error mdct_ola_process(mdct_ola_state *state,
                                   const uint8_t *pcm_data,
                                   size_t pcm_data_len,
                                   hqlc_pcm_format fmt,
                                   int stride,
                                   int channel_idx,
                                   int32_t *out_pcm_q31,
                                   size_t out_pcm_q31_len,
                                   void *work,
                                   size_t work_len) {
  const int N = MDCT_N;
  int32_t spectral[MDCT_N];
  int32_t windowed[MDCT_BLOCK_LEN];

  size_t bps = (fmt == HQLC_PCM16) ? 2 : 3;
  size_t half_len = (size_t)HQLC_FRAME_SAMPLES * stride * bps;
  if (pcm_data_len < 2 * half_len) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  int loss_spec = 0;
  hqlc_error err = mdct_forward(pcm_data,
                                pcm_data + half_len,
                                half_len,
                                fmt,
                                stride,
                                channel_idx,
                                spectral,
                                MDCT_N,
                                work,
                                work_len,
                                &loss_spec);
  if (err != HQLC_OK) {
    return err;
  }

  int loss_time = 0;
  err = mdct_inverse(
      spectral, MDCT_N, loss_spec, windowed, MDCT_BLOCK_LEN, work, work_len, &loss_time);
  if (err != HQLC_OK) {
    return err;
  }

  if (!state->has_overlap) {
    state->loss_td_bits = loss_time;
    state->has_overlap = true;
  }

  int ola_loss = state->loss_td_bits;
  int common_loss = (ola_loss > loss_time) ? ola_loss : loss_time;
  int ola_shift = (common_loss - ola_loss) + 1;
  int win_shift = (common_loss - loss_time) + 1;
  int adj_loss = common_loss + 1;

  for (int i = 0; i < N; i++) {
    int32_t o = (ola_shift < 31) ? fxp_shr_rnd_i32(state->overlap_q31[i], ola_shift) : 0;
    int32_t w = (win_shift < 31) ? fxp_shr_rnd_i32(windowed[i], win_shift) : 0;
    int32_t y = o + w;
    if (adj_loss > 0) {
      out_pcm_q31[i] = fxp_shl_sat_i32(y, adj_loss);
    } else if (adj_loss < 0) {
      out_pcm_q31[i] = fxp_shr_rnd_i32(y, -adj_loss);
    } else {
      out_pcm_q31[i] = y;
    }
  }

  int32_t *new_overlap = &windowed[N];
  uint32_t ola_or = 0;
  for (int i = 0; i < N; i++) {
    ola_or |= (uint32_t)fxp_abs_i32(new_overlap[i]);
  }
  int hr = (ola_or == 0) ? 31 : __builtin_clz(ola_or) - 1;
  for (int i = 0; i < N; i++) {
    state->overlap_q31[i] =
        (hr > 0) ? fxp_shl_sat_i32(new_overlap[i], hr) : new_overlap[i];
  }
  state->loss_td_bits = loss_time - hr;
  return HQLC_OK;
}

// Q31 tolerance: 2^14 ~ -97 dBFS.
#define TOLERANCE 32769 // ~-90 dBFS, accounts for >>32 twiddle approximation

void setUp(void) {}
void tearDown(void) {}

static void fill_sine_pcm16(int16_t *buf, int len, double freq_hz, double fs) {
  for (int i = 0; i < len; i++) {
    buf[i] = (int16_t)(16000.0 * sin(2.0 * M_PI * freq_hz * i / fs));
  }
}

// 3-frame OLA round-trip, returns reconstructed Q31 for middle frame.
static void run_ola_roundtrip(const int16_t *signal, int32_t *recon_q31) {
  mdct_ola_state ola;
  mdct_ola_init(&ola);
  int32_t out[MDCT_N];

  for (int frame = 0; frame < 3; frame++) {
    int offset = frame * MDCT_N;
    hqlc_error err = mdct_ola_process(&ola,
                                      (const uint8_t *)&signal[offset],
                                      (size_t)MDCT_BLOCK_LEN * 2,
                                      HQLC_PCM16,
                                      1,
                                      0,
                                      out,
                                      MDCT_N,
                                      scratch,
                                      sizeof(scratch));
    TEST_ASSERT_EQUAL_INT(HQLC_OK, err);

    if (frame == 1) {
      memcpy(recon_q31, out, (size_t)MDCT_N * sizeof(int32_t));
    }
  }
}

void test_mdct_roundtrip_dc(void) {
  int16_t signal[2048];
  for (int i = 0; i < 2048; i++) {
    signal[i] = 0x4000;
  }

  int32_t recon[MDCT_N];
  run_ola_roundtrip(signal, recon);

  int32_t expected = (int32_t)0x4000 << 16;
  int32_t max_err = 0;
  for (int i = 0; i < MDCT_N; i++) {
    int32_t diff = recon[i] - expected;
    if (diff < 0) {
      diff = -diff;
    }
    if (diff > max_err) {
      max_err = diff;
    }
  }
  TEST_ASSERT_LESS_THAN_INT32(TOLERANCE, max_err);
}

void test_mdct_roundtrip_sine(void) {
  int16_t signal[2048];
  fill_sine_pcm16(signal, 2048, 1000.0, 48000.0);

  int32_t recon[MDCT_N];
  run_ola_roundtrip(signal, recon);

  int32_t max_err = 0;
  for (int i = 0; i < MDCT_N; i++) {
    int32_t expected = (int32_t)signal[MDCT_N + i] << 16;
    int32_t diff = recon[i] - expected;
    if (diff < 0) {
      diff = -diff;
    }
    if (diff > max_err) {
      max_err = diff;
    }
  }
  TEST_ASSERT_LESS_THAN_INT32(TOLERANCE, max_err);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_mdct_roundtrip_dc);
  RUN_TEST(test_mdct_roundtrip_sine);
  return UNITY_END();
}
