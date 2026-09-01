#include "tns.h"

#include <string.h>

#include "fxp.h"
#include "pcm.h"
#include "psy.h"

// TNS parameters
#define TNS_START_BIN   20 // ~940 Hz
#define TNS_Q30_ONE     FXP_Q30(1.0)
#define TNS_MAX_K_Q30   FXP_Q30(0.92)
#define TNS_K_CLAMP_Q30 FXP_Q30(0.999)

// Transient detector tuning
#define TNS_DETECT_RATIO 8
#define TNS_DETECT_FLOOR (1u << 20)

// Normalised bins are scaled to roughly 2^14
#define TNS_NORM_Q 14

// First fine band index above the TNS_START_BIN
#define TNS_ENV_FIRST_FB 19

// Span of bands where the TNS works on
#define TNS_ENV_BANDS (PSY_N_ACTIVE_FINE - TNS_ENV_FIRST_FB)

// Gaussian lag window w[k] = exp(-0.5 * (2*pi*f0*k)^2), Q30
// Smooths the envelope shape a bit, fitting more attack shape
//
// Hangover frames use a softer window, so post-attack frames don't smear quantization
// noise all over the spectrum
static const int32_t tns_lag_win_q30[TNS_MAX_ORDER] = {
    1054834932,
    1000088432,
    915085284,
    808079344,
};

static const int32_t tns_lag_win_soft_q30[TNS_MAX_ORDER] = {
    1022040948,
    881401074,
    688677203,
    487522531,
};

// Dequant LUT, k = tanh(q * 0.25), Q30, index = q + 7
static const int32_t tns_k_dq_q30[15] = {
    -1010794288, // q=-7
    -971895537,  // q=-6
    -910837623,  // q=-5
    -817755498,  // q=-4
    -681985995,  // q=-3
    -496194519,  // q=-2
    -262979411,  // q=-1
    0,           // q= 0
    262979411,   // q=+1
    496194519,   // q=+2
    681985995,   // q=+3
    817755498,   // q=+4
    910837623,   // q=+5
    971895537,   // q=+6
    1010794288,  // q=+7
};

// Quantization boundaries: if |k| >= boundary[i], q >= i+1
// boundary[i] = tanh((i + 0.5) * 0.25) in Q30
static const int32_t tns_quant_boundary_q30[7] = {
    133523019,
    384783327,
    595496917,
    755812887,
    868980407,
    944706725,
    993582944,
};

static int tns_quant_k(int32_t k_q30) {
  int sign = 1;
  int32_t abs_k = k_q30;
  if (k_q30 < 0) {
    sign = -1;
    abs_k = -k_q30;
  }

  int q = 0;
  for (int i = 0; i < 7; i++) {
    if (abs_k >= tns_quant_boundary_q30[i]) {
      q = i + 1;
    } else {
      break;
    }
  }
  return sign * q;
}

bool tns_detect_transient(tns_detect_state *st,
                          const uint8_t *curr_pcm,
                          hqlc_pcm_format fmt,
                          int stride,
                          int ch) {
  const int sub = HQLC_FRAME_SAMPLES / TNS_DETECT_SUBBLOCKS;
  uint64_t e[TNS_DETECT_SUBBLOCKS];
  int32_t last = st->last_sample;

  // HP (first difference) energy per sub-block. 24-bit input is scaled to
  // 16-bit range so energies and TNS_DETECT_FLOOR are format-independent.
  if (fmt == HQLC_PCM16) {
    const int16_t *p = (const int16_t *)curr_pcm;
    for (int b = 0; b < TNS_DETECT_SUBBLOCKS; b++) {
      uint64_t acc = 0;
      for (int i = b * sub; i < (b + 1) * sub; i++) {
        int32_t s = p[i * stride + ch];
        int32_t d = s - last;
        last = s;
        acc += (uint64_t)((int64_t)d * d);
      }
      e[b] = acc;
    }
  } else {
    for (int b = 0; b < TNS_DETECT_SUBBLOCKS; b++) {
      uint64_t acc = 0;
      for (int i = b * sub; i < (b + 1) * sub; i++) {
        int32_t s = pcm_load_native(curr_pcm, fmt, i * stride + ch) >> 8;
        int32_t d = s - last;
        last = s;
        acc += (uint64_t)((int64_t)d * d);
      }
      e[b] = acc;
    }
  }
  st->last_sample = last;

  // Each current sub-block vs the mean of the preceding 8 (window slides
  // from last frame's blocks into this frame's).
  uint64_t win = 0;
  for (int b = 0; b < TNS_DETECT_SUBBLOCKS; b++) {
    win += st->sub_energy[b];
  }

  bool fire = false;
  for (int b = 0; b < TNS_DETECT_SUBBLOCKS; b++) {
    if (e[b] > TNS_DETECT_FLOOR &&
        e[b] * TNS_DETECT_SUBBLOCKS > (uint64_t)TNS_DETECT_RATIO * win) {
      fire = true;
    }
    // Slide: drop the oldest (last frame's block b), add current block b
    win += e[b] - st->sub_energy[b];
  }

  memcpy(st->sub_energy, e, sizeof(e));
  return fire;
}

// Fast square root helper
static uint32_t tns_isqrt(uint64_t v) {
  if (v == 0) {
    return 0;
  }
  // Power-of-two seed just above sqrt(v), newton then needs at most 5 passes
  uint64_t x = 1ull << ((64 - __builtin_clzll(v) + 1) >> 1);
  uint64_t y = (x + v / x) >> 1;
  while (y < x) {
    x = y;
    y = (x + v / x) >> 1;
  }
  return (uint32_t)x;
}

/**
 * @brief Autocorrelation of the spectrum flattened by its fine-band envelope.
 *
 * Only the coefficient derivation works on this flattened spectrum
 */
static void tns_autocorrelation(const int32_t *spec_q31, int64_t *r) {
  for (int k = 0; k <= TNS_MAX_ORDER; k++) {
    r[k] = 0;
  }

  uint64_t psd_q46[PSY_N_ACTIVE_FINE];
  psy_fine_band_psd(spec_q31, psd_q46);
  psy_fine_band_psd_smooth(psd_q46);

  // Per-band envelope amplitude (Q23, the mantissa >>8 domain)
  // Only calculated for 28 values, hence it should be cheap enough
  int32_t rms_q23[TNS_ENV_BANDS];

  // band center for interpolation
  int16_t band_center[TNS_ENV_BANDS];
  for (int b = 0; b < TNS_ENV_BANDS; b++) {
    rms_q23[b] = (int32_t)tns_isqrt(psd_q46[TNS_ENV_FIRST_FB + b]);
    band_center[b] = (int16_t)((psy_fine_band_edges[TNS_ENV_FIRST_FB + b] +
                                psy_fine_band_edges[TNS_ENV_FIRST_FB + b + 1]) /
                               2);
  }

  // Normalised history, newest last
  int32_t hist[TNS_MAX_ORDER + 1] = {0};
  int b = 0;

  for (int i = TNS_START_BIN; i < PSY_ACTIVE_BINS; i++) {
    while (b + 1 < TNS_ENV_BANDS && i >= band_center[b + 1]) {
      b++;
    }
    // Linear interpolation between band centers, flat outside the outer ones
    int64_t env_q23 = rms_q23[b];
    if (b + 1 < TNS_ENV_BANDS && i > band_center[b]) {
      env_q23 += ((int64_t)(rms_q23[b + 1] - rms_q23[b]) * (i - band_center[b])) /
                 (band_center[b + 1] - band_center[b]);
    }

    int32_t norm_q14 = 0;
    if (env_q23 > 0) {
      // We shift the q31 into q23, so the env_q23 cancels it out, leaving the
      // normalization ratio in q14
      norm_q14 = (int32_t)((((int64_t)(spec_q31[i] >> 8)) << TNS_NORM_Q) / env_q23);
    }

    for (int k = TNS_MAX_ORDER; k > 0; k--) {
      hist[k] = hist[k - 1];
    }
    hist[0] = norm_q14;

    int lags = i - TNS_START_BIN < TNS_MAX_ORDER ? i - TNS_START_BIN : TNS_MAX_ORDER;
    for (int k = 0; k <= lags; k++) {
      r[k] += (int64_t)norm_q14 * hist[k];
    }
  }
}

/**
 * @brief Solve for reflection coefficients via Levinson-Durbin.
 *
 * @param r_raw     Autocorrelation values (int64, max_order+1 elements)
 * @param max_order Maximum filter order to solve for
 * @param hangover  Frame is eligible only through the hangover, not its own attack
 * @param k_out     Output reflection coefficients in Q30
 * @return Actual filter order (0 if prediction gain is insufficient)
 */
static int
tns_levinson_durbin(const int64_t *r_raw, int max_order, bool hangover, int32_t *k_out) {
  if (r_raw[0] <= 0) {
    return 0;
  }

  // Normalize r so r[0] fits in ~30 bits.
  int bits_used = 63 - __builtin_clzll((uint64_t)r_raw[0]);
  int shift = (bits_used > 30) ? (bits_used - 30) : 0;

  int32_t r[TNS_MAX_ORDER + 1];
  for (int k = 0; k <= max_order; k++) {
    r[k] = (int32_t)(r_raw[k] >> shift);
  }

  if (r[0] <= 0) {
    return 0;
  }

  // Softer window on hangover frames
  const int32_t *lag_win = hangover ? tns_lag_win_soft_q30 : tns_lag_win_q30;
  for (int k = 1; k <= max_order; k++) {
    // Apply lag window to the autocorrelation values
    r[k] = fxp_scale_q30(r[k], lag_win[k - 1]);
  }

  int32_t error = r[0];
  int32_t a[TNS_MAX_ORDER];
  memset(a, 0, sizeof(a));
  int order = 0;

  for (int i = 0; i < max_order; i++) {
    // acc = r[i+1] + sum(a[j] * r[i-j])
    int64_t acc = (int64_t)r[i + 1];
    for (int j = 0; j < i; j++) {
      acc += fxp_mul_rshift_i64(a[j], r[i - j], 30);
    }

    // ki = -acc / error, in Q30
    int32_t ki = (int32_t)(-((int64_t)((uint64_t)acc << 30) / error));

    ki = fxp_clamp_i32(ki, -TNS_K_CLAMP_Q30, TNS_K_CLAMP_Q30);

    // Update error: error *= (1 - ki^2)
    int32_t ki_sq = fxp_mul_q30(ki, ki);
    error = fxp_scale_q30(error, TNS_Q30_ONE - ki_sq);
    if (error <= 0) {
      break;
    }

    // Update prediction coefficients. On strong transients the update can
    // exceed Q30 headroom. The filter is blowing up, so keep this reflection
    // coefficient and stop extending the order.
    int32_t a_new[TNS_MAX_ORDER];
    bool overflowed = false;
    for (int j = 0; j < i; j++) {
      int64_t sum = (int64_t)a[j] + fxp_mul_q30(ki, a[i - 1 - j]);
      if (sum > INT32_MAX || sum < INT32_MIN) {
        overflowed = true;
        break;
      }
      a_new[j] = (int32_t)sum;
    }

    k_out[order] = ki;
    order++;
    if (overflowed) {
      break;
    }

    a_new[i] = ki;
    memcpy(a, a_new, (size_t)(i + 1) * sizeof(int32_t));
  }

  // Prediction gain gate, fire at >= 1.2
  if (order == 0 || 5 * (int64_t)r[0] < 6 * (int64_t)error) {
    return 0;
  }

  return order;
}

int32_t tns_dequant_k(int q) {
  return tns_k_dq_q30[q + TNS_LAR_HALF];
}

void tns_analyze(const int32_t *spec_q31, bool hangover, tns_info *out) {
  int64_t r[TNS_MAX_ORDER + 1];
  tns_autocorrelation(spec_q31, r);

  int32_t k_raw[TNS_MAX_ORDER];
  int order = tns_levinson_durbin(r, TNS_MAX_ORDER, hangover, k_raw);
  if (order == 0) {
    return;
  }

  int8_t q_lar[TNS_MAX_ORDER];
  int32_t k_dq[TNS_MAX_ORDER];
  for (int i = 0; i < order; i++) {
    int32_t k = k_raw[i];
    k = fxp_clamp_i32(k, -TNS_MAX_K_Q30, TNS_MAX_K_Q30);
    q_lar[i] = (int8_t)tns_quant_k(k);
    k_dq[i] = tns_dequant_k(q_lar[i]);
  }

  while (order > 0 && q_lar[order - 1] == 0) {
    order--;
  }
  if (order == 0) {
    return;
  }

  out->order = order;
  for (int i = 0; i < order; i++) {
    out->q_lar[i] = q_lar[i];
    out->k_q30[i] = k_dq[i];
  }
}

static void tns_lattice_fir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr) {
  if (order <= 0) {
    return;
  }

  int32_t b_state[TNS_MAX_ORDER];
  memset(b_state, 0, sizeof(b_state));
  uint32_t or_acc = 0;

  // Process only coded HF bins (TNS_START_BIN..PSY_ACTIVE_BINS), leave LF untouched
  for (int n = TNS_START_BIN; n < PSY_ACTIVE_BINS; n++) {
    int32_t f = spec_q31[n] >> input_rshift;
    int32_t b_prev = f;

    for (int i = 0; i < order; i++) {
      int32_t b_old = b_state[i];
      int32_t f_next = (int32_t)((int64_t)f + fxp_scale_q30(b_old, k_q30[i]));
      b_state[i] = b_prev;
      b_prev = (int32_t)((int64_t)fxp_scale_q30(f, k_q30[i]) + b_old);
      f = f_next;
    }

    spec_q31[n] = f;
    or_acc |= (uint32_t)(f ^ (f >> 31));
  }

  // Include LF bins in headroom calculation
  for (int n = 0; n < TNS_START_BIN; n++) {
    int32_t v = spec_q31[n] >> input_rshift;
    spec_q31[n] = v;
    or_acc |= (uint32_t)(v ^ (v >> 31));
  }

  if (out_hr) {
    *out_hr = fxp_signed_headroom_u32(or_acc);
  }
}

static void tns_lattice_iir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr) {
  if (order <= 0) {
    return;
  }

  int32_t b_state[TNS_MAX_ORDER];
  memset(b_state, 0, sizeof(b_state));
  uint32_t or_acc = 0;

  // Process only coded HF bins (TNS_START_BIN..PSY_ACTIVE_BINS), leave LF untouched
  for (int n = TNS_START_BIN; n < PSY_ACTIVE_BINS; n++) {
    int32_t f = spec_q31[n] >> input_rshift;

    f = (int32_t)((int64_t)f - fxp_scale_q30(b_state[order - 1], k_q30[order - 1]));

    for (int i = order - 2; i >= 0; i--) {
      int32_t b_old = b_state[i];
      f = (int32_t)((int64_t)f - fxp_scale_q30(b_old, k_q30[i]));
      b_state[i + 1] = (int32_t)((int64_t)fxp_scale_q30(f, k_q30[i]) + b_old);
    }

    spec_q31[n] = f;
    b_state[0] = f;
    or_acc |= (uint32_t)(f ^ (f >> 31));
  }

  // Include LF bins in headroom calculation
  for (int n = 0; n < TNS_START_BIN; n++) {
    int32_t v = spec_q31[n] >> input_rshift;
    spec_q31[n] = v;
    or_acc |= (uint32_t)(v ^ (v >> 31));
  }

  if (out_hr) {
    *out_hr = fxp_signed_headroom_u32(or_acc);
  }
}

/**
 * @brief Estimate pre-shift needed to prevent lattice overflow.
 */
static int tns_required_headroom(const int32_t *k_q30, int order, bool iir) {
  int64_t gain_q30 = TNS_Q30_ONE;
  int bits = 0;

  for (int i = 0; i < order; i++) {
    int32_t ak = k_q30[i] < 0 ? -k_q30[i] : k_q30[i];
    if (iir) {
      int32_t denom = TNS_Q30_ONE - ak;
      if (denom <= 0) {
        return 15; // near-unit pole
      }
      gain_q30 = (gain_q30 << 30) / denom;
    } else {
      gain_q30 = (gain_q30 * ((int64_t)TNS_Q30_ONE + ak)) >> 30;
    }
    // Renormalize to [2^30, 2^31) so the Q30 math never overflows
    while (gain_q30 >= 2 * (int64_t)TNS_Q30_ONE) {
      gain_q30 >>= 1;
      bits++;
    }
    if (bits >= 15) {
      return 15;
    }
  }

  return bits + (gain_q30 > TNS_Q30_ONE ? 1 : 0) + 1;
}

void tns_apply_analysis_filter(bfp_i32 *spectrum, const int32_t *k_q30, int order) {
  if (order <= 0) {
    return;
  }

  int required_headroom = tns_required_headroom(k_q30, order, false);
  bfp_i32_ensure_headroom(spectrum, required_headroom);
  int output_headroom;
  tns_lattice_fir(spectrum->data, k_q30, order, 0, &output_headroom);
  bfp_i32_renormalize(spectrum, (uint8_t)output_headroom);
}

void tns_apply_synthesis_filter(bfp_i32 *spectrum, const int32_t *k_q30, int order) {
  if (order <= 0) {
    return;
  }

  int required_headroom = tns_required_headroom(k_q30, order, true);
  bfp_i32_ensure_headroom(spectrum, required_headroom);
  int output_headroom;
  tns_lattice_iir(spectrum->data, k_q30, order, 0, &output_headroom);
  bfp_i32_renormalize(spectrum, (uint8_t)output_headroom);
}
