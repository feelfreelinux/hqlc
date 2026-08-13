#include "tns.h"

#include <string.h>

#include "fxp.h"
#include "pcm.h"

// TNS parameters
#define TNS_MAX_K_Q30   FXP_Q30(0.92)
#define TNS_K_CLAMP_Q30 FXP_Q30(0.999)

// Gaussian lag window w[k] = exp(-0.5 * (2*pi*f0*k)^2), Q30
// Smooths the envelope shape a bit, fitting more attack shape
//
// Hangover frames use a softer window, so post-attack frames dont smear quantization noise
// all over the spectrum. this can be seen as overshoot in HF energy on some frames
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
const int32_t tns_k_dq_q30[15] = {
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

int tns_quant_k(int32_t k_q30) {
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

// Autocorrelation of MDCT spectrum, pre-shifted to prevent overflow
static void tns_autocorrelation(const int32_t *spec, int n, int64_t *r) {
  for (int k = 0; k <= TNS_MAX_ORDER; k++) {
    r[k] = 0;
  }

  int main_end = n - TNS_MAX_ORDER;
  for (int i = 0; i < main_end; i++) {
    int32_t si = spec[i] >> 9;
    for (int k = 0; k <= TNS_MAX_ORDER; k++) {
      r[k] += (int64_t)si * spec[i + k];
    }
  }
  for (int i = main_end; i < n; i++) {
    int32_t si = spec[i] >> 9;
    for (int k = 0; k <= TNS_MAX_ORDER && i + k < n; k++) {
      r[k] += (int64_t)si * spec[i + k];
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
static int tns_levinson_durbin(const int64_t *r_raw,
                               int max_order,
                               bool hangover,
                               int32_t *k_out) {
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
    r[k] = (int32_t)(((int64_t)r[k] * lag_win[k - 1]) >> 30);
  }

  int32_t error = r[0];
  int32_t a[TNS_MAX_ORDER];
  memset(a, 0, sizeof(a));
  int order = 0;

  for (int i = 0; i < max_order; i++) {
    // acc = r[i+1] + sum(a[j] * r[i-j])
    int64_t acc = (int64_t)r[i + 1];
    for (int j = 0; j < i; j++) {
      acc += ((int64_t)a[j] * r[i - j]) >> 30;
    }

    // ki = -acc / error, in Q30
    int32_t ki = (int32_t)(-((int64_t)((uint64_t)acc << 30) / error));

    ki = fxp_clamp_i32(ki, -TNS_K_CLAMP_Q30, TNS_K_CLAMP_Q30);

    // Update error: error *= (1 - ki^2)
    int32_t ki_sq = (int32_t)(((int64_t)ki * ki) >> 30);
    error = (int32_t)(((int64_t)error * ((1 << 30) - ki_sq)) >> 30);
    if (error <= 0) {
      break;
    }

    // Update prediction coefficients. On strong transients the update can
    // exceed Q30 headroom. The filter is blowing up, so keep this reflection
    // coefficient and stop extending the order.
    int32_t a_new[TNS_MAX_ORDER];
    bool overflowed = false;
    for (int j = 0; j < i; j++) {
      int64_t sum = (int64_t)a[j] + (((int64_t)ki * a[i - 1 - j]) >> 30);
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

  // Matches >= 1.5 on hangover, 1.2 else
  int64_t gate_num = hangover ? 2 : 5;
  int64_t gate_den = hangover ? 3 : 6;
  if (order == 0 || gate_num * (int64_t)r[0] < gate_den * (int64_t)error) {
    return 0;
  }

  return order;
}

void tns_analyze(const int32_t *spec_q31, bool hangover, tns_info *out) {
  int64_t r[TNS_MAX_ORDER + 1];
  tns_autocorrelation(spec_q31 + TNS_START_BIN, HQLC_FRAME_SAMPLES - TNS_START_BIN, r);

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

void tns_lattice_fir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr) {
  if (order <= 0) {
    return;
  }

  int32_t b_state[TNS_MAX_ORDER];
  memset(b_state, 0, sizeof(b_state));
  uint32_t or_acc = 0;

  // Process only HF bins (TNS_START_BIN onward), leave LF untouched
  for (int n = TNS_START_BIN; n < HQLC_FRAME_SAMPLES; n++) {
    int32_t f = spec_q31[n] >> input_rshift;
    int32_t b_prev = f;

    for (int i = 0; i < order; i++) {
      int32_t b_old = b_state[i];
      int32_t f_next = (int32_t)((int64_t)f + ((int64_t)k_q30[i] * b_old >> 30));
      b_state[i] = b_prev;
      b_prev = (int32_t)(((int64_t)k_q30[i] * f >> 30) + b_old);
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
    *out_hr = fxp_headroom_u32(or_acc);
  }
}

void tns_lattice_iir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr) {
  if (order <= 0) {
    return;
  }

  int32_t b_state[TNS_MAX_ORDER];
  memset(b_state, 0, sizeof(b_state));
  uint32_t or_acc = 0;

  // Process only HF bins (TNS_START_BIN onward), leave LF untouched
  for (int n = TNS_START_BIN; n < HQLC_FRAME_SAMPLES; n++) {
    int32_t f = spec_q31[n] >> input_rshift;

    f = (int32_t)((int64_t)f - ((int64_t)k_q30[order - 1] * b_state[order - 1] >> 30));

    for (int i = order - 2; i >= 0; i--) {
      int32_t b_old = b_state[i];
      f = (int32_t)((int64_t)f - ((int64_t)k_q30[i] * b_old >> 30));
      b_state[i + 1] = (int32_t)(((int64_t)k_q30[i] * f >> 30) + b_old);
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
    *out_hr = fxp_headroom_u32(or_acc);
  }
}

/**
 * @brief Estimate pre-shift needed to prevent lattice overflow.
 */
static int tns_preshift(const int32_t *k_q30, int order, bool iir) {
  int64_t gain_q30 = 1 << 30;
  int bits = 0;

  for (int i = 0; i < order; i++) {
    int32_t ak = k_q30[i] < 0 ? -k_q30[i] : k_q30[i];
    if (iir) {
      int32_t denom = (1 << 30) - ak;
      if (denom <= 0) {
        return 15; // near-unit pole
      }
      gain_q30 = (gain_q30 << 30) / denom;
    } else {
      gain_q30 = (gain_q30 * (((int64_t)1 << 30) + ak)) >> 30;
    }
    // Renormalize to [2^30, 2^31) so the Q30 math never overflows
    while (gain_q30 >= ((int64_t)1 << 31)) {
      gain_q30 >>= 1;
      bits++;
    }
    if (bits >= 15) {
      return 15;
    }
  }

  return bits + (gain_q30 > (1 << 30) ? 1 : 0) + 1;
}

// Renormalize spectrum after filtering, reclaiming unused headroom
static void tns_renormalize(int32_t *spec_q31, int hr) {
  if (hr > 0) {
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = (int32_t)((uint32_t)spec_q31[i] << hr);
    }
  }
}

int tns_fir_safe(int32_t *spec_q31, const int32_t *k_q30, int order) {
  if (order <= 0) {
    return 0;
  }

  int preshift = tns_preshift(k_q30, order, false);
  int hr;
  tns_lattice_fir(spec_q31, k_q30, order, preshift, &hr);
  tns_renormalize(spec_q31, hr);
  return preshift - hr;
}

int tns_iir_safe(int32_t *spec_q31, const int32_t *k_q30, int order) {
  if (order <= 0) {
    return 0;
  }

  int preshift = tns_preshift(k_q30, order, true);
  int hr;
  tns_lattice_iir(spec_q31, k_q30, order, preshift, &hr);
  tns_renormalize(spec_q31, hr);
  return preshift - hr;
}
