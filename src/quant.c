#include "quant.h"

#include <math.h>

#include "fxp.h"
#include "hqlc.h"
#include "hqlc_bench.h"
#include "psy.h"

// Quantizer step: step = 2^(E/8) with E = 2*exp - gain_code - QUANT_EXP_OFFSET,
// decomposed as pow2_eighth[E % 8] * 2^(E / 8)

// 2^(f/8) for f=0..7, Q30
const int32_t quant_pow2_eighth_q30[8] = {
    1073741824,
    1170923762,
    1276901417,
    1392470869,
    1518500250,
    1655936265,
    1805811301,
    1969251188,
};

// Reciprocals of the band-center distances, for the per-bin interpolation
static const uint16_t seg_rcp_q15[PSY_N_BANDS - 1] = {
    8192, 6553, 5461, 5461, 4096, 4096, 3640, 3276, 2978, 2340,
    2048, 1724, 1489, 1213, 1057, 862,  744,  630,  528,
};

// Per-bin exponents: linear ramp between band centers (tonal frames)
static void bin_exp_interp(const int32_t *exp_indices, int32_t *bin_exp) {
  int prev = (psy_band_edges[0] + psy_band_edges[1] + 2) >> 1;

  // Flat region before first band center
  for (int i = 0; i < prev; i++) {
    bin_exp[i] = exp_indices[0];
  }

  // Linear ramp between adjacent band centers
  for (int b = 0; b < PSY_N_BANDS - 1; b++) {
    int next = (psy_band_edges[b + 1] + psy_band_edges[b + 2] + 2) >> 1;
    int32_t v0 = exp_indices[b];
    int32_t delta_q15 = (exp_indices[b + 1] - v0) * (int32_t)seg_rcp_q15[b];
    int32_t acc = (v0 << 15) + (delta_q15 >> 1);
    for (int i = prev; i < next; i++) {
      bin_exp[i] = (acc + (1 << 14)) >> 15;
      acc += delta_q15;
    }
    prev = next;
  }

  // Flat region after last band center
  for (int i = prev; i < PSY_ACTIVE_BINS; i++) {
    bin_exp[i] = exp_indices[PSY_N_BANDS - 1];
  }
}

// Per-bin exponents: held flat across each band (TNS frames)
static void bin_exp_flat(const int32_t *exp_indices, int32_t *bin_exp) {
  for (int b = 0; b < PSY_N_BANDS; b++) {
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      bin_exp[i] = exp_indices[b];
    }
  }
}

// NF estimate state, mirrors the decoder fill positions with a 2-bin lag
typedef struct {
  int32_t s1, s2; // scaled_q8 of the previous two bins
  int32_t z;      // consecutive sub-deadzone bins
  int32_t total, ns;
} nf_est;

// Branchless, below/take are all-ones condition masks
static inline void nf_update(nf_est *nf, int32_t scaled_q8) {
  int32_t below = (scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31;
  nf->z = (nf->z + 1) & below;
  int32_t take = (4 - nf->z) >> 31;
  nf->total += nf->s2 & take;
  nf->ns -= take;
  nf->s2 = nf->s1;
  nf->s1 = scaled_q8;
}

// Deadzone, round, restore sign, and feed the NF estimate
static inline void quant_bin(int16_t *out, nf_est *nf, int32_t scaled_q8, int32_t sign) {
  int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
  int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;
  *out = (int16_t)((q ^ sign) - sign);
  nf_update(nf, scaled_q8);
}

// Forward quantizer fused with the NF estimate. bin_exp comes in runs of
// equal exponent, so the step setup is hoisted per run and the bin loops
// stay branchless.
int quant_forward_nf(const int32_t *spec_q31,
                     int loss_bits,
                     const int32_t *exp_indices,
                     int gain_code,
                     bool interp,
                     int16_t *quant_out) {
  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    quant_out[i] = 0;
  }

  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_QLOOP);
  int32_t bin_exp[PSY_ACTIVE_BINS];
  if (interp) {
    bin_exp_interp(exp_indices, bin_exp);
  } else {
    bin_exp_flat(exp_indices, bin_exp);
  }

  nf_est nf = {0, 0, 0, 0, 0};
  int E_bias = gain_code + QUANT_EXP_OFFSET;

  int i = 0;
  while (i < PSY_ACTIVE_BINS) {
    int32_t be = bin_exp[i];
    int run_end = i + 1;
    while (run_end < PSY_ACTIVE_BINS && bin_exp[run_end] == be) {
      run_end++;
    }

    int neg_E = E_bias - 2 * (int)be;
    int32_t inv_step_m = quant_pow2_eighth_q30[neg_E & 7] >> 2;
    int total_shift = QUANT_TOTAL_Q - loss_bits - (neg_E >> 3);

    if (total_shift >= 64) {
      // Step too coarse: the whole run quantizes to zero
      for (; i < run_end; i++) {
        quant_bin(&quant_out[i], &nf, 0, 0);
      }
    } else if (total_shift >= 32) {
      // Common case: only the high word of the product is needed
      int small_shift = total_shift - 32;
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;
        int32_t hi = (int32_t)((int64_t)abs_spec * inv_step_m >> 32);
        quant_bin(&quant_out[i], &nf, hi >> small_shift, sign);
      }
    } else if (total_shift > 0) {
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;
        int32_t scaled_q8 = (int32_t)((int64_t)abs_spec * inv_step_m >> total_shift);
        quant_bin(&quant_out[i], &nf, scaled_q8, sign);
      }
    } else {
      // Saturation path, should be quite rare
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;
        int32_t scaled_q8 =
            fxp_sat_i64_to_i32((int64_t)abs_spec * inv_step_m << (-total_shift));
        quant_bin(&quant_out[i], &nf, scaled_q8, sign);
      }
    }
  }
  HQLC_BENCH_END(HQLC_BENCH_ENC_QLOOP);

  if (nf.ns == 0) {
    return 7;
  }
  int32_t avg_q8 = nf.total / nf.ns;
  int nf_code = (128 - avg_q8 + 8) >> 4;
  return fxp_clamp_i32(nf_code, 0, 7);
}

// Inverse quantizer: integer symbols to reconstructed BFP spectrum
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   bool interp,
                   int32_t *spec_q31,
                   int *loss_bits_out) {
  int max_exp = exp_indices[0];
  for (int b = 1; b < PSY_N_BANDS; b++) {
    if (exp_indices[b] > max_exp) {
      max_exp = exp_indices[b];
    }
  }
  int max_oct = (2 * max_exp - gain_code - QUANT_EXP_OFFSET) >> 3;

  // Interpolated exponents never exceed the band max, so max_oct holds
  int32_t bin_exp[PSY_ACTIVE_BINS];
  if (interp) {
    bin_exp_interp(exp_indices, bin_exp);
  } else {
    bin_exp_flat(exp_indices, bin_exp);
  }

  // Find the max dequantized magnitude (for BFP headroom)
  uint64_t max_val = 0;
  for (int i = 0; i < PSY_ACTIVE_BINS; i++) {
    if (quant_in[i] == 0) {
      continue;
    }
    int mag = (quant_in[i] > 0) ? quant_in[i] : -quant_in[i];
    int E = 2 * (int)bin_exp[i] - gain_code - QUANT_EXP_OFFSET;
    int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
    int oct_shift = max_oct - (E >> 3);
    int32_t dq_q8 = mag * 256 + Q8(QUANT_CENTROID);
    uint64_t val =
        (oct_shift < 63) ? (uint64_t)((int64_t)dq_q8 * step_m >> oct_shift) : 0;
    if (val > max_val) {
      max_val = val;
    }
  }

  if (max_val == 0) {
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = 0;
    }
    *loss_bits_out = 0;
    return;
  }

  int headroom = (int)__builtin_clzll(max_val) - 1;
  int norm_shift = 32 - headroom;

  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    spec_q31[i] = 0;
  }

  for (int i = 0; i < PSY_ACTIVE_BINS; i++) {
    if (quant_in[i] == 0) {
      spec_q31[i] = 0;
      continue;
    }
    int sign = (quant_in[i] > 0) ? 1 : -1;
    int mag = (quant_in[i] > 0) ? quant_in[i] : -quant_in[i];
    int E = 2 * (int)bin_exp[i] - gain_code - QUANT_EXP_OFFSET;
    int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
    int total_shift = (max_oct - (E >> 3)) + norm_shift;
    int32_t dq_q8 = mag * 256 + Q8(QUANT_CENTROID);
    spec_q31[i] =
        (total_shift < 63) ? sign * (int32_t)((int64_t)dq_q8 * step_m >> total_shift) : 0;
  }

  *loss_bits_out = max_oct - headroom + 27;
}

int quant_gain_encode(float gain) {
  float v = log2f(gain > 1e-12f ? gain : 1e-12f) * QUANT_GAIN_Q;
  int code = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f)) + QUANT_GAIN_BIAS;
  return fxp_clamp_i32(code, 0, QUANT_GAIN_MAX_CODE);
}

// Fill runs of >4 consecutive zeros with pseudorandom noise at (8-nf)/16 * step (nf =
// transmitted 3-bit factor, 0 strongest) Operates on the dequantized BFP spectrum,
// widening it first if the fill needs more headroom. Always flat per-band steps, did not
// get significant improvement with interp-envelope fill
void nf_run_length_fill(const int16_t *quant,
                        const int32_t *exp_indices,
                        int gain_code,
                        int nf,
                        uint32_t seed,
                        int32_t *spec_q31,
                        int *loss_bits_io) {
  int loss_bits = *loss_bits_io;
  int nf_scale = 8 - nf; // 1..8
  int E_bias = gain_code + QUANT_EXP_OFFSET;

  // Max octave from band exponents
  int max_fill_oct = -32767;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int oct = (2 * (int)exp_indices[b] - E_bias) >> 3;
    if (oct > max_fill_oct) {
      max_fill_oct = oct;
    }
  }

  // Check if any fills exist
  int z = 0, has_fill = 0;
  for (int i = 0; i < PSY_ACTIVE_BINS; i++) {
    z = (quant[i] == 0) ? z + 1 : 0;
    if (z > 4) {
      has_fill = 1;
      break;
    }
  }
  if (!has_fill) {
    return;
  }

  // Ensure BFP can hold the fill values
  int fill_loss = max_fill_oct - 3;
  if (fill_loss > loss_bits) {
    int delta = fill_loss - loss_bits;
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] >>= delta;
    }
    loss_bits = fill_loss;
  }

  // Fill runs band at a time
  z = 0;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int E = 2 * (int)exp_indices[b] - E_bias;
    int octave = E >> 3;
    int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
    int32_t fill_m = (int32_t)((uint32_t)nf_scale * (uint32_t)step_m >> 4);
    int shift = loss_bits - octave - 3;

    for (int i = s; i < e; i++) {
      z = (quant[i] == 0) ? z + 1 : 0;
      if (z > 4) {
        seed = (13849 + seed * 31821) & 0xFFFF;

        int32_t fill_q31;
        if (shift >= 32) {
          fill_q31 = 0;
        } else if (shift > 0) {
          fill_q31 = fill_m >> shift;
        } else {
          fill_q31 = fxp_shl_sat_i32(fill_m, -shift);
        }

        // Write lags 2 bins, so runs keep a 2-bin guard at each edge
        spec_q31[i - 2] = (seed & 0x8000) ? -fill_q31 : fill_q31;
      }
    }
  }

  *loss_bits_io = loss_bits;
}
