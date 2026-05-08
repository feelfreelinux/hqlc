#include "quant.h"

#include <math.h>

#include "fxp.h"
#include "hqlc.h"
#include "hqlc_bench.h"
#include "psy.h"

/*
 * Quantizer step decomposition.
 *
 * The per-bin quantizer step is:
 *   step = 2^((2*exp - gain_code - 59) / 8) = mantissa * 2^octave
 *
 * where exp is the interpolated exponent (integer), and
 * - mantissa = pow2_eighth[frac]  (Q30 LUT, 8 entries)
 * - octave = E / 8             (integer part)
 * - frac = E % 8             (fractional part, 0..7)
 * - E = 2*exp - gain_code - QUANT_EXP_OFFSET
 *
 * For the forward quantizer we need 1/step (inv_step), for the inverse
 * we need step directly. Both decompose the same way.
 */

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

/*
 * Per-bin exponent interpolation.
 *
 * Linearly interpolates the 20 coarse-band exponent indices to 427 per-bin
 * values using precomputed (lo, hi, t_q15) tables. Result is rounded to
 * integer — the sub-integer precision is not needed since the step LUT
 * has only 8 entries per octave (1.5 dB granularity).
 */
static const uint16_t seg_rcp_q15[PSY_N_BANDS - 1] = {
    8192, 6553, 5461, 5461, 4096, 4096, 3640, 3276, 2978, 2340,
    2048, 1724, 1489, 1213, 1057, 862,  744,  630,  528,
};

void quant_interp_bin_exp(const int32_t *exp_indices, int32_t *bin_exp) {
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

/*
 * Compute |X| / step in Q8, for deadzone comparison and quantization.
 *
 * inv_step mantissa is pow2_eighth[frac] in Q28 (Q30 >> 2).
 * total_shift aligns the Q31 spectrum × Q28 mantissa to Q8 output.
 */
static inline int32_t scale_to_q8(int32_t abs_spec, int32_t inv_step_m, int total_shift) {
  if (total_shift >= 64) {
    return 0;
  }
  if (total_shift >= 32) {
    int32_t hi = (int32_t)((int64_t)abs_spec * inv_step_m >> 32);
    return hi >> (total_shift - 32);
  }
  if (total_shift > 0) {
    return (int32_t)((int64_t)abs_spec * inv_step_m >> total_shift);
  }
  return fxp_sat_i64_to_i32((int64_t)abs_spec * inv_step_m << (-total_shift));
}

// Forward quantizer
void quant_forward(const int32_t *spec_q31,
                   int loss_bits,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out) {
  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    quant_out[i] = 0;
  }

  int32_t bin_exp[PSY_ACTIVE_BINS];
  quant_interp_bin_exp(exp_indices, bin_exp);

  for (int i = 0; i < PSY_ACTIVE_BINS; i++) {
    // Decompose -E for inv_step: -E = 8*int_part + frac, frac in 0..7
    int E = 2 * bin_exp[i] - gain_code - QUANT_EXP_OFFSET;
    int neg_E = -E;
    int int_part = neg_E >> 3;
    int frac = neg_E & 7;

    int32_t inv_step_m = quant_pow2_eighth_q30[frac] >> 2; // Q28
    int total_shift = QUANT_TOTAL_Q - loss_bits - int_part;

    int32_t x = spec_q31[i];
    int32_t sign = x >> 31;
    int32_t abs_spec = (x ^ sign) - sign;

    int32_t scaled_q8 = scale_to_q8(abs_spec, inv_step_m, total_shift);

    // Deadzone, zero if |scaled| < 0.65, else q = floor(|scaled| - 0.65 + 1)
    int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
    int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;
    quant_out[i] = (int16_t)((q ^ sign) - sign);
  }
}

/* ── Forward quantizer + NF estimation (flat per-band exponents) ── */
int quant_forward_nf(const int32_t *spec_q31,
                     int loss_bits,
                     const int32_t *exp_indices,
                     int gain_code,
                     int16_t *quant_out) {
  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    quant_out[i] = 0;
  }

  HQLC_BENCH_BEGIN();
  int z = 0;
  int32_t nf_total = 0, nf_ns = 0;
  int32_t recent[8] = {0};
  int E_bias = gain_code + QUANT_EXP_OFFSET;

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int neg_E = E_bias - 2 * (int)exp_indices[b];
    int32_t inv_step_m = quant_pow2_eighth_q30[neg_E & 7] >> 2;
    int total_shift = QUANT_TOTAL_Q - loss_bits - (neg_E >> 3);

    if (total_shift >= 64) {
      for (int i = s; i < e; i++) {
        quant_out[i] = 0;
        recent[i & 7] = 0;
        if (i >= 6) {
          z++;
          if (z > 4) {
            nf_total += recent[(i - 2) & 7];
            nf_ns++;
          }
        }
      }
    } else if (total_shift >= 32) {
      int small_shift = total_shift - 32;
      for (int i = s; i < e; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;
        int32_t hi = (int32_t)((int64_t)abs_spec * inv_step_m >> 32);
        int32_t scaled_q8 = hi >> small_shift;
        int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
        int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;
        quant_out[i] = (int16_t)((q ^ sign) - sign);

        recent[i & 7] = scaled_q8;
        if (i >= 6) {
          z = (scaled_q8 < QUANT_DZ_THRESH_Q8) ? z + 1 : 0;
          if (z > 4) {
            nf_total += recent[(i - 2) & 7];
            nf_ns++;
          }
        }
      }
    } else {
      for (int i = s; i < e; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;
        int32_t scaled_q8 = scale_to_q8(abs_spec, inv_step_m, total_shift);
        int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
        int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;
        quant_out[i] = (int16_t)((q ^ sign) - sign);

        recent[i & 7] = scaled_q8;
        if (i >= 6) {
          z = (scaled_q8 < QUANT_DZ_THRESH_Q8) ? z + 1 : 0;
          if (z > 4) {
            nf_total += recent[(i - 2) & 7];
            nf_ns++;
          }
        }
      }
    }
  }

  HQLC_BENCH_END(HQLC_BENCH_ENC_QLOOP);

  if (nf_ns == 0) {
    return 7;
  }
  int32_t avg_q8 = nf_total / nf_ns;
  int nf = (128 - avg_q8 + 8) >> 4;
  return fxp_clamp_i32(nf, 0, 7);
}

// Perform inverse quantize
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   int32_t *spec_q31,
                   int *loss_bits_out) {

  // Find max octave and max |quant| per band for analytical headroom
  int max_exp = exp_indices[0];
  for (int b = 1; b < PSY_N_BANDS; b++) {
    if (exp_indices[b] > max_exp) {
      max_exp = exp_indices[b];
    }
  }
  int max_oct = (2 * max_exp - gain_code - QUANT_EXP_OFFSET) >> 3;

  // Find the max dequantized magnitude across all bands (for BFP headroom)
  uint64_t max_val = 0;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int max_mag = 0;
    for (int i = s; i < e; i++) {
      int m = (quant_in[i] > 0) ? quant_in[i] : -quant_in[i];
      if (m > max_mag) {
        max_mag = m;
      }
    }
    if (max_mag == 0) {
      continue;
    }
    int E = 2 * (int)exp_indices[b] - gain_code - QUANT_EXP_OFFSET;
    int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
    int oct_shift = max_oct - (E >> 3);
    int32_t dq_q8 = max_mag * 256 + Q8(QUANT_CENTROID);
    uint64_t val = (uint64_t)((int64_t)dq_q8 * step_m >> oct_shift);
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

  // Single pass, dequantize directly to int32
  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    spec_q31[i] = 0;
  }

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int E = 2 * (int)exp_indices[b] - gain_code - QUANT_EXP_OFFSET;
    int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
    int total_shift = (max_oct - (E >> 3)) + norm_shift;

    for (int i = s; i < e; i++) {
      if (quant_in[i] == 0) {
        spec_q31[i] = 0;
        continue;
      }
      int sign = (quant_in[i] > 0) ? 1 : -1;
      int mag = (quant_in[i] > 0) ? quant_in[i] : -quant_in[i];
      int32_t dq_q8 = mag * 256 + Q8(QUANT_CENTROID);
      spec_q31[i] = sign * (int32_t)((int64_t)dq_q8 * step_m >> total_shift);
    }
  }

  *loss_bits_out = max_oct - headroom + 27;
}

int quant_gain_encode(float gain) {
  float v = log2f(gain > 1e-12f ? gain : 1e-12f) * QUANT_GAIN_Q;
  int code = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f)) + QUANT_GAIN_BIAS;
  return fxp_clamp_i32(code, 0, QUANT_GAIN_MAX_CODE);
}

void nf_run_length_fill(int16_t *quant,
                        const int32_t *exp_indices,
                        int gain_code,
                        int nf,
                        int32_t *spec_q31,
                        int *loss_bits_io) {
  if (nf >= 8) {
    return;
  }

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
  uint32_t seed = NF_SEED_BIAS;
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

        spec_q31[i - 2] = (seed & 0x8000) ? -fill_q31 : fill_q31;
      }
    }
  }

  *loss_bits_io = loss_bits;
}
