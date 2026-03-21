#include "quant.h"

#include <math.h>

#include "fxp.h"
#include "hqlc.h"
#include "psy.h"

// Noise fill parameters for synthesis
#define NF_MUL_Q30       Q30(0.7)                            // NF amplitude multiplier
#define NF_LCG_A         1664525u                            // Knuth LCG multiplier
#define NF_LCG_C         1013904223u                         // Knuth LCG increment
#define NF_L1_INV_K_1024 ((int32_t)(0.8660254 * 1024 + 0.5)) // sqrt(3)/2 * 1024
#define NF_EXP_OFFSET    (2 * PSY_EXP_INDEX_BIAS)            // 86

// 2^(f/8) for f=0..7, in Q30, LUT
const int32_t quant_pow2_eighth_q30[8] = {1073741824,
                                          1170923762,
                                          1276901417,
                                          1392470869,
                                          1518500250,
                                          1655936265,
                                          1805811301,
                                          1969251188};

// Band weights: BW[b] = log2(bin_count[b]) / log2(81), clipped to [0.10, 1.0].
// Wider bands get coarser quantization steps. In Q30.
static const int32_t quant_bw_q30[20] = {
    268435456, 393250835, 393250835, 437799372, 475464623,  508091748,  536870912,
    607163288, 607163288, 661686291, 719445639, 755266803,  796084878,  831050207,
    868715459, 907375988, 950929247, 992130602, 1034563624, 1073741824,
};

// 1/BW[b] in Q28 (precomputed to avoid per-band division at runtime).
const int32_t quant_inv_bw_q28[20] = {
    1073741824, 732942820, 732942820, 658361785, 606207827, 567280175, 536870912,
    474716409,  474716409, 435599740, 400628429, 381627228, 362059856, 346826670,
    331789164,  317652638, 303103913, 290516567, 278600919, 268435456,
};

void quant_forward(const int32_t *spec_q31,
                   int loss_bits,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out) {
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];

    int E = 2 * (int)exp_indices[b] - gain_code - QUANT_EXP_OFFSET;
    int neg_E = -E;

    // Split -E into integer octaves and fractional eighth.
    int int_part = (neg_E >= 0) ? (neg_E / 8) : ((neg_E - 7) / 8);
    int frac = neg_E - 8 * int_part; // always in [0..7]

    // inv_step mantissa in Q28: 2^(frac/8) * inv_bw[b]
    // Q30 * Q28 = Q58, >>30 to Q28
    int32_t inv_step_m =
        (int32_t)((int64_t)quant_pow2_eighth_q30[frac] * quant_inv_bw_q28[b] >> 30);

    int total_shift = QUANT_TOTAL_Q - loss_bits - int_part;

    if (total_shift >= 64) {
      for (int i = s; i < e; i++) {
        quant_out[i] = 0;
      }
    } else if (total_shift >= 32) {
      // Common case, only need high word of 64-bit product.
      int small_shift = total_shift - 32;
      for (int i = s; i < e; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;

        int32_t hi = (int32_t)((int64_t)abs_spec * inv_step_m >> 32);
        int32_t scaled_q8 = hi >> small_shift;

        int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
        int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;

        // Branchless sign restore
        quant_out[i] = (int16_t)((q ^ sign) - sign);
      }
    } else if (total_shift > 0) {
      for (int i = s; i < e; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;

        int32_t scaled_q8 = (int32_t)((int64_t)abs_spec * inv_step_m >> total_shift);

        int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
        int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;

        quant_out[i] = (int16_t)((q ^ sign) - sign);
      }
    } else {
      // Saturation path, should be quite rare
      int neg_shift = -total_shift;
      for (int i = s; i < e; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = (x ^ sign) - sign;

        int32_t scaled_q8 =
            fxp_sat_i64_to_i32((int64_t)abs_spec * inv_step_m << neg_shift);

        int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
        int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;

        quant_out[i] = (int16_t)((q ^ sign) - sign);
      }
    }
  }
}

void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   int32_t *spec_q31,
                   int *loss_bits_out,
                   int64_t *work) {
  // find max e_int across all bands for alignment
  int max_e_int = -32767;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int E = 2 * (int)exp_indices[b] - gain_code - QUANT_EXP_OFFSET;
    int e_int = (E >= 0) ? (E / 8) : ((E - 7) / 8);
    if (e_int > max_e_int) {
      max_e_int = e_int;
    }
  }

  uint64_t or_acc = 0;

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];

    int E = 2 * (int)exp_indices[b] - gain_code - QUANT_EXP_OFFSET;
    int e_int = (E >= 0) ? (E / 8) : ((E - 7) / 8);
    int e_frac = E - 8 * e_int; // always in [0..7]

    // step mantissa = (2^(e_frac/8) * bw[b]) in Q30.
    // quant_pow2_eighth_q30[e_frac] is Q30, quant_bw_q30[b] is Q30.
    // Product is Q60; shift right 30 to Q30.
    int32_t step_m_q30 =
        (int32_t)((int64_t)quant_pow2_eighth_q30[e_frac] * quant_bw_q30[b] >> 30);

    // how many bits to right-shift this band's values so they share the max_e_int
    // exponent
    int align_shift = max_e_int - e_int;

    for (int i = s; i < e; i++) {
      if (quant_in[i] == 0) {
        work[i] = 0;
        continue;
      }

      int sign = (quant_in[i] > 0) ? 1 : -1;
      int mag = (quant_in[i] > 0) ? quant_in[i] : -quant_in[i];

      // Reconstruct magnitude in Q8, adding mid-point bias 0.15
      int32_t dq_q8 = mag * 256 + Q8(0.15);

      // Q8 * Q30 = Q38 (true_value = recon * 2^(e_int - QUANT_EXP_OFFSET))
      int64_t recon = (int64_t)dq_q8 * step_m_q30;

      // Align to max_e_int and apply sign
      int64_t val = (int64_t)sign * (recon >> align_shift);
      work[i] = val;
      or_acc |= (val >= 0) ? (uint64_t)val : (uint64_t)(-val);
    }
  }

  int headroom;
  if (or_acc == 0) {
    // All-zero frame
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = 0;
    }
    *loss_bits_out = 0;
    return;
  }

  // headroom = number of unused sign bits above the MSB
  headroom = (int)__builtin_clzll(or_acc) - 1;

  // Normalize int64 to int32 Q31.
  //
  // MSB position in or_acc = 63 - clzll(or_acc) = 62 - headroom.
  // We want MSB at bit 30 (signed int32), so right-shift by: shift = (62 - headroom) - 30
  // = 32 - headroom Positive shift = right-shift (large values), negative = left-shift.
  int shift = 32 - headroom;

  // Hoist shift-direction branch outside the 512-sample loop
  if (shift > 0) {
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = (int32_t)(work[i] >> shift);
    }
  } else if (shift == 0) {
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = (int32_t)work[i];
    }
  } else {
    int neg_shift = -shift;
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      spec_q31[i] = fxp_sat_i64_to_i32(work[i] << neg_shift);
    }
  }

  // loss_bits = 31 + shift + max_e_int - QUANT_EXP_OFFSET = 31 + (32 - headroom) +
  // max_e_int - QUANT_EXP_OFFSET = max_e_int - headroom + 25
  int loss_bits = max_e_int - headroom + (31 + 32 - QUANT_EXP_OFFSET);

  *loss_bits_out = loss_bits;
}

int quant_gain_encode(float gain) {
  float v = log2f(gain > 1e-12f ? gain : 1e-12f) * QUANT_GAIN_Q;
  int code = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f)) + QUANT_GAIN_BIAS;
  code = fxp_clamp_i32(code, 0, QUANT_GAIN_MAX_CODE);
  return code;
}

// Computes the amplitude of noise fill in Q31 for a given exponent index and loss_bits
int32_t nf_compute_amp_q31(int exp_idx, int loss_bits) {
  // nf_amp = 0.7 * 2^((exp_idx - PSY_EXP_INDEX_BIAS) / 4), in eight oct units
  int nf_eighth = 2 * exp_idx - NF_EXP_OFFSET;
  int e_int = (nf_eighth >= 0) ? (nf_eighth / 8) : ((nf_eighth - 7) / 8);
  int e_frac = nf_eighth - 8 * e_int;

  // mantissa is Q60 (Q30 * Q30), target is Q31 with BFP exponent loss_bits
  // right_shift = Q60 - Q31 + loss_bits - e_int = 29 + loss_bits - e_int
  int64_t mantissa = (int64_t)NF_MUL_Q30 * quant_pow2_eighth_q30[e_frac];
  int right_shift = (60 - 31) + loss_bits - e_int;

  if (right_shift >= 64) {
    return 0;
  }
  if (right_shift <= 0) {
    return fxp_sat_i64_to_i32(mantissa);
  }

  return (int32_t)(mantissa >> right_shift);
}

// Performs noise fill synthesis on a band of the spectrum, scaling noise to nf_amp_q31
void nf_fill_band(int32_t *spec, int start, int end, int32_t nf_amp_q31, uint32_t seed) {
  int n = end - start;
  if (n <= 0 || nf_amp_q31 <= 0) {
    return;
  }

  // Generate LCG noise, accumulate mean
  int32_t raw[PSY_MAX_BAND_WIDTH];
  int64_t sum = 0;
  uint32_t x = seed;
  for (int i = 0; i < n; i++) {
    x = NF_LCG_A * x + NF_LCG_C;
    raw[i] = (int32_t)((x >> 8) & 0xFFFFFF) - 0x800000;
    sum += raw[i];
  }

  // Mean-subtract, compute L1 norm
  int32_t mean_val = (int32_t)(sum / n);
  uint32_t abs_sum = 0;
  for (int i = 0; i < n; i++) {
    raw[i] -= mean_val;
    abs_sum += (uint32_t)(raw[i] < 0 ? -raw[i] : raw[i]);
  }

  if (abs_sum == 0) {
    for (int i = 0; i < n; i++) {
      spec[start + i] = 0;
    }
    return;
  }

  // Scale, output_q31 = raw * nf_amp_q31 * n * (sqrt(3)/2) / abs_sum
  int64_t scale_num = (int64_t)nf_amp_q31 * n * NF_L1_INV_K_1024;
  int64_t scale_q30 = (scale_num << 20) / (int64_t)abs_sum;
  if (scale_q30 > (int64_t)INT32_MAX) {
    scale_q30 = INT32_MAX;
  }
  int32_t scale = (int32_t)scale_q30;

  for (int i = 0; i < n; i++) {
    spec[start + i] = (int32_t)((int64_t)raw[i] * scale >> 30);
  }
}
