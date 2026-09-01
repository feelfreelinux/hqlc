#include "quant.h"

#include "fxp.h"
#include "hqlc.h"
#include "hqlc_bench.h"
#include "psy.h"

// Bin at which we start to trigger NF - roughly ~3.5kHz
// Tuned empirically, but seems to match what other codecs do too
#define NF_START_BIN 75

// The first band the NF cliff fill triggers on (last few bands only)
#define NF_CLIFF_START_BAND 17

// Min drop for the NF cliff fill rule to trigger
#define NF_CLIFF_MIN_DROP 4

// A zeroed bin counts as overfilled when the estimator overshoots its energy by this
// ratio (equals to 6dB)
#define NF_FLAG_BIN_RATIO 2

// Quantizer scale: step = 2^(E/8), E = 2*exp - gain_code - 59
#define QUANT_EXP_OFFSET 59

// Q31 spectrum * Q28 inverse step -> Q8 quantizer domain
#define QUANT_TOTAL_Q 51

// Deadzone threshold and MMSE reconstruction centroid for a Laplacian source
#define QUANT_DZ_THRESH_Q8 (FXP_Q8(0.65) + 1)
#define QUANT_CENTROID_Q8  FXP_Q8(0.15)

// Quantizer step: step = 2^(E/8) with E = 2*exp - gain_code - QUANT_EXP_OFFSET,
// decomposed as pow2_eighth[E % 8] * 2^(E / 8)
// 2^(f/8) for f=0..7, Q30
static const int32_t quant_pow2_eighth_q30[8] = {
    1073741824,
    1170923762,
    1276901417,
    1392470869,
    1518500250,
    1655936265,
    1805811301,
    1969251188,
};

quant_scale quant_forward_scale(int exp_index, int gain_code, int spectrum_exp) {
  int inverse_step_eighths = gain_code + QUANT_EXP_OFFSET - 2 * exp_index;
  quant_scale scale = {
      .multiplier_q28 = quant_pow2_eighth_q30[inverse_step_eighths & 7] >> 2,
      .product_rshift = QUANT_TOTAL_Q - spectrum_exp - (inverse_step_eighths >> 3),
  };
  return scale;
}

// Reciprocals of the band-center distances, for the per-bin interpolation
static const uint16_t seg_rcp_q15[PSY_N_BANDS - 1] = {
    8192, 6553, 5461, 5461, 4096, 4096, 3640, 3276, 2978, 2340,
    2048, 1724, 1489, 1213, 1057, 862,  744,  630,  528,
};

// Sub-deadzone RMS of a Laplacian with zero-fraction z = (k + 0.5)/32, Q8 * step:
// L = -ln(1-z), rms/step = (0.65/L) * sqrt(2 - (1-z)*(L*L + 2*L + 2))
static const uint8_t nf_laplace_rms_q8[32] = {
    12, 21, 27, 31, 35, 39, 42, 45, 48, 50, 52, 54, 56, 58, 60, 61,
    63, 64, 65, 66, 67, 68, 68, 68, 69, 68, 68, 67, 66, 63, 59, 50,
};

// Per-bin exponents, linear ramp between band centers
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

/**
 * @brief Coarsen the exponent after a cliff for NF
 *
 * The 1-2-1 smoothing done on exponents in the psy biases the step a bit.
 * A louder exponent will influence the exponent next to it, up to +6dB.
 * Its needed for quantization, but we try to correct for it in NF, to prevent overshoot.
 */
static void bin_exp_cliff(const int32_t *exp_indices, int32_t *bin_exp) {
  for (int b = NF_CLIFF_START_BAND; b < PSY_N_BANDS; b++) {
    int drop = (int)exp_indices[b - 1] - (int)exp_indices[b];
    if (drop <= NF_CLIFF_MIN_DROP) {
      continue;
    }
    int fill_exp = (int)exp_indices[b] - 3 * (drop - NF_CLIFF_MIN_DROP) / 4;
    if (fill_exp < PSY_EXP_INDEX_MIN) {
      fill_exp = PSY_EXP_INDEX_MIN;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      bin_exp[i] = fill_exp;
    }
  }
}

// Deadzone, round, and restore sign.
static inline void quant_bin(int16_t *out, int32_t scaled_q8, int32_t sign) {
  int32_t dz_mask = ~((scaled_q8 - QUANT_DZ_THRESH_Q8) >> 31);
  int32_t q = ((scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8) & dz_mask;
  *out = (int16_t)((q ^ sign) - sign);
}

void quant_forward(const bfp_i32 *spectrum,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out) {
  const int32_t *spec_q31 = spectrum->data;
  for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
    quant_out[i] = 0;
  }

  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_QLOOP);

  int32_t bin_exp[PSY_ACTIVE_BINS];
  bin_exp_interp(exp_indices, bin_exp);

  int i = 0;
  while (i < PSY_ACTIVE_BINS) {
    int32_t be = bin_exp[i];
    int run_end = i + 1;
    while (run_end < PSY_ACTIVE_BINS && bin_exp[run_end] == be) {
      run_end++;
    }

    quant_scale scale = quant_forward_scale((int)be, gain_code, spectrum->exp2);

    if (scale.product_rshift >= 64) {
      // Step too coarse: the whole run quantizes to zero
      for (; i < run_end; i++) {
        quant_bin(&quant_out[i], 0, 0);
      }
    } else if (scale.product_rshift >= 32) {
      // Common case: only the high word of the product is needed
      int small_shift = scale.product_rshift - 32;
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = fxp_abs_i32(x);
        int32_t hi = fxp_mul_rshift_i32(abs_spec, scale.multiplier_q28, 32);
        int32_t scaled_q8 = hi >> small_shift;

        quant_bin(&quant_out[i], scaled_q8, sign);
      }
    } else if (scale.product_rshift > 0) {
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = fxp_abs_i32(x);
        int32_t scaled_q8 =
            (int32_t)((int64_t)abs_spec * scale.multiplier_q28 >> scale.product_rshift);

        quant_bin(&quant_out[i], scaled_q8, sign);
      }
    } else {
      // Saturation path, should be quite rare
      for (; i < run_end; i++) {
        int32_t x = spec_q31[i];
        int32_t sign = x >> 31;
        int32_t abs_spec = fxp_abs_i32(x);
        int32_t scaled_q8 = fxp_sat_i64_to_i32((int64_t)abs_spec * scale.multiplier_q28
                                               << (-scale.product_rshift));

        quant_bin(&quant_out[i], scaled_q8, sign);
      }
    }
  }
  HQLC_BENCH_END(HQLC_BENCH_ENC_QLOOP);
}

// Inverse quantizer: integer symbols to reconstructed BFP spectrum
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   bfp_i32 *spectrum) {
  int32_t *spec_q31 = spectrum->data;
  int max_exp = exp_indices[0];
  for (int b = 1; b < PSY_N_BANDS; b++) {
    if (exp_indices[b] > max_exp) {
      max_exp = exp_indices[b];
    }
  }
  int max_oct = (2 * max_exp - gain_code - QUANT_EXP_OFFSET) >> 3;

  // Interpolated exponents never exceed the band max, so max_oct holds
  int32_t bin_exp[PSY_ACTIVE_BINS];
  bin_exp_interp(exp_indices, bin_exp);

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
    int32_t dq_q8 = fxp_rescale_i32(mag, 0, 8) + QUANT_CENTROID_Q8;
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
    spectrum->exp2 = 0;
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
    int32_t dq_q8 = fxp_rescale_i32(mag, 0, 8) + QUANT_CENTROID_Q8;
    spec_q31[i] =
        (total_shift < 63) ? sign * (int32_t)((int64_t)dq_q8 * step_m >> total_shift) : 0;
  }

  spectrum->exp2 = max_oct - headroom + 27;
}

/**
 * @brief Estimate the noise fill level for one band, occupancy-derived
 *
 * @param nz Number of zeros in given band
 * @param bincount amount of bins a band spans
 */
static uint32_t nf_band_factor_q8(int nz, int bincount) {
  int32_t z_q15 = (nz << 15) / bincount;
  if (z_q15 >= FXP_Q15(1.0)) {
    // All-zero bands get the last entry of the level table
    z_q15 = FXP_Q15(1.0) - 1;
  }
  int32_t excess_q15 = (z_q15 << 1) - FXP_Q15(1.0);
  if (excess_q15 <= 0) {
    // Only apply NF if more than 50% of the bins are zero
    return 0;
  }

  // Return the NF estimation based on the lookup table
  return (uint32_t)fxp_mul_rshift_rnd_i32(nf_laplace_rms_q8[z_q15 >> 10], excess_q15, 15);
}

// Per-bin |X| in quantizer step units (Q8). quant_forward hoists this ladder
// out of its hot loop per bin_exp run; this is the cold per-bin form
static int32_t quant_abs_scaled_q8(int32_t abs_spec, quant_scale scale) {
  if (scale.product_rshift >= 64) {
    return 0;
  }
  if (scale.product_rshift >= 32) {
    int32_t hi = fxp_mul_rshift_i32(abs_spec, scale.multiplier_q28, 32);
    return hi >> (scale.product_rshift - 32);
  }
  if (scale.product_rshift > 0) {
    return (int32_t)((int64_t)abs_spec * scale.multiplier_q28 >> scale.product_rshift);
  }
  return fxp_sat_i64_to_i32((int64_t)abs_spec * scale.multiplier_q28
                            << (-scale.product_rshift));
}

void noise_fill_refinement_mask(const bfp_i32 *spectrum,
                                const int16_t *quant,
                                const int32_t *exp_indices,
                                int gain_code,
                                bool *mask) {
  const int32_t *spec_q31 = spectrum->data;
  // Same per-bin exponents the decoder fill will use
  int32_t bin_exp[PSY_ACTIVE_BINS];
  bin_exp_interp(exp_indices, bin_exp);
  bin_exp_cliff(exp_indices, bin_exp);

  for (int b = 0; b < PSY_N_BANDS; b++) {
    mask[b] = false;
    if (psy_band_edges[b] < NF_START_BIN) {
      continue;
    }
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];

    int nz = 0;
    // True NF fill leven per-bin
    int32_t true_nf_factor_q8[64];
    for (int i = s; i < e; i++) {
      if (quant[i] != 0) {
        continue;
      }

      // Apply quantizers scaling on the spectrum
      quant_scale scale = quant_forward_scale((int)bin_exp[i], gain_code, spectrum->exp2);
      true_nf_factor_q8[nz++] = quant_abs_scaled_q8(fxp_abs_i32(spec_q31[i]), scale);
    }
    if (nz == 0) {
      continue;
    }

    // Estimate the NF factor same way as decoder does
    uint32_t factor_q8 = nf_band_factor_q8(nz, e - s);
    if (factor_q8 == 0) {
      continue;
    }

    int bins_overshoot = 0;

    // Count the amount of bins that have overshooted the energy by factor of
    // NF_FLAG_BIN_RATIO
    for (int i = 0; i < nz; i++) {
      if ((int32_t)factor_q8 > NF_FLAG_BIN_RATIO * true_nf_factor_q8[i]) {
        bins_overshoot++;
      }
    }

    // We skip the band when move than half the bins overshooted
    mask[b] = 2 * bins_overshoot > nz;
  }
}

void noise_fill(const int16_t *quant,
                const int32_t *exp_indices,
                int gain_code,
                uint32_t seed,
                const bool *skip_bands,
                bfp_i32 *spectrum) {
  int32_t *spec_q31 = spectrum->data;
  int32_t bin_exp[PSY_ACTIVE_BINS];
  bin_exp_interp(exp_indices, bin_exp);
  bin_exp_cliff(exp_indices, bin_exp);

  // We derive the level from the zero occupancy in each band's quantized bins
  uint8_t factor_q8[PSY_N_BANDS] = {0};
  bool has_fill = false;
  int max_fill_oct = -32767;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    // NF_START_BIN sits on a band edge, so partial bands never occur
    if (psy_band_edges[b] < NF_START_BIN || (skip_bands && skip_bands[b])) {
      continue;
    }
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int nz = 0;
    int32_t max_zero_exp = INT32_MIN;
    for (int i = s; i < e; i++) {
      if (quant[i] == 0) {
        nz++;
        if (bin_exp[i] > max_zero_exp) {
          max_zero_exp = bin_exp[i];
        }
      }
    }
    if (nz == 0) {
      continue;
    }
    // Every zero bin is filled at the band's occupancy-derived level
    factor_q8[b] = (uint8_t)nf_band_factor_q8(nz, e - s);
    if (factor_q8[b] == 0) {
      continue;
    }
    int oct = (2 * (int)max_zero_exp - gain_code - QUANT_EXP_OFFSET) >> 3;
    if (oct > max_fill_oct) {
      max_fill_oct = oct;
    }
    has_fill = true;
  }
  if (!has_fill) {
    // Nothing to noise fill here
    return;
  }

  const int nf_guard_bits = 3;
  int fill_exp2 = max_fill_oct - nf_guard_bits;
  bfp_i32_coarsen(spectrum, fill_exp2);

  for (int b = 0; b < PSY_N_BANDS; b++) {
    // Skipped and pre-start bands never acquire a factor, so this gates both
    if (factor_q8[b] == 0) {
      continue;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      if (quant[i] != 0) {
        continue;
      }
      int E = 2 * (int)bin_exp[i] - gain_code - QUANT_EXP_OFFSET;
      int octave = E >> 3;
      int32_t step_m = quant_pow2_eighth_q30[E & 7] >> 2;
      int32_t fill_m = fxp_mul_rshift_i32(factor_q8[b], step_m, 8);
      int shift = spectrum->exp2 - octave - nf_guard_bits;
      int32_t fill_q31;
      if (shift >= 32) {
        fill_q31 = 0;
      } else if (shift > 0) {
        fill_q31 = fill_m >> shift;
      } else {
        fill_q31 = fxp_shl_sat_i32(fill_m, -shift);
      }

      // Marsaglia's xorshift
      // https://en.wikipedia.org/wiki/Xorshift
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;

      spec_q31[i] = (seed & 0x80000000u) ? -fill_q31 : fill_q31;
    }
  }
}
