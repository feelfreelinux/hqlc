#include "psy.h"

#include "fxp.h"

// pre-calculated band edges, 20 bands + 1 end, ERB scale
const uint16_t psy_band_edges[PSY_N_BANDS + 1] = {
    0,   3,   8,   13,  19,  26,  34,  43,  55,  67,  82,
    101, 123, 149, 179, 214, 255, 304, 362, 431, 512,
};

/*
 * @brief Precomputed 2*log2(band_width) in Q4
 *
 * For band b with width w: entry = round(2 * log2(w) * 16).
 * Used to convert sum-of-squares to mean-square in log domain:
 *   2*log2(mean_sq) = 2*log2(sum_sq) - 2*log2(w)
 */
static const int16_t two_log2_bw_q4[PSY_N_BANDS] = {
    51,  74,  74,  83,  90,  96,  101, 115, 115, 125,
    136, 143, 150, 157, 164, 171, 180, 187, 195, 203,
};

// log2(bin_count) in Q4, per band
static const int32_t psy_log2_bins_q4[PSY_N_BANDS] = {
    25,  /* b= 0: w=3  */
    37,  /* b= 1: w=5  */
    37,  /* b= 2: w=5  */
    41,  /* b= 3: w=6  */
    45,  /* b= 4: w=7  */
    48,  /* b= 5: w=8  */
    51,  /* b= 6: w=9  */
    57,  /* b= 7: w=12 */
    57,  /* b= 8: w=12 */
    63,  /* b= 9: w=15 */
    68,  /* b=10: w=19 */
    71,  /* b=11: w=22 */
    75,  /* b=12: w=26 */
    79,  /* b=13: w=30 */
    82,  /* b=14: w=35 */
    86,  /* b=15: w=41 */
    90,  /* b=16: w=49 */
    94,  /* b=17: w=58 */
    98,  /* b=18: w=69 */
    101, /* b=19: w=81 */
};

// Per-adjacent-band upward masking decay in Q4 (ERB-rate based)
static const int32_t psy_spread_decay_up_q4[PSY_N_BANDS] = {
    0,   -240, -194, -151, -134, -120, -108, -107, -99, -92,
    -96, -95,  -92,  -90,  -87,  -85,  -85,  -85,  -85, -85,
};

// Per-adjacent-band downward masking decay in Q4
static const int32_t psy_spread_decay_down_q4[PSY_N_BANDS] = {
    -361, -291, -227, -201, -179, -162, -161, -149, -138, -144,
    -143, -138, -134, -130, -128, -128, -128, -128, -127, 0,
};

#define PSY_Q4_NEG_INF (-10000)

void psy_spreading_envelope(const int32_t *exp_indices, int32_t *smr_q4_out) {
  // PSD-level energy per band in Q4 log2
  int32_t energy_q4[PSY_N_BANDS];
  for (int b = 0; b < PSY_N_BANDS; b++) {
    energy_q4[b] = ((int)exp_indices[b] - PSY_EXP_INDEX_BIAS) * 8;
  }

  // Convert PSD to total band energy for cross-band spreading
  int32_t total_q4[PSY_N_BANDS];
  for (int b = 0; b < PSY_N_BANDS; b++) {
    total_q4[b] = energy_q4[b] + psy_log2_bins_q4[b];
  }

  // Forward sweep, upward masking (from left)
  int32_t mask_left[PSY_N_BANDS];
  mask_left[0] = PSY_Q4_NEG_INF;
  for (int b = 1; b < PSY_N_BANDS; b++) {
    int32_t prev = fxp_max_i32(total_q4[b - 1], mask_left[b - 1]);
    mask_left[b] = prev + psy_spread_decay_up_q4[b];
  }

  // Backward sweep, downward masking (from right)
  int32_t mask_right[PSY_N_BANDS];
  mask_right[PSY_N_BANDS - 1] = PSY_Q4_NEG_INF;
  for (int b = PSY_N_BANDS - 2; b >= 0; b--) {
    int32_t nxt = fxp_max_i32(total_q4[b + 1], mask_right[b + 1]);
    mask_right[b] = nxt + psy_spread_decay_down_q4[b];
  }

  // Cross-band mask in total energy, convert back to PSD, compute SMR
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int32_t spread_total = fxp_max_i32(mask_left[b], mask_right[b]);
    int32_t mask_q4 = spread_total - psy_log2_bins_q4[b];
    smr_q4_out[b] = energy_q4[b] - mask_q4;
  }
}

void psy_band_analysis(const int32_t *spec_q31,
                       int loss_bits,
                       int32_t *exp_indices,
                       uint64_t *band_energy,
                       uint32_t *band_peak) {
  // idx = round(2*log2(mean_sq_true)) + BIAS
  //     = round(2*log2(sum_sq) - 2*log2(w) + 4*loss_bits - 92 + BIAS)
  // With BIAS=43: constant = 92 - 43 = 49
  int32_t bias_q4 = (4 * loss_bits - (92 - PSY_EXP_INDEX_BIAS)) * 16;

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];

    uint64_t sum_sq = 0;
    uint32_t peak = 0;

    for (int i = s; i < e; i++) {
      uint32_t av = (uint32_t)fxp_abs_i32(spec_q31[i]) >> 8;
      if (av > peak) {
        peak = av;
      }
      sum_sq += (uint64_t)av * av;
    }

    band_energy[b] = sum_sq;
    band_peak[b] = peak;

    if (sum_sq == 0) {
      exp_indices[b] = PSY_EXP_INDEX_MIN;
      continue;
    }

    int32_t idx_q4 = fxp_log2_q5_u64(sum_sq) - two_log2_bw_q4[b] + bias_q4;
    int32_t idx = (idx_q4 + 8) >> 4;

    idx = fxp_clamp_i32(idx, PSY_EXP_INDEX_MIN, PSY_EXP_INDEX_MAX);

    exp_indices[b] = idx;
  }
}
