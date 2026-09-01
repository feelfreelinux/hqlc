#include "psy.h"

#include "fxp.h"

// pre-calculated band edges, 20 bands + 1 end, ERB scale
// Stops at bin 427 (~20 kHz), zero'ed above that
const uint16_t psy_band_edges[PSY_N_BANDS + 1] = {
    0,  3,   8,   13,  19,  26,  34,  43,  52,  62,  75,
    89, 107, 127, 152, 180, 215, 255, 303, 360, 427,
};

// 48 fine bands, single-bin spaced below 844 Hz (18 bands), ERB-spaced above
const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15, 16,
    17,  18,  20,  23,  26,  29,  33,  37,  42,  47,  53,  59,  66,  74,  83,  92, 102,
    114, 127, 141, 157, 174, 193, 214, 238, 264, 293, 325, 361, 400, 427, 512,
};

int psy_tilt_for_bitrate(uint32_t bitrate) {
  // 35 dB at >=128 kbps, ramps down 5 dB per 32 kbps, floor at 15 dB
  if (bitrate >= 128000) {
    return 35;
  }
  int tilt = 35 - (int)((128000 - bitrate) * 5 / 32000);
  return (tilt < 15) ? 15 : tilt;
}

int psy_tilt_step_q7(int tilt_db) {
  // dB tilt to per-fine-band step in EXP_Q7, spread over the 47 active
  // fine bands at ~1.505 dB per exponent unit: step = tilt_db * 1.81
  return fxp_mul_rshift_rnd_i32(tilt_db, FXP_Q16(1.81), 16);
}

// Hat-basis weights, precomputed from the band edges: fine band fb contributes
// (256 - t) to coarse band k and t to band k + 1
static const struct {
  uint8_t k;
  uint16_t t;
} psy_hat_w[PSY_N_ACTIVE_FINE] = {
    {0, 0},    {0, 0},    {0, 0},    {0, 64},   {0, 128}, {0, 192},  {1, 0},    {1, 51},
    {1, 102},  {1, 153},  {1, 204},  {2, 0},    {2, 42},  {2, 85},   {2, 128},  {2, 170},
    {2, 213},  {3, 0},    {3, 85},   {3, 170},  {4, 32},  {4, 128},  {5, 0},    {5, 128},
    {6, 0},    {6, 142},  {7, 51},   {7, 204},  {8, 93},  {9, 18},   {9, 164},  {10, 64},
    {10, 224}, {11, 121}, {12, 23},  {12, 186}, {13, 85}, {13, 237}, {14, 132}, {15, 33},
    {15, 188}, {16, 87},  {16, 244}, {17, 142}, {18, 45}, {18, 198}, {18, 256},
};

// Per-coarse-band weight totals for psy_hat_w
static const int32_t psy_hat_ws[PSY_N_BANDS] = {
    1152, 1154, 1408, 1151, 607, 544, 498, 399, 418, 423,
    406,  423,  424,  399,  446, 423, 402, 445, 411, 499,
};

// Performs a 52-bit unsigned division, num / den, with den < 2^12, avoiding 64 bit div
static inline uint64_t psy_div_u52(uint64_t num, uint32_t den) {
  uint32_t hi = (uint32_t)(num >> 20);
  uint32_t lo = (uint32_t)num & 0xFFFFFu;
  uint32_t q_hi = hi / den;
  uint32_t rem = ((hi % den) << 20) | lo;
  return ((uint64_t)q_hi << 20) + rem / den;
}

void psy_fine_band_psd(const int32_t *spec_q31, uint64_t *psd_q46) {
  for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
    int s = psy_fine_band_edges[fb];
    int e = psy_fine_band_edges[fb + 1];
    int w = e - s;
    uint64_t sum_sq = 0;
    for (int i = s; i < e; i++) {
      uint32_t av = (uint32_t)fxp_abs_i32(spec_q31[i]) >> 8;
      sum_sq += (uint64_t)av * av;
    }
    // sum_sq < 2^46 * width, well under the 2^52 helper bound
    psd_q46[fb] = (w > 1) ? psy_div_u52(sum_sq, (uint32_t)w) : sum_sq;
  }
}

void psy_fine_band_psd_smooth(uint64_t *psd_q46) {
  uint64_t prev = psd_q46[0];
  for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
    uint64_t next = (fb + 1 < PSY_N_ACTIVE_FINE) ? psd_q46[fb + 1] : psd_q46[fb];
    uint64_t sm = (prev + 2 * psd_q46[fb] + next) >> 2;
    prev = psd_q46[fb];
    psd_q46[fb] = sm;
  }
}

void psy_fine_band_exponents(const bfp_i32 *spectrum,
                             int tilt_step_q7,
                             int32_t *exp_indices) {
  const int32_t *spec_q31 = spectrum->data;
  // Bias in EXP_Q7
  int32_t bias_index = 4 * spectrum->exp2 - 49;
  int32_t bias_q7 = bias_index * 128 + (bias_index < 0);

  uint64_t psd_q46[PSY_N_ACTIVE_FINE];
  psy_fine_band_psd(spec_q31, psd_q46);
  psy_fine_band_psd_smooth(psd_q46);

  // Hat-basis aggregation: split each fine band's log-PSD between the two nearest coarse
  // band centers, mirroring the quantizer's per-bin interpolation weights.
  // weight * lg sums stay well under 2^31, 32-bit accumulators suffice
  int32_t weighted_log_q7[PSY_N_BANDS] = {0};
  int32_t tilt_acc_q7 = 0;
  for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
    // Reinterpreting raw log2(PSD) Q8 as Q7 does the factor of two conversion
    int32_t log_index_q7 = (psd_q46[fb] == 0) ? 0 : fxp_log2_q8_u64(psd_q46[fb]);
    log_index_q7 += tilt_acc_q7;
    tilt_acc_q7 += tilt_step_q7;
    int k = psy_hat_w[fb].k;
    int32_t t = psy_hat_w[fb].t;
    weighted_log_q7[k] += (256 - t) * log_index_q7;
    weighted_log_q7[k + 1] += t * log_index_q7;
  }
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int32_t log_index_q7 = weighted_log_q7[b] / psy_hat_ws[b] + bias_q7;
    exp_indices[b] = fxp_clamp_i32(
        fxp_round_to_int(log_index_q7, 7), PSY_EXP_INDEX_MIN, PSY_EXP_INDEX_MAX);
  }
}
