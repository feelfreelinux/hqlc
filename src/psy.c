#include "psy.h"

#include "fxp.h"

#define PSY_N_FINE_BANDS  48
#define PSY_N_ACTIVE_FINE 47

// pre-calculated band edges, 20 bands + 1 end, ERB scale
// Stops at bin 427 (~20 kHz), zero'ed above that
const uint16_t psy_band_edges[PSY_N_BANDS + 1] = {
    0,  3,   8,   13,  19,  26,  34,  43,  52,  62,  75,
    89, 107, 127, 152, 180, 215, 255, 303, 360, 427,
};

// 48 fine bands, single-bin spaced below 844 Hz (18 bands), ERB-spaced above
static const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15, 16,
    17,  18,  20,  23,  26,  29,  33,  37,  42,  47,  53,  59,  66,  74,  83,  92, 102,
    114, 127, 141, 157, 174, 193, 214, 238, 264, 293, 325, 361, 400, 427, 512,
};

// Maps psy_fine_band_edges to psy_band_edges (by center frequency)
static const uint8_t psy_fb_coarse[PSY_N_FINE_BANDS] = {
    0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,
    3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  7,  8,  9,  9,  10, 10,
    11, 12, 12, 13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, 19, 19,
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

// Fine bands per coarse band, divisor for the transient path
static const uint8_t fb_per_coarse[PSY_N_BANDS] = {
    3, 5, 5, 5, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
};

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

// The exponent-index domain uses 2*log2(PSD) in Q7. Reinterpreting the raw
// log2(PSD) Q8 result as Q7 performs that factor-of-two conversion exactly.
static inline int32_t psy_psd_to_log_index_q7(uint64_t psd) {
  return (psd == 0) ? 0 : fxp_log2_q8_u64(psd);
}

// 20 coarse-band exponents from 47 fine-band PSDs. Non-transient frames get
// 1-2-1 PSD smoothing + hat-basis aggregation, transient frames use a plain
// geometric mean (flat steps, attacks need a sharp envelope)
void psy_fine_band_exponents(const bfp_i32 *spectrum,
                             int tilt_step_q7,
                             int transient,
                             int32_t *exp_indices) {
  const int32_t *spec_q31 = spectrum->data;
  // Bias in EXP_Q7
  int32_t bias_index = 4 * spectrum->exp2 - 49;
  int32_t bias_q7 = bias_index * 128 + (bias_index < 0);

  // Per-fine-band mean power per bin
  uint64_t psd[PSY_N_ACTIVE_FINE];
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
    psd[fb] = (w > 1) ? psy_div_u52(sum_sq, (uint32_t)w) : sum_sq;
  }

  if (!transient) {
    // 1-2-1 low-pass across fine-band PSDs (linear domain, edges replicated).
    uint64_t prev = psd[0];
    for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
      uint64_t next = (fb + 1 < PSY_N_ACTIVE_FINE) ? psd[fb + 1] : psd[fb];
      uint64_t sm = (prev + 2 * psd[fb] + next) >> 2;
      prev = psd[fb];
      psd[fb] = sm;
    }
  }

  // Hat-basis aggregation: split each fine band's log-PSD between the two nearest coarse
  // band centers, mirroring the quantizer's per-bin interpolation weights
  if (!transient) {
    // weight * lg sums stay well under 2^31, 32-bit accumulators suffice
    int32_t weighted_log_q7[PSY_N_BANDS] = {0};
    int32_t tilt_acc_q7 = 0;
    for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
      int32_t log_index_q7 = psy_psd_to_log_index_q7(psd[fb]) + tilt_acc_q7;
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
    return;
  }

  // Simple geometric mean per coarse band
  int32_t log_index_sum_q7[PSY_N_BANDS] = {0};
  int32_t tilt_acc_q7 = 0;

  for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
    int32_t log_index_q7 = psy_psd_to_log_index_q7(psd[fb]);
    log_index_sum_q7[psy_fb_coarse[fb]] += log_index_q7 + tilt_acc_q7;
    tilt_acc_q7 += tilt_step_q7;
  }

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int32_t log_index_q7 = log_index_sum_q7[b] / fb_per_coarse[b] + bias_q7;
    exp_indices[b] = fxp_clamp_i32(
        fxp_round_to_int(log_index_q7, 7), PSY_EXP_INDEX_MIN, PSY_EXP_INDEX_MAX);
  }
}
