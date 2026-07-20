#include "psy.h"

#include "fxp.h"

// pre-calculated band edges, 20 bands + 1 end, ERB scale
// Stops at bin 427 (~20 kHz); bins 427-512 are zeroed (above audible range)
const uint16_t psy_band_edges[PSY_N_BANDS + 1] = {
    0,  3,   8,   13,  19,  26,  34,  43,  52,  62,  75,
    89, 107, 127, 152, 180, 215, 255, 303, 360, 427,
};

/* ── Fine-band exponent tables ── */

// 48 fine bands: single-bin below 844 Hz (18 bands), ERB-spaced above
const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15, 16,
    17,  18,  20,  23,  26,  29,  33,  37,  42,  47,  53,  59,  66,  74,  83,  92, 102,
    114, 127, 141, 157, 174, 193, 214, 238, 264, 293, 325, 361, 400, 427, 512,
};

// Maps each fine band to its parent coarse band (by center frequency)
static const uint8_t psy_fb_coarse[PSY_N_FINE_BANDS] = {
    0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,
    3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  7,  8,  9,  9,  10, 10,
    11, 12, 12, 13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, 19, 19,
};

// Number of active fine bands (excludes the inaudible [427,512) tail)
#define PSY_N_ACTIVE_FINE 47

int psy_tilt_for_bitrate(uint32_t bitrate) {
  // 35 dB at >=128 kbps, ramps down 5 dB per 32 kbps, floor at 15 dB
  if (bitrate >= 128000) {
    return 35;
  }
  int tilt = 35 - (int)((128000 - bitrate) * 5 / 32000);
  return (tilt < 15) ? 15 : tilt;
}

// EXP_Q7: working format for exponent math, 128 units per exponent index.
// Since exp = round(2*log2(psd) + bias) and 2*log2(x)*128 = log2(x)*256,
// the Q8 output of fxp_log2_q8_u64 is already in EXP_Q7.
#define EXP_Q7_FRAC_BITS 7
#define EXP_Q7(x)        ((int32_t)((x) * 128.0 + 0.5))

int psy_tilt_step_q7(int tilt_db) {
  // dB tilt to per-fine-band step in EXP_Q7, spread over the 47 active
  // fine bands at ~1.505 dB per exponent unit: step ~= tilt_db * 1.81
  return (tilt_db * 118612 + 32768) >> 16;
}

// Fine bands per coarse band, divisor for the transient path
static const uint8_t fb_per_coarse[PSY_N_BANDS] = {
    3, 5, 5, 5, 3, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2,
};

// 20 coarse-band exponents from 47 fine-band PSDs. Non-transient frames get
// 1-2-1 PSD smoothing + hat-basis aggregation; transient frames use a plain
// geometric mean (flat synthesis steps, attacks need a sharp envelope).
void psy_fine_band_exponents(const int32_t *spec_q31,
                             int loss_bits,
                             int tilt_step,
                             int transient,
                             int32_t *exp_indices) {
  int32_t bias = EXP_Q7(4 * loss_bits - 49);

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
    psd[fb] = (w > 1) ? sum_sq / (unsigned)w : sum_sq;
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

  // Hat-basis aggregation: split each fine band's log-PSD between the two
  // nearest coarse band centers, mirroring the quantizer's per-bin
  // interpolation weights.
  if (!transient) {
    int32_t centers[PSY_N_BANDS];
    for (int b = 0; b < PSY_N_BANDS; b++) {
      centers[b] = (psy_band_edges[b] + psy_band_edges[b + 1] + 2) >> 1;
    }
    int64_t wl[PSY_N_BANDS] = {0};
    int32_t ws[PSY_N_BANDS] = {0};
    int32_t tilt_acc = 0;
    int k = 0;
    for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
      int32_t lg = ((psd[fb] == 0) ? 0 : fxp_log2_q8_u64(psd[fb])) + tilt_acc;
      tilt_acc += tilt_step;
      int x = (psy_fine_band_edges[fb] + psy_fine_band_edges[fb + 1]) >> 1;
      if (x <= centers[0]) {
        wl[0] += 256 * (int64_t)lg;
        ws[0] += 256;
      } else if (x >= centers[PSY_N_BANDS - 1]) {
        wl[PSY_N_BANDS - 1] += 256 * (int64_t)lg;
        ws[PSY_N_BANDS - 1] += 256;
      } else {
        while (x >= centers[k + 1]) {
          k++;
        }
        int32_t t = (x - centers[k]) * 256 / (centers[k + 1] - centers[k]);
        wl[k] += (256 - t) * (int64_t)lg;
        ws[k] += 256 - t;
        wl[k + 1] += t * (int64_t)lg;
        ws[k + 1] += t;
      }
    }
    for (int b = 0; b < PSY_N_BANDS; b++) {
      int32_t v = (ws[b] > 0) ? (int32_t)(wl[b] / ws[b]) : 0;
      int32_t exp_q7 = v + bias;
      exp_indices[b] = fxp_clamp_i32(fxp_round_to_int(exp_q7, EXP_Q7_FRAC_BITS),
                                     PSY_EXP_INDEX_MIN,
                                     PSY_EXP_INDEX_MAX);
    }
    return;
  }
  // Transient path: plain geometric mean per coarse band
  int32_t log_sum[PSY_N_BANDS] = {0};
  int32_t tilt_acc = 0;

  for (int fb = 0; fb < PSY_N_ACTIVE_FINE; fb++) {
    int32_t log_psd = (psd[fb] == 0) ? 0 : fxp_log2_q8_u64(psd[fb]);
    log_sum[psy_fb_coarse[fb]] += log_psd + tilt_acc;
    tilt_acc += tilt_step;
  }

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int32_t exp_q7 = log_sum[b] / fb_per_coarse[b] + bias;
    exp_indices[b] = fxp_clamp_i32(
        fxp_round_to_int(exp_q7, EXP_Q7_FRAC_BITS), PSY_EXP_INDEX_MIN, PSY_EXP_INDEX_MAX);
  }
}
