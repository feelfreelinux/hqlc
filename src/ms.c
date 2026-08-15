#include "ms.h"

#include "psy.h"

// S-channel exponent bias on flagged bands:
//   bias = clamp(exp0 - exp1 - MS_BIAS_GAP, 0, MS_BIAS_CAP)
#define MS_BIAS_GAP 2
#define MS_BIAS_CAP 6

// Pre-shift to keep the energy calculations from overflowing within uint64
#define MS_GATE_ENERGY_SHIFT 7

// ratio for ms enter gate (1/4 = -6.02 dB,)
#define MS_GATE_ENTER_NUM 1ULL
#define MS_GATE_ENTER_DEN 4ULL

// Ratio for ms reset gate (13/32 = -3.91 dB)
#define MS_GATE_RESET_NUM 13ULL
#define MS_GATE_RESET_DEN 32ULL

static bool ms_gate(const bfp_i32 *channel0, const bfp_i32 *channel1, bool *ms_flags) {
  bfp_alignment alignment = bfp_i32_alignment(channel0, channel1, 0);

  bool any = false;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    uint64_t mid_energy = 0;
    uint64_t side_energy = 0;

    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      // Calculate the mid/side energy for each channel, shifted accordingly
      int64_t ch0 =
          ((int64_t)channel0->data[i] >> alignment.a_rshift) >> MS_GATE_ENERGY_SHIFT;
      int64_t ch1 =
          ((int64_t)channel1->data[i] >> alignment.b_rshift) >> MS_GATE_ENERGY_SHIFT;
      int64_t mid = ch0 + ch1;
      int64_t side = ch0 - ch1;

      mid_energy += (uint64_t)(mid * mid);
      side_energy += (uint64_t)(side * side);
    }

    bool enter_gate = side_energy * MS_GATE_ENTER_DEN < MS_GATE_ENTER_NUM * mid_energy;
    bool exit_gate = side_energy * MS_GATE_RESET_DEN > MS_GATE_RESET_NUM * mid_energy;

    if (mid_energy == 0) {
      ms_flags[b] = false;
    } else if (enter_gate) {
      ms_flags[b] = true;
    } else if (exit_gate) {
      ms_flags[b] = false;
    }
    any |= ms_flags[b];
  }
  return any;
}

void ms_encode(bfp_i32 *channel0, bfp_i32 *channel1, bool *ms_flags) {
  if (!ms_gate(channel0, channel1, ms_flags)) {
    return;
  }

  // Align BFP into same domain before replacing flagged L/R bands with M/S.
  bfp_i32_align_pair(channel0, channel1, 0);

  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (!ms_flags[b]) {
      continue;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      int64_t ch0 = channel0->data[i];
      int64_t ch1 = channel1->data[i];
      channel0->data[i] = (int32_t)((ch0 + ch1) >> 1); // M
      channel1->data[i] = (int32_t)((ch0 - ch1) >> 1); // S
    }
  }
}

void ms_decode(bfp_i32 *mid, bfp_i32 *side, const bool *ms_flags) {
  bool has_ms = false;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (ms_flags[b]) {
      has_ms = true;
      break;
    }
  }
  if (!has_ms) {
    // early return if no MS bands are flagged, so we don't unnecessarily align the
    // spectra
    return;
  }

  // Give headroom for the M/S bands after quant inverse.
  bfp_i32_align_pair(mid, side, 1);

  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (!ms_flags[b]) {
      continue;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      int32_t mid_value = mid->data[i];
      int32_t side_value = side->data[i];
      mid->data[i] = mid_value + side_value; // L
      side->data[i] = mid_value - side_value; // R
    }
  }
}

void ms_apply_side_exp_bias(int32_t *exp0, int32_t *exp1, const bool *ms_flags) {
  // Coarsen S exponents for flagged bands.
  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (!ms_flags[b]) {
      continue;
    }

    int exp_gap = (int)exp0[b] - (int)exp1[b];
    int side_bias = exp_gap - MS_BIAS_GAP;
    if (side_bias > MS_BIAS_CAP) {
      side_bias = MS_BIAS_CAP;
    }
    if (side_bias > 0) {
      exp1[b] += side_bias;
      if (exp1[b] > PSY_EXP_INDEX_MAX) {
        exp1[b] = PSY_EXP_INDEX_MAX;
      }
    }
  }
}
