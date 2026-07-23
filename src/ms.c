#include "ms.h"

#include "psy.h"


// Pre-shift to keep the energy calculations from overflowing within uint64
#define MS_GATE_ENERGY_SHIFT 7

// ratio for ms enter gate (1/4 = -6.02 dB,)
#define MS_GATE_ENTER_NUM 1ULL
#define MS_GATE_ENTER_DEN 4ULL

// Ratio for ms reset gate (13/32 = -3.91 dB)
#define MS_GATE_RESET_NUM 13ULL
#define MS_GATE_RESET_DEN 32ULL

// Align two BFP spectra into a shared exponent domain.
static int spec_bfp_align(int32_t *spec0_q31,
                          int32_t *spec1_q31,
                          int loss_bits0,
                          int loss_bits1,
                          int headroom_bits) {
  int target_loss = (loss_bits0 > loss_bits1) ? loss_bits0 : loss_bits1;
  target_loss += headroom_bits;

  int shift0 = target_loss - loss_bits0;
  int shift1 = target_loss - loss_bits1;
  for (int i = 0; i < PSY_ACTIVE_BINS; i++) {
    spec0_q31[i] >>= shift0;
    spec1_q31[i] >>= shift1;
  }

  return target_loss;
}

static bool ms_gate(const int32_t *spec0_q31,
                    const int32_t *spec1_q31,
                    int loss_bits0,
                    int loss_bits1,
                    bool *ms_flags) {
  // Calculate common loss bits and shift factors for both channels.
  int common_loss_bits = (loss_bits0 > loss_bits1) ? loss_bits0 : loss_bits1;
  int shift0 = common_loss_bits - loss_bits0;
  int shift1 = common_loss_bits - loss_bits1;

  bool any = false;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    uint64_t mid_energy = 0;
    uint64_t side_energy = 0;

    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      // Calculate the mid/side energy for each channel, shifted accordingly
      int64_t ch0 = ((int64_t)spec0_q31[i] >> shift0) >> MS_GATE_ENERGY_SHIFT;
      int64_t ch1 = ((int64_t)spec1_q31[i] >> shift1) >> MS_GATE_ENERGY_SHIFT;
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

void ms_encode(int32_t *spec0_q31,
               int32_t *spec1_q31,
               int *loss_bits0,
               int *loss_bits1,
               bool *ms_flags) {
  if (!ms_gate(spec0_q31, spec1_q31, *loss_bits0, *loss_bits1, ms_flags)) {
    return;
  }

  // Align BFP into same domain before replacing flagged L/R bands with M/S.
  *loss_bits0 = spec_bfp_align(spec0_q31, spec1_q31, *loss_bits0, *loss_bits1, 0);
  *loss_bits1 = *loss_bits0;

  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (!ms_flags[b]) {
      continue;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      int64_t ch0 = spec0_q31[i];
      int64_t ch1 = spec1_q31[i];
      spec0_q31[i] = (int32_t)((ch0 + ch1) >> 1); // M
      spec1_q31[i] = (int32_t)((ch0 - ch1) >> 1); // S
    }
  }
}

void ms_decode(int32_t *spec0_q31,
               int32_t *spec1_q31,
               int *loss_bits0,
               int *loss_bits1,
               bool *ms_flags) {
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
  *loss_bits0 = spec_bfp_align(spec0_q31, spec1_q31, *loss_bits0, *loss_bits1, 1);
  *loss_bits1 = *loss_bits0;

  for (int b = 0; b < PSY_N_BANDS; b++) {
    if (!ms_flags[b]) {
      continue;
    }
    for (int i = psy_band_edges[b]; i < psy_band_edges[b + 1]; i++) {
      int32_t mid = spec0_q31[i];
      int32_t side = spec1_q31[i];
      spec0_q31[i] = mid + side; // L
      spec1_q31[i] = mid - side; // R
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
