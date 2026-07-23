#ifndef HQLC_MS_H
#define HQLC_MS_H

#include <stdbool.h>
#include <stdint.h>

// S-channel exponent bias on flagged bands:
//   bias = clamp(exp0 - exp1 - MS_BIAS_GAP, 0, MS_BIAS_CAP)
#define MS_BIAS_GAP 2
#define MS_BIAS_CAP 6

// Flagged S bands use biased exponents. At the cap, bias=6 raises the quantizer
// step by 2^(6/4), shrinking |q| by 1.5. One alpha bin spans about 2647/12 Q8 = 0.862, so 1.5/0.862 rounded up is 2
#define MS_RANS_ALPHA_SHIFT 2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Replace flagged L/R bands with M/S in place.
 *
 * Aligns both spectra to a common BFP exponent before the butterfly. Unflagged
 * bands remain L/R, so the output is a per-band L/M and R/S patchwork.
 */
void ms_encode(int32_t *spec0_q31,
               int32_t *spec1_q31,
               int *loss_bits0,
               int *loss_bits1,
               bool *ms_flags);

/**
 * @brief Replace flagged M/S bands with L/R in place.
 *
 * Aligns both reconstructed spectra to a common exponent with one extra bit of
 * headroom before summing/differencing. Unflagged bands pass through unchanged.
 */
void ms_decode(int32_t *spec0_q31,
               int32_t *spec1_q31,
               int *loss_bits0,
               int *loss_bits1,
               bool *ms_flags);

/**
 * @brief Coarsen flagged side-channel exponents to match the M/S bit allocation.
 *
 * The biased side exponents are transmitted normally, so this requires no
 * decoder-side signalling beyond the per-band M/S flags.
 */
void ms_apply_side_exp_bias(int32_t *exp0, int32_t *exp1, const bool *ms_flags);

#ifdef __cplusplus
}
#endif

#endif // HQLC_MS_H
