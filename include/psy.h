#ifndef HQLC_PSY_H
#define HQLC_PSY_H

#include <stdint.h>

#include "fxp.h"

#ifdef __cplusplus
extern "C" {
#endif

// 20 final exponent points
#define PSY_N_BANDS 20

// 48 fine exponent bands (used for analysis only)
#define PSY_N_FINE_BANDS  48
#define PSY_N_ACTIVE_FINE 47

// Only bins up to 427 are active (cutoff ~20 kHz at 48 kHz sample rate)
#define PSY_ACTIVE_BINS 427

// exp index is a 6-bit log-scale energy descriptor per band, ~1.5 dB/step
#define PSY_EXP_INDEX_MIN 0
#define PSY_EXP_INDEX_MAX 63

// Fine band edges, single-bin below 844Hz, ERB-spaced above
extern const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1];

// Coarse band edges, used for the final exponents & transmitted logic
extern const uint16_t psy_band_edges[PSY_N_BANDS + 1];

/**
 * @brief Compute spectral tilt for a target bitrate.
 *
 * The tilt is 35 dB at 128 kbps and above, then ramps down to a 15 dB floor at
 * lower bitrates.
 *
 * @param bitrate Target bitrate in bits per second.
 * @return Tilt in dB.
 */
int psy_tilt_for_bitrate(uint32_t bitrate);

/**
 * @brief Convert tilt in dB to a per-fine-band exponent step.
 *
 * @param tilt_db Tilt in dB.
 * @return Per-fine-band tilt step in EXP_Q7 format.
 */
int psy_tilt_step_q7(int tilt_db);

/**
 * @brief Compute coarse exponent indices from an MDCT spectrum.
 *
 * Computes fine-band PSDs, applies tilt, aggregates to coarse bands, and rounds
 * to 20 exponent indices.
 *
 * @param spectrum MDCT spectrum in Q31 BFP format, 512 bins.
 * @param tilt_step_q7 Per-fine-band tilt in EXP_Q7 format.
 * @param exp_indices Destination for 20 exponent indices in the range 0..63.
 */
void psy_fine_band_exponents(const bfp_i32 *spectrum,
                             int tilt_step_q7,
                             int32_t *exp_indices);

/**
 * @brief Mean power per bin for the 47 active fine bands.
 *
 * @param spec_q31 Raw spectrum mantissas. This is called from TNS module, so no bfp_i32.
 * at least PSY_ACTIVE_BINS
 * @param psd_q46 Destination, one Q46 power per fine band (shifted to Q23 then squared)
 */
void psy_fine_band_psd(const int32_t *spec_q31, uint64_t *psd_q46);

/**
 * Runs a 1-2-1 lowpass in-place on the provided fine band PSDs
 */
void psy_fine_band_psd_smooth(uint64_t *psd_q46);

#ifdef __cplusplus
}
#endif

#endif // HQLC_PSY_H
