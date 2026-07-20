#ifndef HQLC_PSY_H
#define HQLC_PSY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 20 exponent points
#define PSY_N_BANDS 20

extern const uint16_t psy_band_edges[PSY_N_BANDS + 1];

// Only bins up to 427 are active (cutoff ~20 kHz at 48 kHz sample rate)
#define PSY_ACTIVE_BINS 427

// Exponent index: 6-bit log-scale energy descriptor per band, ~1.5 dB/step
#define PSY_EXP_INDEX_MIN 0
#define PSY_EXP_INDEX_MAX 63

// 48 fine bands for exponent computation (single-bin LF, ERB-spaced HF);
// the last band [427,512) is above the coded range and never analyzed
#define PSY_N_FINE_BANDS 48

extern const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1];

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
 * @param spec_q31 MDCT spectrum in Q31 BFP format, 512 bins.
 * @param loss_bits BFP exponent for `spec_q31`.
 * @param tilt_step Per-fine-band tilt in EXP_Q7 format.
 * @param transient Nonzero for TNS-eligible transient frames.
 * @param exp_indices Destination for 20 exponent indices in the range 0..63.
 */
void psy_fine_band_exponents(const int32_t *spec_q31,
                             int loss_bits,
                             int tilt_step,
                             int transient,
                             int32_t *exp_indices);

#ifdef __cplusplus
}
#endif

#endif // HQLC_PSY_H
