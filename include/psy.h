#ifndef HQLC_PSY_H
#define HQLC_PSY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PSY_N_BANDS 20

extern const uint16_t psy_band_edges[PSY_N_BANDS + 1];

// Only bins 0..426 are active (~20 kHz at 48 kHz sample rate)
#define PSY_ACTIVE_BINS 427
#define PSY_MAX_BAND_WIDTH 67

// Exponent index: 6-bit log-scale energy descriptor per band.
// Quantizer step = 2^((idx - BIAS) / 4), giving ~1.5 dB per index.
#define PSY_EXP_INDEX_BIAS 43
#define PSY_EXP_INDEX_MIN  0
#define PSY_EXP_INDEX_MAX  63

// 48 fine bands for exponent computation (single-bin LF, ERB-spaced HF)
#define PSY_N_FINE_BANDS 48

extern const uint16_t psy_fine_band_edges[PSY_N_FINE_BANDS + 1];

/**
 * @brief Compute tilt dB for a given bitrate.
 *
 * 35 dB at >=128 kbps, linear ramp down to 15 dB floor.
 */
int psy_tilt_for_bitrate(uint32_t bitrate);

/**
 * @brief Per-fine-band tilt step in EXP_Q7 (128 per exponent unit).
 *
 * In the log domain, tilt[fb] = 2^(fb * step / 128).
 * The step is an exact integer — no accumulation error.
 */
int psy_tilt_step_q7(int tilt_db);

/**
 * @brief Compute 20 exponent indices from the MDCT spectrum.
 *
 * Pipeline per frame:
 *   47 fine-band PSD → log2 + per-fine-band tilt (log domain) →
 *   average per coarse band → round.
 *
 * Tilt accumulates per fine band so HF pre-emphasis is continuous
 * across the spectrum.  All post-log arithmetic uses EXP_Q7
 * (128 per exponent unit); fxp_log2_q8(x) = log2(x)*256 = EXP_Q7.
 *
 * @param spec_q31    MDCT spectrum (Q31 BFP, 512 bins)
 * @param loss_bits   BFP exponent
 * @param tilt_step   Per-fine-band tilt in EXP_Q7 (from psy_tilt_step_q7)
 * @param transient   Nonzero on TNS-eligible frames (gates analysis smoothing)
 * @param exp_indices Output: 20 exponent indices [0..63]
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
