#ifndef HQLC_PSY_H
#define HQLC_PSY_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Band count
#define PSY_N_BANDS 20

// Band edges definition
extern const uint16_t psy_band_edges[PSY_N_BANDS + 1];

// Maximum width of any single band in bins (last band: 512 - 431 = 81)
#define PSY_MAX_BAND_WIDTH 81

// Exponent index bias - applied to the calculated exponent index, in order to move it
// around the perceptually valid range Calculated empirically to minimize clipping It
// further controls the quantizer step - step = 2^((idx - BIAS) / 4)
#define PSY_EXP_INDEX_BIAS 43
#define PSY_EXP_INDEX_MIN  0
#define PSY_EXP_INDEX_MAX  63

// Noise fill parameters
#define PSY_NF_CREST_RATIO      100     // 10^(20/10) = 100 linear power ratio
#define PSY_NF_EXP_MAX          7       // Tier 1 NF: crest-based
#define PSY_NF_EXP_MAX_TIER2    24      // Tier 2 NF: envelope-driven
#define PSY_NF_SMR_THRESHOLD_Q4 Q4(0.5) // SMR threshold for tier 2 noise fill

/* ── Band analysis ── */

/**
 * @brief Spectral band analysis
 *
 * @param spec_q31    Spectral coeffs, taken from mdct
 * @param loss_bits   BFP exponent of the coeffs
 * @param exp_indices Output for 6-bit exponent index per band
 * @param band_energy Output of the band energies (sum of (|X|>>16)^2 per band)
 * @param band_peak   Output of the band peaks (max(|X|>>16) per band)
 */
void psy_band_analysis(const int32_t *spec_q31,
                       int loss_bits,
                       int32_t *exp_indices,
                       uint64_t *band_energy,
                       uint32_t *band_peak);

/**
 * @brief Test whether a band's crest factor is below the noise-fill threshold
 *
 * Returns 1 if peak^2 * w < PSY_NF_CREST_RATIO * sum_sq (non-tonal band
 * suitable for noise fill).
 *
 * @param band   Band index
 * @param sum_sq sum of squared scaled magnitudes for the band
 * @param peak   peak mag in the band
 * @return True if crest factor is below threshold, false otherwise
 */
static inline bool psy_nf_crest_below(int band, uint64_t sum_sq, uint32_t peak) {
  int w = psy_band_edges[band + 1] - psy_band_edges[band];
  uint64_t lhs = (uint64_t)peak * peak * (uint32_t)w;
  uint64_t rhs = PSY_NF_CREST_RATIO * sum_sq;
  return lhs < rhs;
}

/**
 * @brief Calculates the Cross-band masking via ERB-rate spreading
 *
 * Computes Signal-to-Mask Ratio per band in Q4 log2 units
 *
 * @param exp_indices Band exponent indices from psy_band_analysis()
 * @param smr_q4_out  Output SMR per band in Q4 (energy_q4 - mask_q4)
 */
void psy_spreading_envelope(const int32_t *exp_indices, int32_t *smr_q4_out);

#ifdef __cplusplus
}
#endif

#endif // HQLC_PSY_H
