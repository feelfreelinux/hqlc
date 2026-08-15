#ifndef HQLC_TNS_H
#define HQLC_TNS_H

#include <stdbool.h>
#include <stdint.h>

#include "fxp.h"
#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Max order for the TNS filter
#define TNS_MAX_ORDER 4

#define TNS_K_BITS   4
#define TNS_LAR_HALF 7

// Transient detector runs 8 sub-blocks of 64 samples
#define TNS_DETECT_SUBBLOCKS 8

// Per-channel detector state (zero-init = preceding silence)
typedef struct {
  uint64_t sub_energy[TNS_DETECT_SUBBLOCKS]; /**< last frame's HP sub-block energies */
  int32_t last_sample;                       /**< HP first-difference continuity */
} tns_detect_state;

// TNS analysis result for one channel
typedef struct {
  int order;                    /**< 0 = inactive */
  int8_t q_lar[TNS_MAX_ORDER];  /**< quantized LAR indices (-7 to 7) */
  int32_t k_q30[TNS_MAX_ORDER]; /**< dequantized reflection coeffs Q30 */
} tns_info;

/**
 * @brief Dequantize a LAR index to a reflection coefficient.
 *
 * @param q Quantized LAR index in the range -7..7.
 * @return Reflection coefficient in Q30 format.
 */
int32_t tns_dequant_k(int q);

/**
 * @brief Detect an attack transient in the current frame.
 *
 * A transient is detected when any 64-sample high-pass sub-block has enough
 * energy compared with the mean of the preceding sub-blocks.
 *
 * @param st Per-channel detector state, updated in place.
 * @param curr_pcm Current frame PCM samples.
 * @param fmt PCM sample format.
 * @param stride Channel interleave stride, usually the total channel count.
 * @param ch Channel index to analyze.
 * @return True if an attack transient is detected.
 */
bool tns_detect_transient(tns_detect_state *st,
                          const uint8_t *curr_pcm,
                          hqlc_pcm_format fmt,
                          int stride,
                          int ch);

/**
 * @brief Analyze a spectrum and produce TNS filter parameters.
 *
 * @param spec_q31 MDCT spectrum, HQLC_FRAME_SAMPLES elements. Not modified.
 * @param hangover Frame is TNS-eligible only through the hangover, not its own attack.
 * @param out Destination for TNS analysis results.
 */
void tns_analyze(const int32_t *spec_q31, bool hangover, tns_info *out);

/**
 * @brief Apply the TNS analysis filter with automatic headroom management.
 *
 * Ensures the required input headroom, runs the lattice FIR filter, then
 * renormalizes the result and updates its BFP exponent.
 *
 * @param spectrum Spectrum to filter in place, including its BFP exponent.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 */
void tns_apply_analysis_filter(bfp_i32 *spectrum, const int32_t *k_q30, int order);

/**
 * @brief Apply the TNS synthesis filter with automatic headroom management.
 *
 * Ensures the required input headroom, runs the lattice IIR filter, then
 * renormalizes the result and updates its BFP exponent.
 *
 * @param spectrum Spectrum to filter in place, including its BFP exponent.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 */
void tns_apply_synthesis_filter(bfp_i32 *spectrum, const int32_t *k_q30, int order);

#ifdef __cplusplus
}
#endif

#endif // HQLC_TNS_H
