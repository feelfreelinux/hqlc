#ifndef HQLC_TNS_H
#define HQLC_TNS_H

#include <stdbool.h>
#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TNS_MAX_ORDER 4
#define TNS_K_BITS    4
#define TNS_LAR_HALF  7
#define TNS_START_BIN 43 // ~2 kHz: TNS analyses and filters only above this bin

// TNS analysis result for one channel
typedef struct {
  int order;                    /**< 0 = inactive */
  int8_t q_lar[TNS_MAX_ORDER];  /**< quantized LAR indices (-7 to 7) */
  int32_t k_q30[TNS_MAX_ORDER]; /**< dequantized reflection coeffs Q30 */
} tns_info;

extern const int32_t tns_k_dq_q30[15];

// Dequantize LAR index to reflection coefficient in Q30
static inline int32_t tns_dequant_k(int q) {
  return tns_k_dq_q30[q + 7];
}

// Quantize reflection coefficient (Q30) to LAR index (-7, 7)
int tns_quant_k(int32_t k_q30);

/**
 * @brief Detect transient by comparing frame energies.
 *
 * @param prev_pcm Previous frame PCM
 * @param curr_pcm Current frame PCM
 * @param fmt      PCM sample format
 * @param stride   Channel interleave stride / channel count
 * @param ch       Channel index
 * @return 1 if current frame is >= 2x louder (attack transient)
 */
bool tns_detect_transient(const uint8_t *prev_pcm,
                          const uint8_t *curr_pcm,
                          hqlc_pcm_format fmt,
                          int stride,
                          int ch);

/**
 * @brief Analyse spectrum and produce TNS filter parameters
 *
 * @param spec_q31 MDCT spectrum (HQLC_FRAME_SAMPLES, not modified)
 * @param out Output TNS results
 */
void tns_analyze(const int32_t *spec_q31, tns_info *out);

/**
 * @brief Lattice FIR (encoder analysis filter, in-place)
 *
 * @param spec_q31     Spectrum to filter (HQLC_FRAME_SAMPLES, modified in-place)
 * @param k_q30        Reflection coefficients in Q30
 * @param order        Filter order
 * @param input_rshift Right-shift applied to each input sample (0 = none)
 * @param out_hr       If non-NULL, receives output block headroom
 */
void tns_lattice_fir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr);

/**
 * @brief Lattice IIR (decoder synthesis filter, in-place)
 *
 * @param spec_q31     Spectrum to filter (HQLC_FRAME_SAMPLES, modified in-place)
 * @param k_q30        Reflection coefficients in Q30
 * @param order        Filter order
 * @param input_rshift Right-shift applied to each input sample (0 = none)
 * @param out_hr       If non-NULL, receives output block headroom
 */
void tns_lattice_iir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr);

/**
 * @brief Safe FIR with automatic headroom management.
 *
 * Pre-shifts spectrum, runs lattice FIR, re-normalizes.
 * @return Net loss_bits adjustment (caller adds to loss_bits)
 */
int tns_fir_safe(int32_t *spec_q31, const int32_t *k_q30, int order);

/**
 * @brief Safe IIR with automatic headroom management.
 *
 * Pre-shifts spectrum, runs lattice IIR, re-normalizes
 * @return Net loss_bits adjustment (caller adds to loss_bits)
 */
int tns_iir_safe(int32_t *spec_q31, const int32_t *k_q30, int order);

#ifdef __cplusplus
}
#endif

#endif // HQLC_TNS_H
