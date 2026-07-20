#ifndef HQLC_TNS_H
#define HQLC_TNS_H

#include <stdbool.h>
#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Max order for the TNS filter
#define TNS_MAX_ORDER 4

#define TNS_K_BITS   4
#define TNS_LAR_HALF 7

// Start TNS analysis and filter at ~940 Hz
#define TNS_START_BIN 20

// Transient detector runs 8 sub-blocks of 64 samples
#define TNS_DETECT_SUBBLOCKS 8

// Ratio for transient detection (higher = less sensitive)
#define TNS_DETECT_RATIO 8

// Absolute HP-energy floor per sub-block (16-bit sample scale)
#define TNS_DETECT_FLOOR (1u << 20)

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

extern const int32_t tns_k_dq_q30[15];

/**
 * @brief Dequantize a LAR index to a reflection coefficient.
 *
 * @param q Quantized LAR index in the range -7..7.
 * @return Reflection coefficient in Q30 format.
 */
static inline int32_t tns_dequant_k(int q) {
  return tns_k_dq_q30[q + 7];
}

/**
 * @brief Quantize a reflection coefficient to a LAR index.
 *
 * @param k_q30 Reflection coefficient in Q30 format.
 * @return Quantized LAR index in the range -7..7.
 */
int tns_quant_k(int32_t k_q30);

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
 * @param out Destination for TNS analysis results.
 */
void tns_analyze(const int32_t *spec_q31, tns_info *out);

/**
 * @brief Apply the encoder TNS lattice FIR filter in place.
 *
 * @param spec_q31 Spectrum to filter, HQLC_FRAME_SAMPLES elements.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 * @param input_rshift Right shift applied to each input sample, or 0 for none.
 * @param out_hr If non-NULL, receives output block headroom.
 */
void tns_lattice_fir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr);

/**
 * @brief Apply the decoder TNS lattice IIR filter in place.
 *
 * @param spec_q31 Spectrum to filter, HQLC_FRAME_SAMPLES elements.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 * @param input_rshift Right shift applied to each input sample, or 0 for none.
 * @param out_hr If non-NULL, receives output block headroom.
 */
void tns_lattice_iir(
    int32_t *spec_q31, const int32_t *k_q30, int order, int input_rshift, int *out_hr);

/**
 * @brief Apply the FIR filter with automatic headroom management.
 *
 * Pre-shifts the spectrum, runs the lattice FIR filter, then renormalizes the
 * result.
 *
 * @param spec_q31 Spectrum to filter in place.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 * @return Net `loss_bits` adjustment for the caller to add.
 */
int tns_fir_safe(int32_t *spec_q31, const int32_t *k_q30, int order);

/**
 * @brief Apply the IIR filter with automatic headroom management.
 *
 * Pre-shifts the spectrum, runs the lattice IIR filter, then renormalizes the
 * result.
 *
 * @param spec_q31 Spectrum to filter in place.
 * @param k_q30 Reflection coefficients in Q30 format.
 * @param order Filter order.
 * @return Net `loss_bits` adjustment for the caller to add.
 */
int tns_iir_safe(int32_t *spec_q31, const int32_t *k_q30, int order);

#ifdef __cplusplus
}
#endif

#endif // HQLC_TNS_H
