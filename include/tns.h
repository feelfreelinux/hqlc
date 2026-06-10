#ifndef HQLC_TNS_H
#define HQLC_TNS_H

#include <stdbool.h>
#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// TNS_MAX_ORDER and TNS_START_BIN are overridable via -D for experiments
// (scripts/tns_sweep.py). Order is coded as 3 bits (order-1), so <= 8.
#ifndef TNS_MAX_ORDER
#define TNS_MAX_ORDER 4
#endif
#define TNS_K_BITS   4
#define TNS_LAR_HALF 7
#ifndef TNS_START_BIN
// ~940 Hz: TNS analyses and filters only above this bin. Lowered from 43
// (~2 kHz): including the 1-2 kHz region in the autocorrelation lets the
// filter capture the broadband transient structure — castanets pre-echo
// -11 dB at 96k with no measurable cost on tonal clips (see
// scripts/tns_sweep.py).
#define TNS_START_BIN 20
#endif

// Sub-block transient detector: the frame is split into 8 sub-blocks of 64
// samples; each sub-block's high-pass energy is compared against the mean of
// the preceding 8 sub-blocks (sliding across the frame boundary via state).
// TNS_DETECT_RATIO is the energy ratio that counts as an attack; overridable
// via -D for experiments (scripts/transient_sweep.py).
#define TNS_DETECT_SUBBLOCKS 8
#ifndef TNS_DETECT_RATIO
#define TNS_DETECT_RATIO 8
#endif
// Absolute HP-energy floor per sub-block (16-bit sample scale): don't call
// noise-floor wiggle in silence an attack.
#ifndef TNS_DETECT_FLOOR
#define TNS_DETECT_FLOOR (1u << 20)
#endif

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

// Dequantize LAR index to reflection coefficient in Q30
static inline int32_t tns_dequant_k(int q) {
  return tns_k_dq_q30[q + 7];
}

// Quantize reflection coefficient (Q30) to LAR index (-7, 7)
int tns_quant_k(int32_t k_q30);

/**
 * @brief Detect an attack transient inside the current frame.
 *
 * High-passes the frame (first difference), accumulates energy per 64-sample
 * sub-block, and fires when any sub-block exceeds TNS_DETECT_RATIO x the
 * mean of the preceding 8 sub-blocks. Catches mid-frame attacks and repeated
 * hits that a whole-frame energy ratio misses.
 *
 * @param st       Per-channel detector state (updated)
 * @param curr_pcm Current frame PCM
 * @param fmt      PCM sample format
 * @param stride   Channel interleave stride / channel count
 * @param ch       Channel index
 * @return true if an attack was detected in this frame
 */
bool tns_detect_transient(tns_detect_state *st,
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
