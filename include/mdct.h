#ifndef HQLC_MDCT_H
#define HQLC_MDCT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// MDCT transform lengths (derived from codec frame size)
#define MDCT_N              HQLC_FRAME_SAMPLES // 512: transform length
#define MDCT_BLOCK_LEN      HQLC_BLOCK_SAMPLES // 1024: full block with overlap
#define MDCT_FFT_N          (MDCT_N / 2)       // 256: half-length complex FFT
#define MDCT_DCT_BITS       10
#define MDCT_MATH_GAIN_BITS 8

// Enough scratch space for MDCT + overlap-add
#define MDCT_SCRATCH_BYTES ((MDCT_N + 2 * MDCT_FFT_N) * (int)sizeof(int32_t))

// Overlap-add state for the decoder IMDCT
typedef struct {
  bool has_overlap;
  int loss_td_bits;
  int32_t overlap_q31[MDCT_N];
} mdct_ola_state;

static inline void mdct_ola_init(mdct_ola_state *state) {
  if (!state) {
    return;
  }
  state->has_overlap = false;
  state->loss_td_bits = 0;
  for (int i = 0; i < MDCT_N; i++) {
    state->overlap_q31[i] = 0;
  }
}

/**
 * @brief Forward MDCT, converts interleaved PCM to spectral coefficients (Q31
 * BFP).
 *
 * @param prev_pcm      Previous frame PCM
 * @param curr_pcm      Current frame PCM
 * @param half_pcm_len  Byte length of each half (the prev and curr pcm halves)
 * @param fmt           PCM sample format
 * @param stride        Channel interleave stride, aka total channels
 * @param channel_idx   ch index for extraction
 * @param spec_q31      Output spectral coefficients - MDCT_N count.
 * @param spec_q31_len  Capacity of @p spec_q31 in elements.
 * @param scratch       Scratch buffer
 * @param scratch_len   Capacity of scratch in bytes.
 * @param loss_bits_out BFP exponent of the output spectrum.
 */
hqlc_error mdct_forward(const uint8_t *restrict prev_pcm,
                        const uint8_t *restrict curr_pcm,
                        size_t half_pcm_len,
                        hqlc_pcm_format fmt,
                        int stride,
                        int channel_idx,
                        int32_t *restrict spec_q31,
                        size_t spec_q31_len,
                        void *restrict scratch,
                        size_t scratch_len,
                        int *restrict loss_bits_out);

/**
 * @brief Inverse MDCT, spectral coefficients to windowed time samples (Q31
 * BFP).
 *
 * @param spec_q31         Input spectral coefficients, MDCT_N elements
 * @param spec_q31_len     spec_q31 count
 * @param loss_bits_in     BFP exponent of the input spectrum
 * @param windowed_q31     Output windowed samples MDCT_BLOCK_LEN elements.
 * @param windowed_q31_len Capacity of windowed_q31 in elements.
 * @param scratch          scratch buffer
 * @param scratch_len      scratch buffer bytes
 * @param loss_bits_out    BFP exponent of the output time samples
 */
hqlc_error mdct_inverse(const int32_t *restrict spec_q31,
                        size_t spec_q31_len,
                        int loss_bits_in,
                        int32_t *restrict windowed_q31,
                        size_t windowed_q31_len,
                        void *restrict scratch,
                        size_t scratch_len,
                        int *restrict loss_bits_out);

#ifdef __cplusplus
}
#endif

#endif // HQLC_MDCT_H
