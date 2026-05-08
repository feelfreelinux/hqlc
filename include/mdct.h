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

// Overlap-add state for the decoder IMDCT.
// Stores N/2 raw DCT-IV values from the previous frame;
// the window multiply and OLA happen together in mdct_inverse_ola().
typedef struct {
  bool has_overlap;
  int loss_bits;                        // BFP exponent of stored overlap
  int32_t overlap[MDCT_N / 2];         // first half of previous DCT-IV output
} mdct_ola_state;

static inline void mdct_ola_init(mdct_ola_state *state) {
  if (!state) {
    return;
  }
  state->has_overlap = false;
  state->loss_bits = 0;
  for (int i = 0; i < MDCT_N / 2; i++) {
    state->overlap[i] = 0;
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
 * @brief Fused inverse MDCT + overlap-add + PCM write.
 *
 * Performs DCT-IV, then combines the windowed result with the previous frame's
 * overlap in a single pass, writing interleaved PCM directly.  Replaces the
 * old mdct_inverse() + manual OLA pattern.
 *
 * @param spec_q31     Input spectral coefficients, MDCT_N elements
 * @param spec_q31_len spec_q31 count
 * @param loss_bits_in BFP exponent of the input spectrum
 * @param ola          Overlap-add state (updated in-place)
 * @param pcm_out      Output PCM buffer (interleaved)
 * @param fmt          PCM sample format (HQLC_PCM16 or HQLC_PCM24)
 * @param stride       Channel interleave stride (total channels)
 * @param channel_idx  Channel index to write
 * @param scratch      Scratch buffer (MDCT_SCRATCH_BYTES)
 * @param scratch_len  Scratch buffer bytes
 */
hqlc_error mdct_inverse_ola(const int32_t *restrict spec_q31,
                            size_t spec_q31_len,
                            int loss_bits_in,
                            mdct_ola_state *restrict ola,
                            uint8_t *restrict pcm_out,
                            hqlc_pcm_format fmt,
                            int stride,
                            int channel_idx,
                            void *restrict scratch,
                            size_t scratch_len);

#ifdef __cplusplus
}
#endif

#endif // HQLC_MDCT_H
