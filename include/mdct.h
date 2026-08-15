#ifndef HQLC_MDCT_H
#define HQLC_MDCT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "fxp.h"
#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// MDCT transform lengths
#define MDCT_N              HQLC_FRAME_SAMPLES // 512
#define MDCT_BLOCK_LEN      HQLC_BLOCK_SAMPLES // 1024 (full block with overlap)
#define MDCT_FFT_N          (MDCT_N / 2)       // 256 (half-length complex FFT)

// Enough scratch space for MDCT + overlap-add
#define MDCT_SCRATCH_BYTES ((MDCT_N + 2 * MDCT_FFT_N) * (int)sizeof(int32_t))

// Overlap-add state for the decoder IMDCT
typedef struct {
  bool has_overlap;
  int exp2;                    /**< BFP exponent of stored overlap */
  int32_t overlap[MDCT_N / 2]; /**< first half of previous DCT-IV output */
} mdct_ola_state;

/**
 * @brief Initialize decoder overlap-add state.
 *
 * @param state Overlap-add state to reset. NULL is ignored.
 */
static inline void mdct_ola_init(mdct_ola_state *state) {
  if (!state) {
    return;
  }
  state->has_overlap = false;
  state->exp2 = 0;
  for (int i = 0; i < MDCT_N / 2; i++) {
    state->overlap[i] = 0;
  }
}

/**
 * @brief Convert PCM input to MDCT spectral coefficients.
 *
 * Uses the previous and current half frames to build the overlapped MDCT block.
 * Output coefficients are Q31 block-floating-point values.
 *
 * @param prev_pcm Previous frame PCM samples.
 * @param curr_pcm Current frame PCM samples.
 * @param half_pcm_len Byte length of each PCM half frame.
 * @param fmt PCM sample format.
 * @param stride Channel interleave stride, usually the total channel count.
 * @param channel_idx Channel index to read.
 * @param spectrum Destination for MDCT_N BFP spectral coefficients.
 * @param scratch Temporary workspace buffer.
 * @param scratch_len Capacity of `scratch` in bytes.
 * @return HQLC_OK on success, or an error code on failure.
 */
hqlc_error mdct_forward(const uint8_t *prev_pcm,
                        const uint8_t *curr_pcm,
                        size_t half_pcm_len,
                        hqlc_pcm_format fmt,
                        int stride,
                        int channel_idx,
                        bfp_i32 *spectrum,
                        void *scratch,
                        size_t scratch_len);

/**
 * @brief Run inverse MDCT, overlap-add, and PCM output in one pass.
 *
 * Performs DCT-IV, combines the windowed result with the previous frame's
 * overlap, and writes interleaved PCM directly.
 *
 * @param spectrum Input BFP spectral coefficients, MDCT_N elements.
 * @param ola Overlap-add state, updated in place.
 * @param pcm_out Interleaved PCM output buffer.
 * @param fmt PCM sample format.
 * @param stride Channel interleave stride, usually the total channel count.
 * @param channel_idx Channel index to write.
 * @param scratch Temporary workspace buffer.
 * @param scratch_len Capacity of `scratch` in bytes.
 * @return HQLC_OK on success, or an error code on failure.
 */
hqlc_error mdct_inverse_ola(const bfp_i32 *spectrum,
                            mdct_ola_state *ola,
                            uint8_t *pcm_out,
                            hqlc_pcm_format fmt,
                            int stride,
                            int channel_idx,
                            void *scratch,
                            size_t scratch_len);

#ifdef __cplusplus
}
#endif

#endif // HQLC_MDCT_H
