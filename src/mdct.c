#include "mdct.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "fxp.h"
#include "hqlc_bench.h"
#include "mdct_tables.h"
#include "pcm.h"

#define MDCT_DCT_BITS       10
#define MDCT_MATH_GAIN_BITS 8

// Headroom above PCM full scale in the overlap-add fixed-point domain
#define MDCT_OLA_HEADROOM_BITS 4

/**
 * @brief Return one sample from the mirrored 1024-point KBD window
 */
static inline int32_t mdct_win_q31(int i) {
  return kbd_window_half_q31[(i < MDCT_N) ? i : (MDCT_BLOCK_LEN - 1 - i)];
}

/**
 * @brief Q31 multiply with one extra guard bit
 *
 * Uses >> 32 to give one extra bit of headroom, needed in the add/subtract steps that
 * follow it. It's tracked later in the BFP exponent, so the lost scale is accounted for.
 */
static inline int32_t mul_q31_guard(int32_t a, int32_t b) {
  return fxp_mul_rshift_i32(a, b, 32);
}

/**
 * @brief Branchless mag for OR-based headroom measurement.
 *
 * This is not a saturating abs for full int32, but it's good enough for the OR
 * accumulator to find highest occupied magnitude bit
 */
static inline uint32_t mag_or_i32(int32_t v) {
  return (uint32_t)(v ^ (v >> 31));
}

/**
 * @brief Scale a DCT-IV output sample into the overlap-add fixed-point domain
 *
 * @param shift_left Shift left when true, right otherwise
 * @param shift Shift amount, 0-31 for left and 1-31 for right
 */
static inline int32_t ola_to_fixed(int32_t v, bool shift_left, int shift) {
  if (shift_left) {
    int32_t shifted = (int32_t)((uint32_t)v << shift);
    // Saturate if the shift dropped any bits
    if ((shifted >> shift) != v) {
      return (v < 0) ? INT32_MIN : INT32_MAX;
    }
    return shifted;
  }
  return (v >> shift) + ((v >> (shift - 1)) & 1);
}

/**
 * @brief Round, clip and store one overlap-added sample as PCM
 *
 * @param base Interleaved PCM output buffer
 * @param fmt PCM sample format
 * @param idx Interleaved sample index
 * @param v Sample in the overlap-add fixed-point domain
 */
static inline void ola_store(uint8_t *base, hqlc_pcm_format fmt, int idx, int32_t v) {
  if (fmt == HQLC_PCM16) {
    const int rshift = 30 - MDCT_OLA_HEADROOM_BITS - 15;
    int32_t pcm16 = (v >> rshift) + ((v >> (rshift - 1)) & 1);
    ((int16_t *)base)[idx] = (int16_t)fxp_clamp_i32(pcm16, INT16_MIN, INT16_MAX);
  } else {
    const int rshift = 30 - MDCT_OLA_HEADROOM_BITS - 23;
    int32_t pcm24 = (v >> rshift) + ((v >> (rshift - 1)) & 1);
    pcm24 = fxp_clamp_i32(pcm24, -(1 << 23), (1 << 23) - 1);
    uint8_t *p = base + 3 * idx;
    p[0] = (uint8_t)(pcm24 & 0xFF);
    p[1] = (uint8_t)((pcm24 >> 8) & 0xFF);
    p[2] = (uint8_t)((pcm24 >> 16) & 0xFF);
  }
}

/**
 * @brief Overlap-add loop of ola_block_to_pcm, for one format and shift direction
 *
 * Always inlined with constant fmt and shift_left, so neither is tested per sample
 */
static inline __attribute__((always_inline)) void ola_block_loop(const int32_t *time,
                                                                 int32_t *overlap,
                                                                 bool shift_left,
                                                                 int shift,
                                                                 uint8_t *pcm_out,
                                                                 hqlc_pcm_format fmt,
                                                                 int stride,
                                                                 int channel_idx) {
  const int N = MDCT_N;
  const int half_n = N / 2;
  const int32_t *win = kbd_window_half_q31;

  for (int j = 0; j < half_n; j++) {
    int32_t p = overlap[j];
    int32_t t = ola_to_fixed(time[N - 1 - j], shift_left, shift);
    overlap[j] = ola_to_fixed(time[j], shift_left, shift);

    int32_t c = win[half_n - 1 - j];
    int32_t s = win[half_n + j];

    int32_t first = mul_q31_guard(t, c) - mul_q31_guard(p, s);
    int32_t second = -(mul_q31_guard(p, c) + mul_q31_guard(t, s));

    ola_store(pcm_out, fmt, (half_n - 1 - j) * stride + channel_idx, first);
    ola_store(pcm_out, fmt, (half_n + j) * stride + channel_idx, second);
  }
}

/**
 * @brief Overlap-add one IMDCT block with the stored overlap into PCM
 *
 * @param time DCT-IV output of the current block
 * @param overlap Stored first half of the previous block, replaced with this one's
 * @param exp2 Block exponent relative to the overlap-add fixed-point domain, -31 to 31
 * @param pcm_out Interleaved PCM output buffer
 * @param fmt PCM sample format
 * @param stride Number of interleaved channels in pcm_out
 * @param channel_idx Channel to write
 */
static void ola_block_to_pcm(const int32_t *time,
                             int32_t *overlap,
                             int exp2,
                             uint8_t *pcm_out,
                             hqlc_pcm_format fmt,
                             int stride,
                             int channel_idx) {
  // A separate copy of the loop per format and shift direction
  if (fmt == HQLC_PCM16 && exp2 >= 0) {
    ola_block_loop(time, overlap, true, exp2, pcm_out, HQLC_PCM16, stride, channel_idx);
  } else if (fmt == HQLC_PCM16) {
    ola_block_loop(time, overlap, false, -exp2, pcm_out, HQLC_PCM16, stride, channel_idx);
  } else if (exp2 >= 0) {
    ola_block_loop(time, overlap, true, exp2, pcm_out, HQLC_PCM24, stride, channel_idx);
  } else {
    ola_block_loop(time, overlap, false, -exp2, pcm_out, HQLC_PCM24, stride, channel_idx);
  }
}

/**
 * @brief Lookup the FFT twiddle W_256^k
 *
 * @param k Index into the twiddle lookup table
 * @param re Pointer to store the real part of the twiddle
 * @param im Pointer to store the imaginary part of the twiddle
 *
 * Done so we only store the first half, second half is the negated first half
 */
static inline void tw_lookup(int k, int32_t *re, int32_t *im) {
  k &= 255;
  if (k < 128) {
    *re = lut_fft_twiddle_q31[2 * k];
    *im = lut_fft_twiddle_q31[2 * k + 1];
  } else {
    *re = -lut_fft_twiddle_q31[2 * (k - 128)];
    *im = -lut_fft_twiddle_q31[2 * (k - 128) + 1];
  }
}

/**
 * @brief In-place fixed-size radix-4 FFT used by the DCT-IV.
 *
 * The FFT length is 256 = 4^4. Each radix-4 stage scales by >>2, giving >> 8 total
 * headroom protection. Complex values are interleaved as real/imaginary pairs.
 */
static void fft_scaled(int32_t *buf) {
  const int fft_len = MDCT_FFT_N;

  for (int swap = 0; swap < MDCT_DIGIT_REV_PAIRS; swap++) {
    int i = lut_digit_rev[2 * swap];
    int j = lut_digit_rev[2 * swap + 1];
    int32_t tmp_re = buf[2 * i];
    int32_t tmp_im = buf[2 * i + 1];
    buf[2 * i] = buf[2 * j];
    buf[2 * i + 1] = buf[2 * j + 1];
    buf[2 * j] = tmp_re;
    buf[2 * j + 1] = tmp_im;
  }

  for (int base = 0; base < fft_len; base += 4) {
    int32_t x0_re = buf[2 * base], x0_im = buf[2 * base + 1];
    int32_t x1_re = buf[2 * (base + 1)], x1_im = buf[2 * (base + 1) + 1];
    int32_t x2_re = buf[2 * (base + 2)], x2_im = buf[2 * (base + 2) + 1];
    int32_t x3_re = buf[2 * (base + 3)], x3_im = buf[2 * (base + 3) + 1];

    int32_t even_sum_re = (x0_re >> 1) + (x2_re >> 1);
    int32_t even_sum_im = (x0_im >> 1) + (x2_im >> 1);
    int32_t even_diff_re = (x0_re >> 1) - (x2_re >> 1);
    int32_t even_diff_im = (x0_im >> 1) - (x2_im >> 1);
    int32_t odd_sum_re = (x1_re >> 1) + (x3_re >> 1);
    int32_t odd_sum_im = (x1_im >> 1) + (x3_im >> 1);
    int32_t odd_diff_re = (x1_re >> 1) - (x3_re >> 1);
    int32_t odd_diff_im = (x1_im >> 1) - (x3_im >> 1);

    buf[2 * base] = (even_sum_re >> 1) + (odd_sum_re >> 1);
    buf[2 * base + 1] = (even_sum_im >> 1) + (odd_sum_im >> 1);
    buf[2 * (base + 1)] = (even_diff_re >> 1) + (odd_diff_im >> 1);
    buf[2 * (base + 1) + 1] = (even_diff_im >> 1) - (odd_diff_re >> 1);
    buf[2 * (base + 2)] = (even_sum_re >> 1) - (odd_sum_re >> 1);
    buf[2 * (base + 2) + 1] = (even_sum_im >> 1) - (odd_sum_im >> 1);
    buf[2 * (base + 3)] = (even_diff_re >> 1) - (odd_diff_im >> 1);
    buf[2 * (base + 3) + 1] = (even_diff_im >> 1) + (odd_diff_re >> 1);
  }

  for (int stage = 1; stage <= 3; stage++) {
    int quarter_stride = 1 << (2 * stage);
    int butterfly_span = quarter_stride << 2;
    int groups = fft_len / butterfly_span;

    for (int offset = 0; offset < quarter_stride; offset++) {
      int32_t w1_re, w1_im, w2_re, w2_im, w3_re, w3_im;
      tw_lookup(offset * groups, &w1_re, &w1_im);
      tw_lookup(2 * offset * groups, &w2_re, &w2_im);
      tw_lookup(3 * offset * groups, &w3_re, &w3_im);

      for (int group = 0; group < groups; group++) {
        int i0 = group * butterfly_span + offset;
        int i1 = i0 + quarter_stride;
        int i2 = i0 + 2 * quarter_stride;
        int i3 = i0 + 3 * quarter_stride;

        int32_t x0_re = buf[2 * i0], x0_im = buf[2 * i0 + 1];
        int32_t raw1_re = buf[2 * i1], raw1_im = buf[2 * i1 + 1];
        int32_t raw2_re = buf[2 * i2], raw2_im = buf[2 * i2 + 1];
        int32_t raw3_re = buf[2 * i3], raw3_im = buf[2 * i3 + 1];

        // >> 32 leaves the twiddled inputs already halved for the butterfly
        int32_t h1_re = mul_q31_guard(raw1_re, w1_re) - mul_q31_guard(raw1_im, w1_im);
        int32_t h1_im = mul_q31_guard(raw1_re, w1_im) + mul_q31_guard(raw1_im, w1_re);
        int32_t h2_re = mul_q31_guard(raw2_re, w2_re) - mul_q31_guard(raw2_im, w2_im);
        int32_t h2_im = mul_q31_guard(raw2_re, w2_im) + mul_q31_guard(raw2_im, w2_re);
        int32_t h3_re = mul_q31_guard(raw3_re, w3_re) - mul_q31_guard(raw3_im, w3_im);
        int32_t h3_im = mul_q31_guard(raw3_re, w3_im) + mul_q31_guard(raw3_im, w3_re);

        int32_t even_sum_re = (x0_re >> 1) + h2_re;
        int32_t even_sum_im = (x0_im >> 1) + h2_im;
        int32_t even_diff_re = (x0_re >> 1) - h2_re;
        int32_t even_diff_im = (x0_im >> 1) - h2_im;
        int32_t odd_sum_re = h1_re + h3_re;
        int32_t odd_sum_im = h1_im + h3_im;
        int32_t odd_diff_re = h1_re - h3_re;
        int32_t odd_diff_im = h1_im - h3_im;

        buf[2 * i0] = (even_sum_re >> 1) + (odd_sum_re >> 1);
        buf[2 * i0 + 1] = (even_sum_im >> 1) + (odd_sum_im >> 1);
        buf[2 * i1] = (even_diff_re >> 1) + (odd_diff_im >> 1);
        buf[2 * i1 + 1] = (even_diff_im >> 1) - (odd_diff_re >> 1);
        buf[2 * i2] = (even_sum_re >> 1) - (odd_sum_re >> 1);
        buf[2 * i2 + 1] = (even_sum_im >> 1) - (odd_sum_im >> 1);
        buf[2 * i3] = (even_diff_re >> 1) - (odd_diff_im >> 1);
        buf[2 * i3 + 1] = (even_diff_im >> 1) + (odd_diff_re >> 1);
      }
    }
  }
}

/**
 * @brief DCT-IV implemented with a half-length complex FFT
 *
 * @param in Input buffer, may be the same as out
 * @param out Output buffer
 * @param work Scratch buffer for FFT
 */
static void dct_iv(const int32_t *in, int32_t *out, int32_t *work) {
  const int N = MDCT_N;
  const int N_FFT = MDCT_FFT_N;

  HQLC_BENCH_BEGIN(HQLC_BENCH_MDCT_PRE_TW);
  for (int k = 0; k < N_FFT; k++) {
    int32_t even_sample = in[2 * k];
    int32_t odd_mirror = in[N - 1 - 2 * k];
    int32_t tw_re = lut_pre_twiddle_q31[2 * k];
    int32_t tw_im = lut_pre_twiddle_q31[2 * k + 1];

    work[2 * k] = mul_q31_guard(even_sample, tw_re) - mul_q31_guard(odd_mirror, tw_im);
    work[2 * k + 1] =
        mul_q31_guard(even_sample, tw_im) + mul_q31_guard(odd_mirror, tw_re);
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_PRE_TW);

  HQLC_BENCH_BEGIN(HQLC_BENCH_MDCT_FFT);
  fft_scaled(work);
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FFT);

  HQLC_BENCH_BEGIN(HQLC_BENCH_MDCT_POST_TW);
  for (int k = 0; k < N_FFT; k++) {
    int32_t fft_re = work[2 * k];
    int32_t fft_im = work[2 * k + 1];
    int32_t tw_re = lut_post_twiddle_q31[2 * k];
    int32_t tw_im = lut_post_twiddle_q31[2 * k + 1];

    out[2 * k] = mul_q31_guard(fft_re, tw_re) - mul_q31_guard(fft_im, tw_im);
    out[N - 1 - 2 * k] = -(mul_q31_guard(fft_re, tw_im) + mul_q31_guard(fft_im, tw_re));
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_POST_TW);
}

hqlc_error mdct_forward(const uint8_t *prev_pcm,
                        const uint8_t *curr_pcm,
                        size_t half_pcm_len,
                        hqlc_pcm_format fmt,
                        int stride,
                        int channel_idx,
                        bfp_i32 *spectrum,
                        void *scratch,
                        size_t scratch_len) {
  if (!prev_pcm || !curr_pcm || !spectrum || !spectrum->data || !scratch) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (spectrum->length < (size_t)MDCT_N) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (scratch_len < (size_t)MDCT_SCRATCH_BYTES) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (fmt != HQLC_PCM16 && fmt != HQLC_PCM24) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (stride < 1 || channel_idx < 0 || channel_idx >= stride) {
    return HQLC_ERR_INVALID_ARG;
  }

  size_t bytes_per_sample = (fmt == HQLC_PCM16) ? 2 : 3;
  size_t required_pcm_bytes =
      (size_t)HQLC_FRAME_SAMPLES * (size_t)stride * bytes_per_sample;
  if (half_pcm_len < required_pcm_bytes) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  const int N = MDCT_N;
  const int half_n = N / 2;
  int32_t *folded = spectrum->data;

  HQLC_BENCH_BEGIN(HQLC_BENCH_MDCT_FOLD);
  uint32_t fold_mag = 0;

  // The 1024-sample MDCT window is [A B C D], four 256-sample quarters
  // Folding turns it into the 512 sample DCT-IV input:
  //  folded[0..N/2] = -D - reverse(C)
  //  folded[N/2..N] =  A - reverse(B)
  // (windowing is applied before folding in the same loop)
  if (fmt == HQLC_PCM16) {
    const int16_t *curr = (const int16_t *)curr_pcm;
    const int16_t *prev = (const int16_t *)prev_pcm;
    const int32_t *win = kbd_window_half_q31;

    for (int n = 0; n < half_n; n++) {
      int32_t curr_late =
          (int32_t)((uint32_t)curr[(half_n + n) * stride + channel_idx] << 16);
      int32_t curr_early_rev =
          (int32_t)((uint32_t)curr[(half_n - 1 - n) * stride + channel_idx] << 16);

      int32_t late_windowed = mul_q31_guard(curr_late, win[half_n - 1 - n]);
      int32_t early_windowed = mul_q31_guard(curr_early_rev, win[half_n + n]);
      int32_t folded_sample = -late_windowed - early_windowed;

      folded[n] = folded_sample;
      fold_mag |= mag_or_i32(folded_sample);
    }

    for (int n = 0; n < half_n; n++) {
      int32_t prev_early = (int32_t)((uint32_t)prev[n * stride + channel_idx] << 16);
      int32_t prev_late_rev =
          (int32_t)((uint32_t)prev[(N - 1 - n) * stride + channel_idx] << 16);

      int32_t early_windowed = mul_q31_guard(prev_early, win[n]);
      int32_t late_windowed = mul_q31_guard(prev_late_rev, win[N - 1 - n]);
      int32_t folded_sample = early_windowed - late_windowed;

      folded[half_n + n] = folded_sample;
      fold_mag |= mag_or_i32(folded_sample);
    }
  } else {
    for (int n = 0; n < half_n; n++) {
      int32_t curr_late =
          pcm_load_q31(curr_pcm, fmt, (half_n + n) * stride + channel_idx);
      int32_t curr_early_rev =
          pcm_load_q31(curr_pcm, fmt, (half_n - 1 - n) * stride + channel_idx);

      int32_t late_windowed = mul_q31_guard(curr_late, mdct_win_q31(3 * half_n + n));
      int32_t early_windowed =
          mul_q31_guard(curr_early_rev, mdct_win_q31(3 * half_n - 1 - n));
      int32_t folded_sample = -late_windowed - early_windowed;

      folded[n] = folded_sample;
      fold_mag |= mag_or_i32(folded_sample);
    }

    for (int n = 0; n < half_n; n++) {
      int32_t prev_early = pcm_load_q31(prev_pcm, fmt, n * stride + channel_idx);
      int32_t prev_late_rev =
          pcm_load_q31(prev_pcm, fmt, (N - 1 - n) * stride + channel_idx);

      int32_t early_windowed = mul_q31_guard(prev_early, mdct_win_q31(n));
      int32_t late_windowed = mul_q31_guard(prev_late_rev, mdct_win_q31(N - 1 - n));
      int32_t folded_sample = early_windowed - late_windowed;

      folded[half_n + n] = folded_sample;
      fold_mag |= mag_or_i32(folded_sample);
    }
  }

  // Guard-bit multiplication leaves the folded mantissas in Q30, represented
  // as Q31 BFP with exp2 = 1. Reclaim the headroom measured during folding.
  bfp_i32 folded_block = bfp_i32_view(folded, N, 1);
  int fold_headroom = fxp_signed_headroom_u32(fold_mag);
  bfp_i32_renormalize(&folded_block, (uint8_t)fold_headroom);
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FOLD);

  int32_t *fft_work = (int32_t *)scratch;
  dct_iv(folded, folded, fft_work);

  spectrum->exp2 = folded_block.exp2 + MDCT_DCT_BITS;
  return HQLC_OK;
}

hqlc_error mdct_inverse_ola(const bfp_i32 *spectrum,
                            mdct_ola_state *ola,
                            uint8_t *pcm_out,
                            hqlc_pcm_format fmt,
                            int stride,
                            int channel_idx,
                            void *scratch,
                            size_t scratch_len) {
  if (!spectrum || !spectrum->data || !ola || !pcm_out || !scratch) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (spectrum->length < (size_t)MDCT_N) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (scratch_len < (size_t)MDCT_SCRATCH_BYTES) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (fmt != HQLC_PCM16 && fmt != HQLC_PCM24) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (stride < 1 || channel_idx < 0 || channel_idx >= stride) {
    return HQLC_ERR_INVALID_ARG;
  }

  const int N = MDCT_N;

  int32_t *time = (int32_t *)scratch;
  int32_t *fft_work = &time[N];

  dct_iv(spectrum->data, time, fft_work);

  // Block exponent relative to the overlap-add fixed-point domain
  int exp2 =
      spectrum->exp2 + MDCT_DCT_BITS - MDCT_MATH_GAIN_BITS - MDCT_OLA_HEADROOM_BITS;

  // Clamp the exp
  if (exp2 > 31) {
    exp2 = 31;
  } else if (exp2 < -31) {
    exp2 = -31;
  }

  ola_block_to_pcm(time, ola->overlap, exp2, pcm_out, fmt, stride, channel_idx);

  return HQLC_OK;
}
