#include "mdct.h"

#include <string.h>

#include "fxp.h"
#include "hqlc_bench.h"
#include "mdct_tables.h"
#include "pcm.h"

/**
 * @brief Return one sample from the mirrored 1024-point KBD window
 */
static inline int32_t mdct_win_q31(int i) {
  return kbd_window_half_q31[(i < MDCT_N) ? i : (MDCT_BLOCK_LEN - 1 - i)];
}

/**
 * @brief Q31 multiply with one extra guard bit
 *
 * Uses >> 32 to give one extra bit of headroom, needed in the add/substract steps that
 * follow it. Its tracked later in the bfp, so the lost scale is accounted for.
 */
static inline int32_t mul_q31_guard(int32_t a, int32_t b) {
  return (int32_t)((int64_t)a * b >> 32);
}

/**
 * @brief Branchless mag for OR-based headroom measurement.
 *
 * This is not a saturating abs for full int32, but its good enough for the OR accumulator
 * to find highest occupied magnitude bit
 */
static inline uint32_t mag_or_i32(int32_t v) {
  return (uint32_t)(v ^ (v >> 31));
}

/**
 * @brief Align one BFP sample into a common exponent domain.
 */
static inline int32_t align_bfp_sample(int32_t sample, int shift) {
  return (shift < 31) ? fxp_shr_rnd_i32(sample, shift) : 0;
}

/**
 * @brief Apply the final PCM exponent after overlap-add.
 */
static inline int32_t apply_pcm_exp(int32_t sample, int exp_shift) {
  if (exp_shift > 0) {
    return fxp_shl_sat_i32(sample, exp_shift);
  }
  if (exp_shift < 0) {
    return fxp_shr_rnd_i32(sample, -exp_shift);
  }
  return sample;
}

/**
 * @brief Convert a Q31 PCM sample to clamped signed 16-bit PCM.
 */
static inline int16_t pcm_q31_to_i16(int32_t sample_q31) {
  int32_t pcm16 = (sample_q31 >> 16) + ((sample_q31 >> 15) & 1);
  return pcm_clamp_i16(pcm16);
}

/**
 * @brief Window, align, and overlap-add one previous/current IMDCT pair.
 *
 * The signs are part of the MDCT unfold formula, callers pass +1 or -1 values
 * for the previous and current terms.
 */
static inline int32_t ola_mix_q31(int32_t prev_sample,
                                  int32_t prev_window,
                                  int prev_sign,
                                  int prev_shift,
                                  int32_t curr_sample,
                                  int32_t curr_window,
                                  int curr_sign,
                                  int curr_shift,
                                  int pcm_exp) {
  int32_t prev_windowed = mul_q31_guard(prev_sample, prev_window);
  int32_t curr_windowed = mul_q31_guard(curr_sample, curr_window);

  if (prev_sign < 0) {
    prev_windowed = -prev_windowed;
  }
  if (curr_sign < 0) {
    curr_windowed = -curr_windowed;
  }

  int32_t prev_aligned = align_bfp_sample(prev_windowed, prev_shift);
  int32_t curr_aligned = align_bfp_sample(curr_windowed, curr_shift);
  int32_t mixed = fxp_sat_i64_to_i32((int64_t)prev_aligned + curr_aligned);
  return apply_pcm_exp(mixed, pcm_exp);
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

        int32_t x1_re = fxp_mul_q31(raw1_re, w1_re) - fxp_mul_q31(raw1_im, w1_im);
        int32_t x1_im = fxp_mul_q31(raw1_re, w1_im) + fxp_mul_q31(raw1_im, w1_re);
        int32_t x2_re = fxp_mul_q31(raw2_re, w2_re) - fxp_mul_q31(raw2_im, w2_im);
        int32_t x2_im = fxp_mul_q31(raw2_re, w2_im) + fxp_mul_q31(raw2_im, w2_re);
        int32_t x3_re = fxp_mul_q31(raw3_re, w3_re) - fxp_mul_q31(raw3_im, w3_im);
        int32_t x3_im = fxp_mul_q31(raw3_re, w3_im) + fxp_mul_q31(raw3_im, w3_re);

        int32_t even_sum_re = (x0_re >> 1) + (x2_re >> 1);
        int32_t even_sum_im = (x0_im >> 1) + (x2_im >> 1);
        int32_t even_diff_re = (x0_re >> 1) - (x2_re >> 1);
        int32_t even_diff_im = (x0_im >> 1) - (x2_im >> 1);
        int32_t odd_sum_re = (x1_re >> 1) + (x3_re >> 1);
        int32_t odd_sum_im = (x1_im >> 1) + (x3_im >> 1);
        int32_t odd_diff_re = (x1_re >> 1) - (x3_re >> 1);
        int32_t odd_diff_im = (x1_im >> 1) - (x3_im >> 1);

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
 * @brief In-place DCT-IV implemented with a half-length complex FFT
 *
 * @param data Input/output buffer for DCT-IV
 * @param work Scratch buffer for FFT
 */
static void dct_iv(int32_t *data, int32_t *work) {
  const int N = MDCT_N;
  const int N_FFT = MDCT_FFT_N;

  HQLC_BENCH_BEGIN();
  for (int k = 0; k < N_FFT; k++) {
    int32_t even_sample = data[2 * k];
    int32_t odd_mirror = data[N - 1 - 2 * k];
    int32_t tw_re = lut_pre_twiddle_q31[2 * k];
    int32_t tw_im = lut_pre_twiddle_q31[2 * k + 1];

    work[2 * k] = mul_q31_guard(even_sample, tw_re) - mul_q31_guard(odd_mirror, tw_im);
    work[2 * k + 1] =
        mul_q31_guard(even_sample, tw_im) + mul_q31_guard(odd_mirror, tw_re);
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_PRE_TW);

  HQLC_BENCH_BEGIN();
  fft_scaled(work);
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FFT);

  HQLC_BENCH_BEGIN();
  for (int k = 0; k < N_FFT; k++) {
    int32_t fft_re = work[2 * k];
    int32_t fft_im = work[2 * k + 1];
    int32_t tw_re = lut_post_twiddle_q31[2 * k];
    int32_t tw_im = lut_post_twiddle_q31[2 * k + 1];

    data[2 * k] = mul_q31_guard(fft_re, tw_re) - mul_q31_guard(fft_im, tw_im);
    data[N - 1 - 2 * k] = -(mul_q31_guard(fft_re, tw_im) + mul_q31_guard(fft_im, tw_re));
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_POST_TW);
}

hqlc_error mdct_forward(const uint8_t *prev_pcm,
                        const uint8_t *curr_pcm,
                        size_t half_pcm_len,
                        hqlc_pcm_format fmt,
                        int stride,
                        int channel_idx,
                        int32_t *spec_q31,
                        size_t spec_q31_len,
                        void *scratch,
                        size_t scratch_len,
                        int *loss_bits_out) {
  if (!prev_pcm || !curr_pcm || !spec_q31 || !scratch || !loss_bits_out) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (spec_q31_len < (size_t)MDCT_N) {
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
  int32_t *folded = spec_q31;

  HQLC_BENCH_BEGIN();
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

  int headroom = fxp_headroom_u32(fold_mag);
  int fold_gain;
  if (fold_mag == 0) {
    fold_gain = 30;
  } else {
    if (headroom > 0) {
      for (int i = 0; i < N; i++) {
        folded[i] = (int32_t)((uint32_t)folded[i] << headroom);
      }
    }
    fold_gain = headroom - 1;
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FOLD);

  int32_t *fft_work = (int32_t *)scratch;
  dct_iv(folded, fft_work);

  *loss_bits_out = -fold_gain + MDCT_DCT_BITS;
  return HQLC_OK;
}

hqlc_error mdct_inverse_ola(const int32_t *spec_q31,
                            size_t spec_q31_len,
                            int loss_bits_in,
                            mdct_ola_state *ola,
                            uint8_t *pcm_out,
                            hqlc_pcm_format fmt,
                            int stride,
                            int channel_idx,
                            void *scratch,
                            size_t scratch_len) {
  if (!spec_q31 || !ola || !pcm_out || !scratch) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (spec_q31_len < (size_t)MDCT_N) {
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
  const int half_n = N / 2;

  int32_t *time = (int32_t *)scratch;
  int32_t *fft_work = &time[N];
  memcpy(time, spec_q31, (size_t)N * sizeof(int32_t));

  dct_iv(time, fft_work);

  int curr_exp = loss_bits_in + MDCT_DCT_BITS - MDCT_MATH_GAIN_BITS;
  if (!ola->has_overlap) {
    ola->loss_bits = curr_exp;
    ola->has_overlap = true;
  }

  int prev_exp = ola->loss_bits;
  int common_exp = (prev_exp > curr_exp) ? prev_exp : curr_exp;
  int prev_shift = (common_exp - prev_exp) + 1;
  int curr_shift = (common_exp - curr_exp) + 1;
  int pcm_exp = common_exp + 2;

  const int32_t *win = kbd_window_half_q31;
  const int32_t *prev_time = ola->overlap;

  if (fmt == HQLC_PCM16) {
    int16_t *out = (int16_t *)pcm_out;
    for (int n = 0; n < half_n; n++) {
      int32_t first_pcm = ola_mix_q31(prev_time[half_n - 1 - n],
                                      win[N - 1 - n],
                                      -1,
                                      prev_shift,
                                      time[half_n + n],
                                      win[n],
                                      1,
                                      curr_shift,
                                      pcm_exp);
      out[n * stride + channel_idx] = pcm_q31_to_i16(first_pcm);

      int32_t second_pcm = ola_mix_q31(prev_time[n],
                                       win[half_n - 1 - n],
                                       -1,
                                       prev_shift,
                                       time[N - 1 - n],
                                       win[half_n + n],
                                       -1,
                                       curr_shift,
                                       pcm_exp);
      out[(half_n + n) * stride + channel_idx] = pcm_q31_to_i16(second_pcm);
    }
  } else {
    for (int n = 0; n < half_n; n++) {
      int32_t first_pcm = ola_mix_q31(prev_time[half_n - 1 - n],
                                      win[N - 1 - n],
                                      -1,
                                      prev_shift,
                                      time[half_n + n],
                                      win[n],
                                      1,
                                      curr_shift,
                                      pcm_exp);
      pcm_store_q31(pcm_out, fmt, n * stride + channel_idx, first_pcm);

      int32_t second_pcm = ola_mix_q31(prev_time[n],
                                       win[half_n - 1 - n],
                                       -1,
                                       prev_shift,
                                       time[N - 1 - n],
                                       win[half_n + n],
                                       -1,
                                       curr_shift,
                                       pcm_exp);
      pcm_store_q31(pcm_out, fmt, (half_n + n) * stride + channel_idx, second_pcm);
    }
  }

  uint32_t overlap_mag = 0;
  for (int i = 0; i < half_n; i++) {
    overlap_mag |= mag_or_i32(time[i]);
  }

  int overlap_headroom = fxp_headroom_u32(overlap_mag);
  if (overlap_headroom > 0) {
    for (int i = 0; i < half_n; i++) {
      ola->overlap[i] = (int32_t)((uint32_t)time[i] << overlap_headroom);
    }
  } else {
    for (int i = 0; i < half_n; i++) {
      ola->overlap[i] = time[i];
    }
  }
  ola->loss_bits = curr_exp - overlap_headroom;

  return HQLC_OK;
}
