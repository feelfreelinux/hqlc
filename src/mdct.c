#include "mdct.h"

#include <string.h>

#include "fxp.h"
#include "hqlc_bench.h"
#include "mdct_tables.h"
#include "pcm.h"

// Lookup a twiddle factor W_256^k from the half-period table.
// As an optimization, for k >= 128 we use W^k = -W^(k-128) to derive the negative value
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

// Core FFT with >>2 scaling per stage (>>8 total)
// N = 256 = 4^4
// Input is interleaved complex Q31: buf[2*i] = re, buf[2*i+1] = im
// Calculations are in place
static void fft_scaled(int32_t *buf) {
  const int n = MDCT_FFT_N;

  // Digit-reversal permutation via precomputed swap-pair LUT
  for (int k = 0; k < MDCT_DIGIT_REV_PAIRS; k++) {
    int i = lut_digit_rev[2 * k];
    int j = lut_digit_rev[2 * k + 1];
    int32_t tr = buf[2 * i], ti = buf[2 * i + 1];
    buf[2 * i] = buf[2 * j];
    buf[2 * i + 1] = buf[2 * j + 1];
    buf[2 * j] = tr;
    buf[2 * j + 1] = ti;
  }

  // Trivial radix-4 butterflies (all twiddles = 1), stage 0
  for (int base = 0; base < n; base += 4) {
    int32_t ar = buf[2 * base], ai = buf[2 * base + 1];
    int32_t br = buf[2 * (base + 1)], bi = buf[2 * (base + 1) + 1];
    int32_t cr = buf[2 * (base + 2)], ci = buf[2 * (base + 2) + 1];
    int32_t dr = buf[2 * (base + 3)], di = buf[2 * (base + 3) + 1];

    int32_t u0r = (ar >> 1) + (cr >> 1), u0i = (ai >> 1) + (ci >> 1);
    int32_t u1r = (ar >> 1) - (cr >> 1), u1i = (ai >> 1) - (ci >> 1);
    int32_t u2r = (br >> 1) + (dr >> 1), u2i = (bi >> 1) + (di >> 1);
    int32_t u3r = (br >> 1) - (dr >> 1), u3i = (bi >> 1) - (di >> 1);

    buf[2 * base] = (u0r >> 1) + (u2r >> 1);
    buf[2 * base + 1] = (u0i >> 1) + (u2i >> 1);
    buf[2 * (base + 1)] = (u1r >> 1) + (u3i >> 1);
    buf[2 * (base + 1) + 1] = (u1i >> 1) - (u3r >> 1);
    buf[2 * (base + 2)] = (u0r >> 1) - (u2r >> 1);
    buf[2 * (base + 2) + 1] = (u0i >> 1) - (u2i >> 1);
    buf[2 * (base + 3)] = (u1r >> 1) - (u3i >> 1);
    buf[2 * (base + 3) + 1] = (u1i >> 1) + (u3r >> 1);
  }

  // Stages 1–3, radix-4 butterflies with twiddle factors
  for (int s = 1; s <= 3; s++) {
    int stride = 1 << (2 * s);
    int block_size = stride << 2;
    int tw_step = n / block_size;

    for (int k = 0; k < stride; k++) {
      int32_t w1r, w1i, w2r, w2i, w3r, w3i;
      tw_lookup(k * tw_step, &w1r, &w1i);
      tw_lookup(2 * k * tw_step, &w2r, &w2i);
      tw_lookup(3 * k * tw_step, &w3r, &w3i);

      for (int b = 0; b < tw_step; b++) {
        int i0 = b * block_size + k;
        int i1 = i0 + stride;
        int i2 = i0 + 2 * stride;
        int i3 = i0 + 3 * stride;

        int32_t ar = buf[2 * i0], ai = buf[2 * i0 + 1];
        int32_t br = buf[2 * i1], bi = buf[2 * i1 + 1];
        int32_t cr = buf[2 * i2], ci = buf[2 * i2 + 1];
        int32_t dr = buf[2 * i3], di = buf[2 * i3 + 1];

        int32_t t1r = fxp_mul_q31(br, w1r) - fxp_mul_q31(bi, w1i);
        int32_t t1i = fxp_mul_q31(br, w1i) + fxp_mul_q31(bi, w1r);
        int32_t t2r = fxp_mul_q31(cr, w2r) - fxp_mul_q31(ci, w2i);
        int32_t t2i = fxp_mul_q31(cr, w2i) + fxp_mul_q31(ci, w2r);
        int32_t t3r = fxp_mul_q31(dr, w3r) - fxp_mul_q31(di, w3i);
        int32_t t3i = fxp_mul_q31(dr, w3i) + fxp_mul_q31(di, w3r);

        int32_t u0r = (ar >> 1) + (t2r >> 1), u0i = (ai >> 1) + (t2i >> 1);
        int32_t u1r = (ar >> 1) - (t2r >> 1), u1i = (ai >> 1) - (t2i >> 1);
        int32_t u2r = (t1r >> 1) + (t3r >> 1), u2i = (t1i >> 1) + (t3i >> 1);
        int32_t u3r = (t1r >> 1) - (t3r >> 1), u3i = (t1i >> 1) - (t3i >> 1);

        buf[2 * i0] = (u0r >> 1) + (u2r >> 1);
        buf[2 * i0 + 1] = (u0i >> 1) + (u2i >> 1);
        buf[2 * i1] = (u1r >> 1) + (u3i >> 1);
        buf[2 * i1 + 1] = (u1i >> 1) - (u3r >> 1);
        buf[2 * i2] = (u0r >> 1) - (u2r >> 1);
        buf[2 * i2 + 1] = (u0i >> 1) - (u2i >> 1);
        buf[2 * i3] = (u1r >> 1) - (u3i >> 1);
        buf[2 * i3 + 1] = (u1i >> 1) + (u3r >> 1);
      }
    }
  }
}

// In place DCT-IV using a half-length complex FFT
static void dct_iv(int32_t *restrict data, int32_t *restrict work) {
  const int N = MDCT_N;
  const int N_FFT = MDCT_FFT_N;

  // Pre-twiddle: pack real DCT input into complex FFT input.
  // Uses >>32 (mulsh-only) instead of >>1 + fxp_mul_q31 — same net precision.
  HQLC_BENCH_BEGIN();
  for (int k = 0; k < N_FFT; k++) {
    int32_t re = data[2 * k];
    int32_t im = data[N - 1 - 2 * k];
    int32_t wr = lut_pre_twiddle_q31[2 * k];
    int32_t wi = lut_pre_twiddle_q31[2 * k + 1];
    work[2 * k] = (int32_t)((int64_t)re * wr >> 32) - (int32_t)((int64_t)im * wi >> 32);
    work[2 * k + 1] =
        (int32_t)((int64_t)re * wi >> 32) + (int32_t)((int64_t)im * wr >> 32);
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_PRE_TW);

  HQLC_BENCH_BEGIN();
  // Do the FFT
  fft_scaled(work);
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FFT);

  // Post-twiddle: unpack FFT output to real DCT coefficients.
  // Uses >>32 (mulsh-only) — same tradeoff as pre-twiddle and inverse unfold.
  HQLC_BENCH_BEGIN();
  for (int k = 0; k < N_FFT; k++) {
    int32_t zr = work[2 * k], zi = work[2 * k + 1];
    int32_t wr = lut_post_twiddle_q31[2 * k];
    int32_t wi = lut_post_twiddle_q31[2 * k + 1];
    data[2 * k] = (int32_t)((int64_t)zr * wr >> 32) - (int32_t)((int64_t)zi * wi >> 32);
    data[N - 1 - 2 * k] =
        -((int32_t)((int64_t)zr * wi >> 32) + (int32_t)((int64_t)zi * wr >> 32));
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_POST_TW);
}

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
                        int *restrict loss_bits_out) {
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

  // Check buffer sizes and format
  size_t bps = (fmt == HQLC_PCM16) ? 2 : 3;
  size_t needed = (size_t)HQLC_FRAME_SAMPLES * (size_t)stride * bps;
  if (half_pcm_len < needed) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  const int N = MDCT_N;
  const int N2 = N / 2;

  int32_t *folded = spec_q31;

  // Window + fold into Q30 using >>32 (mulsh-only on Xtensa).
  // The >>32 combines the Q31 multiply with the >>1 overflow guard into one shift,
  // costing 1 LSB vs fxp_mul_q31 but saving 3 instructions per multiply.
  HQLC_BENCH_BEGIN();
  uint32_t or_acc = 0; // branchless abs via XOR: (v ^ (v >> 31)) avoids INT_MIN branch

  if (fmt == HQLC_PCM16) {
    const int16_t *curr = (const int16_t *)curr_pcm;
    const int16_t *prev = (const int16_t *)prev_pcm;
    const int32_t *wh = kbd_window_half_q31;

    for (int n = 0; n < N2; n++) {
      int32_t s1 = (int32_t)curr[(N2 + n) * stride + channel_idx] << 16;
      int32_t s2 = (int32_t)curr[(N2 - 1 - n) * stride + channel_idx] << 16;
      int32_t t1 = (int32_t)((int64_t)s1 * wh[N2 - 1 - n] >> 32);
      int32_t t2 = (int32_t)((int64_t)s2 * wh[N2 + n] >> 32);
      int32_t v = -t1 - t2;
      folded[n] = v;
      or_acc |= (uint32_t)(v ^ (v >> 31));
    }

    for (int n = 0; n < N2; n++) {
      int32_t s1 = (int32_t)prev[n * stride + channel_idx] << 16;
      int32_t s2 = (int32_t)prev[(N - 1 - n) * stride + channel_idx] << 16;
      int32_t t1 = (int32_t)((int64_t)s1 * wh[n] >> 32);
      int32_t t2 = (int32_t)((int64_t)s2 * wh[N - 1 - n] >> 32);
      int32_t v = t1 - t2;
      folded[N2 + n] = v;
      or_acc |= (uint32_t)(v ^ (v >> 31));
    }
  } else {
    for (int n = 0; n < N2; n++) {
      int32_t s1 = pcm_load_q31(curr_pcm, fmt, (N2 + n) * stride + channel_idx);
      int32_t s2 = pcm_load_q31(curr_pcm, fmt, (N2 - 1 - n) * stride + channel_idx);
      int32_t t1 = (int32_t)((int64_t)s1 * kbd_window_q31(3 * N2 + n) >> 32);
      int32_t t2 = (int32_t)((int64_t)s2 * kbd_window_q31(3 * N2 - 1 - n) >> 32);
      int32_t v = -t1 - t2;
      folded[n] = v;
      or_acc |= (uint32_t)(v ^ (v >> 31));
    }
    for (int n = 0; n < N2; n++) {
      int32_t s1 = pcm_load_q31(prev_pcm, fmt, n * stride + channel_idx);
      int32_t s2 = pcm_load_q31(prev_pcm, fmt, (N - 1 - n) * stride + channel_idx);
      int32_t t1 = (int32_t)((int64_t)s1 * kbd_window_q31(n) >> 32);
      int32_t t2 = (int32_t)((int64_t)s2 * kbd_window_q31(N - 1 - n) >> 32);
      int32_t v = t1 - t2;
      folded[N2 + n] = v;
      or_acc |= (uint32_t)(v ^ (v >> 31));
    }
  }

  // Headroom normalization (data is Q30 after the >>32 window multiply)
  int hr = fxp_headroom_u32(or_acc);
  int fold_gain;
  if (or_acc == 0) {
    fold_gain = 30;
  } else {
    if (hr > 0) {
      for (int i = 0; i < N; i++) {
        folded[i] <<= hr;
      }
    }
    fold_gain = hr - 1; // -1 accounts for the pre shift
  }
  HQLC_BENCH_END(HQLC_BENCH_MDCT_FOLD);

  int32_t *work = (int32_t *)scratch;
  dct_iv(folded, work);

  // Account for the DCT gain and the pre-shift normalization
  *loss_bits_out = -fold_gain + MDCT_DCT_BITS;
  return HQLC_OK;
}

hqlc_error mdct_inverse(const int32_t *restrict spec_q31,
                        size_t spec_q31_len,
                        int loss_bits_in,
                        int32_t *restrict windowed_q31,
                        size_t windowed_q31_len,
                        void *restrict scratch,
                        size_t scratch_len,
                        int *restrict loss_bits_out) {
  if (!spec_q31 || !windowed_q31 || !scratch || !loss_bits_out) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (spec_q31_len < (size_t)MDCT_N) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (windowed_q31_len < (size_t)MDCT_BLOCK_LEN) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }
  if (scratch_len < (size_t)MDCT_SCRATCH_BYTES) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  const int N = MDCT_N;
  const int N2 = N / 2;

  int32_t *u = (int32_t *)scratch;
  int32_t *fft_work = &u[N];

  memcpy(u, spec_q31, (size_t)N * sizeof(int32_t));
  dct_iv(u, fft_work);

  // Unfold + window (>>32 approximation, costs 1 bit)
  const int32_t *wh = kbd_window_half_q31;
  for (int n = 0; n < N2; n++) {
    windowed_q31[n] = (int32_t)((int64_t)u[N2 + n] * wh[n] >> 32);
    windowed_q31[N2 + n] = -(int32_t)((int64_t)u[N - 1 - n] * wh[N2 + n] >> 32);
    windowed_q31[N + n] = -(int32_t)((int64_t)u[N2 - 1 - n] * wh[N - 1 - n] >> 32);
    windowed_q31[N + N2 + n] = -(int32_t)((int64_t)u[n] * wh[N2 - 1 - n] >> 32);
  }

  *loss_bits_out = loss_bits_in + MDCT_DCT_BITS - MDCT_MATH_GAIN_BITS + 1;
  return HQLC_OK;
}
