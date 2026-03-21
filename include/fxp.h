#ifndef HQLC_FXP_H
#define HQLC_FXP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FXP_Q31_MAX ((int32_t)0x7FFFFFFF)
#define FXP_Q31_MIN ((int32_t)0x80000000)

#define Q4(x)  ((int32_t)((x) * 16.0 + 0.5))
#define Q8(x)  ((int32_t)((x) * 256.0 + 0.5))
#define Q16(x) ((int32_t)((x) * 65536.0 + 0.5))
#define Q30(x) ((int32_t)((x) * (1 << 30) + 0.5))

// Saturating negation of a Q31 value
static inline int32_t fxp_neg_sat_i32(int32_t x) {
  if (x == (int32_t)0x80000000) {
    return (int32_t)0x7FFFFFFF;
  }
  return -x;
}

// Saturating absolute value of a 32-bit integer
static inline int32_t fxp_abs_i32(int32_t x) {
  if (x == (int32_t)0x80000000) {
    return (int32_t)0x7FFFFFFF;
  }
  return x < 0 ? -x : x;
}

// Multiply two Q31 values, returning a Q31 result
static inline int32_t fxp_mul_q31(int32_t a, int32_t b) {
  return (int32_t)((int64_t)a * b >> 31);
}

// Saturate a 64-bit integer to a 32-bit integer
static inline int32_t fxp_sat_i64_to_i32(int64_t x) {
  if (x > (int64_t)0x7FFFFFFF) {
    return (int32_t)0x7FFFFFFF;
  }
  if (x < (int64_t)(-0x80000000LL)) {
    return (int32_t)0x80000000;
  }
  return (int32_t)x;
}

/**
 * @brief Left-shift a 32-bit integer with saturation
 *
 * @param x Value to shift
 * @param shift Shift amount
 * @return Saturated result
 */
static inline int32_t fxp_shl_sat_i32(int32_t x, int shift) {
  if (shift <= 0) {
    return x;
  }
  int32_t r = (int32_t)((uint32_t)x << shift);
  if ((r >> shift) != x) {
    return (x > 0) ? FXP_Q31_MAX : FXP_Q31_MIN;
  }
  return r;
}

/**
 * @brief Right-shift a 32-bit integer with rounding
 *
 * @param x     Value to shift
 * @param shift Shift amount
 * @return Rounded result
 */
static inline int32_t fxp_shr_rnd_i32(int32_t x, int shift) {
  if (shift <= 0) {
    return x;
  }
  return (x >> shift) + ((x >> (shift - 1)) & 1);
}

// Integer max/min
static inline int32_t fxp_max_i32(int32_t a, int32_t b) {
  return a > b ? a : b;
}

static inline int32_t fxp_min_i32(int32_t a, int32_t b) {
  return a < b ? a : b;
}

static inline int32_t fxp_clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

// Headroom of a uint32 OR-accumulator: number of safe left-shift bits.
// Returns 31 for zero (all-silent), otherwise clz - 1 (sign bit guard).
static inline int fxp_headroom_u32(uint32_t or_acc) {
  return (or_acc == 0) ? 31 : __builtin_clz(or_acc) - 1;
}

// Fixed-point log2 of a uint64, returns result in Q5 (5 fractional bits).
// Useful for energy/exponent computations. x must be > 0.
static const uint8_t _fxp_log2_frac_q5[16] = {
    0, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 24, 26, 27, 29, 31};

static inline int32_t fxp_log2_q5_u64(uint64_t x) {
  int msb = 63 - __builtin_clzll(x);
  int frac_idx =
      (msb >= 4) ? (int)((x >> (msb - 4)) & 0xF) : (int)((x << (4 - msb)) & 0xF);
  return msb * 32 + (int32_t)_fxp_log2_frac_q5[frac_idx];
}

#ifdef __cplusplus
}
#endif

#endif // HQLC_FXP_H
