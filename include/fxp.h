#ifndef HQLC_FXP_H
#define HQLC_FXP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FXP_Q31_MAX ((int32_t)0x7FFFFFFF)
#define FXP_Q31_MIN ((int32_t)0x80000000)

#define Q8(x)  ((int32_t)((x) * 256.0 + 0.5))
#define Q30(x) ((int32_t)((x) * (1 << 30) + 0.5))

/**
 * @brief Return the saturating absolute value of a 32-bit integer.
 *
 * @param x Value to convert.
 * @return Absolute value, saturated to INT32_MAX for INT32_MIN.
 */
static inline int32_t fxp_abs_i32(int32_t x) {
  if (x == (int32_t)0x80000000) {
    return (int32_t)0x7FFFFFFF;
  }
  return x < 0 ? -x : x;
}

/**
 * @brief Multiply two Q31 values.
 *
 * @param a First Q31 value.
 * @param b Second Q31 value.
 * @return Product in Q31 format.
 */
static inline int32_t fxp_mul_q31(int32_t a, int32_t b) {
  return (int32_t)((int64_t)a * b >> 31);
}

/**
 * @brief Saturate a 64-bit integer to the 32-bit signed range.
 *
 * @param x Value to saturate.
 * @return `x` clamped to INT32_MIN..INT32_MAX.
 */
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
 * @brief Left-shift a 32-bit integer with saturation.
 *
 * @param x Value to shift.
 * @param shift Shift amount. Non-positive values leave `x` unchanged.
 * @return Shifted value, saturated on overflow.
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
 * @brief Right-shift a 32-bit integer with rounding.
 *
 * @param x Value to shift.
 * @param shift Shift amount. Non-positive values leave `x` unchanged.
 * @return Rounded shifted value.
 */
static inline int32_t fxp_shr_rnd_i32(int32_t x, int shift) {
  if (shift <= 0) {
    return x;
  }
  return (x >> shift) + ((x >> (shift - 1)) & 1);
}

/**
 * @brief Clamp a 32-bit integer to a range.
 *
 * @param x Value to clamp.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return `x` clamped to `lo`..`hi`.
 */
static inline int32_t fxp_clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/**
 * @brief Convert a fixed-point value to an integer with rounding.
 *
 * Computes `round(x / 2^frac_bits)`.
 *
 * @param x Fixed-point value.
 * @param frac_bits Number of fractional bits in `x`.
 * @return Rounded integer value.
 */
static inline int32_t fxp_round_to_int(int32_t x, int frac_bits) {
  return (x + (1 << (frac_bits - 1))) >> frac_bits;
}

/**
 * @brief Count safe left-shift bits for an unsigned OR accumulator.
 *
 * Zero input is treated as all-silent and returns 31.
 *
 * @param or_acc OR accumulator value.
 * @return Number of safe left-shift bits while preserving a sign-bit guard.
 */
static inline int fxp_headroom_u32(uint32_t or_acc) {
  return (or_acc == 0) ? 31 : __builtin_clz(or_acc) - 1;
}

// log2(1 + i/64) in Q8, used by fxp_log2_q8_u64()
static const uint8_t fxp_log2_frac_q8[64] = {
    0,   6,   11,  17,  22,  28,  33,  38,  44,  49,  54,  59,  63,  68,  73,  78,
    82,  87,  92,  96,  100, 105, 109, 113, 118, 122, 126, 130, 134, 138, 142, 146,
    150, 154, 157, 161, 165, 169, 172, 176, 179, 183, 186, 190, 193, 197, 200, 203,
    207, 210, 213, 216, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253,
};

/**
 * @brief Approximate log2 of a nonzero unsigned integer in Q8 format.
 *
 * Uses a small lookup table for the first six fractional mantissa bits.
 *
 * @param x Nonzero input value.
 * @return Approximate `log2(x)` in Q8 format.
 */
static inline int32_t fxp_log2_q8_u64(uint64_t x) {
  // x must be non-zero. msb is floor(log2(x))
  int msb = 63 - __builtin_clzll(x);

  // Normalize x to [1, 2) and keep the first 6 fractional mantissa bits.
  // Example: if msb=10, bits 9..4 index log2(1 + i/64).
  uint64_t norm = (msb >= 6) ? (x >> (msb - 6)) : (x << (6 - msb));
  int frac_idx = (int)(norm & 0x3F);

  return msb * 256 + (int32_t)fxp_log2_frac_q8[frac_idx];
}

#ifdef __cplusplus
}
#endif

#endif // HQLC_FXP_H
