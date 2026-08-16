#ifndef HQLC_FXP_H
#define HQLC_FXP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FXP_Q31_MAX INT32_MAX
#define FXP_Q31_MIN INT32_MIN

// Compile-time floating-point literal to Q-format mapping.
#define FXP_Q(x, n) \
  ((int32_t)((x) * (double)(UINT64_C(1) << (n)) + ((x) < 0 ? -0.5 : 0.5)))

// Shorthand for common q formats
#define FXP_Q8(x)  FXP_Q(x, 8)
#define FXP_Q15(x) FXP_Q(x, 15)
#define FXP_Q16(x) FXP_Q(x, 16)
#define FXP_Q30(x) FXP_Q(x, 30)

/**
 * A view into an int32_t array, keeping track of its BFP exponents.
 * Used with bfp_* operations
 *
 * Each real sample is `data[i] * 2^(exp2 - 31)`
 */
typedef struct {
  int32_t *data;
  size_t length;
  int16_t exp2;
} bfp_i32;

// Shifts needed to express two BFP blocks in one exponent domain
typedef struct {
  int16_t common_exp2;
  uint8_t a_rshift;
  uint8_t b_rshift;
} bfp_alignment;

// log2(1 + i/64) in Q8, used by fxp_log2_q8_u64()
static const uint8_t fxp_log2_frac_q8[64] = {
    0,   6,   11,  17,  22,  28,  33,  38,  44,  49,  54,  59,  63,  68,  73,  78,
    82,  87,  92,  96,  100, 105, 109, 113, 118, 122, 126, 130, 134, 138, 142, 146,
    150, 154, 157, 161, 165, 169, 172, 176, 179, 183, 186, 190, 193, 197, 200, 203,
    207, 210, 213, 216, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253,
};

/**
 * @brief Return the saturating absolute value of a 32-bit integer.
 *
 * @param x Value to convert.
 * @return Absolute value, saturated to INT32_MAX for INT32_MIN.
 */
static inline int32_t fxp_abs_i32(int32_t x) {
  if (x == INT32_MIN) {
    return INT32_MAX;
  }
  return (x < 0) ? -x : x;
}

/**
 * @brief Multiply two 32-bit values and arithmetic-right-shift the product.
 *
 * A shift of 63 or greater returns the sign extension of the 64-bit product.
 *
 * @param a First value.
 * @param b Second value.
 * @param rshift Number of bits to right-shift the 64-bit product.
 * @return Shifted product, retained as a 64-bit value.
 */
static inline int64_t fxp_mul_rshift_i64(int32_t a, int32_t b, uint8_t rshift) {
  int64_t product = (int64_t)a * b;
  if (rshift >= 63) {
    return (product < 0) ? -1 : 0;
  }
  return product >> rshift;
}

/**
 * @brief Multiply two 32-bit values, right-shift, and narrow the result.
 *
 * The caller is responsible for ensuring the shifted result fits in 32 bits.
 */
static inline int32_t fxp_mul_rshift_i32(int32_t a, int32_t b, uint8_t rshift) {
  return (int32_t)fxp_mul_rshift_i64(a, b, rshift);
}

/**
 * @brief Same as fxp_mul_rshift_i32, but using a rounding shift
 */
static inline int32_t fxp_mul_rshift_rnd_i32(int32_t a, int32_t b, uint8_t rshift) {
  int64_t product = (int64_t)a * b;
  if (rshift == 0) {
    return (int32_t)product;
  }
  if (rshift >= 64) {
    return 0;
  }
  return (int32_t)((product >> rshift) + ((product >> (rshift - 1)) & 1));
}

/**
 * @brief Multiply two Q31 values.
 */
static inline int32_t fxp_mul_q31(int32_t a_q31, int32_t b_q31) {
  return fxp_mul_rshift_i32(a_q31, b_q31, 31);
}

/**
 * @brief Multiply two Q30 values.
 */
static inline int32_t fxp_mul_q30(int32_t a_q30, int32_t b_q30) {
  return fxp_mul_rshift_i32(a_q30, b_q30, 30);
}

/**
 * @brief Scale a 32-bit value by a Q30 factor, preserving its Q format.
 */
static inline int32_t fxp_scale_q30(int32_t value, int32_t scale_q30) {
  return fxp_mul_rshift_i32(value, scale_q30, 30);
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
  if (x == 0) {
    return 0;
  }
  if (shift >= 31) {
    return (x > 0) ? FXP_Q31_MAX : FXP_Q31_MIN;
  }
  return fxp_sat_i64_to_i32((int64_t)x * ((int64_t)1 << shift));
}

/**
 * @brief Same as fxp_shl_sat_i32, but using a rounding shift
 */
static inline int32_t fxp_shr_rnd_i32(int32_t x, int shift) {
  if (shift <= 0) {
    return x;
  }
  if (shift >= 32) {
    return 0;
  }
  return (x >> shift) + ((x >> (shift - 1)) & 1);
}

/**
 * @brief Convert a fixed-point value between Q formats.
 *
 * If the result does not fit in int32 range, its saturated
 *
 * @param x Source fixed-point value.
 * @param src_frac_bits Number of fractional bits in `x`.
 * @param dst_frac_bits Number of fractional bits in the result.
 * @return `x` represented with `dst_frac_bits` fractional bits
 */
static inline int32_t
fxp_rescale_i32(int32_t x, unsigned src_frac_bits, unsigned dst_frac_bits) {
  if (dst_frac_bits > src_frac_bits) {
    unsigned shift = dst_frac_bits - src_frac_bits;
    if (shift >= 31) {
      if (x == 0) {
        return 0;
      }
      return (x > 0) ? FXP_Q31_MAX : FXP_Q31_MIN;
    }
    return fxp_shl_sat_i32(x, (int)shift);
  }
  if (src_frac_bits > dst_frac_bits) {
    unsigned shift = src_frac_bits - dst_frac_bits;
    return (shift >= 32) ? 0 : fxp_shr_rnd_i32(x, (int)shift);
  }
  return x;
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
 * @param frac_bits Number of fractional bits in `x`
 * @return Rounded integer value.
 */
static inline int32_t fxp_round_to_int(int32_t x, int frac_bits) {
  return fxp_shr_rnd_i32(x, frac_bits);
}

/**
 * @brief Count safe left-shift bits for an unsigned OR accumulator.
 *
 * Zero input is treated as all-silent and returns 31
 *
 * @param magnitude_or OR accumulator of sign-stripped magnitudes.
 * @return Number of safe left-shift bits for signed int32 mantissas.
 */
static inline int fxp_signed_headroom_u32(uint32_t magnitude_or) {
  return (magnitude_or == 0) ? 31 : __builtin_clz(magnitude_or) - 1;
}

/**
 * @brief Approximate log2 of a nonzero unsigned integer in Q8 format
 *
 * Uses a small lookup table for the first six fractional mantissa bits
 *
 * @param x Nonzero input value.
 * @return Approximate `log2(x)` in Q8 format.
 */
static inline int32_t fxp_log2_q8_u64(uint64_t x) {
  // x must be non-zero. msb is floor(log2(x))
  int msb = 63 - __builtin_clzll(x);

  // Normalize x to (1, 2) and keep the first 6 fractional mantissa bits
  // Example: if msb=10, bits 9..4 index log2(1 + i/64)
  uint64_t norm = (msb >= 6) ? (x >> (msb - 6)) : (x << (6 - msb));
  int frac_idx = (int)(norm & 0x3F);

  return msb * 256 + (int32_t)fxp_log2_frac_q8[frac_idx];
}

/**
 * @brief Construct a BFP view from params
 */
static inline bfp_i32 bfp_i32_view(int32_t *data, size_t length, int exp2) {
  bfp_i32 block = {.data = data, .length = length, .exp2 = exp2};
  return block;
}

/**
 * @brief Returns true if every mantissa in block is zero
 */
static inline bool bfp_i32_is_zero(const bfp_i32 *block) {
  uint32_t nonzero_bits = 0;
  for (size_t i = 0; i < block->length; i++) {
    nonzero_bits |= (uint32_t)block->data[i];
  }
  return nonzero_bits == 0;
}

/**
 * @brief Return the number of bits by which every mantissa can safely be left-shifted
 */
static inline int bfp_i32_headroom(const bfp_i32 *block) {
  uint32_t magnitude_or = 0;
  for (size_t i = 0; i < block->length; i++) {
    int32_t value = block->data[i];
    magnitude_or |= (uint32_t)(value ^ (value >> 31));
  }
  return fxp_signed_headroom_u32(magnitude_or);
}

/**
 * Left-renormalize a block using previously measured headroom
 *
 * @param block BFP block to renormalize, mantissas are shifted while the exp gets
 * adjusted
 * @param headroom Shift amount of what to renormalize with
 */
static inline void bfp_i32_renormalize(bfp_i32 *block, uint8_t headroom) {
  // Headroom is a mantissa property, so values above 31 are equivalent to 31.
  if (headroom > 31) {
    headroom = 31;
  }
  for (size_t i = 0; i < block->length; i++) {
    block->data[i] = (int32_t)((uint32_t)block->data[i] << headroom);
  }
  block->exp2 -= headroom;
}

/**
 * @brief Coarsens the BFP into a given target exponent, attempting to refine does
 * nothing.
 */
static inline void bfp_i32_coarsen(bfp_i32 *block, int target_exp2) {
  if (target_exp2 <= block->exp2) {
    return;
  }
  unsigned rshift = (unsigned)(target_exp2 - block->exp2);
  for (size_t i = 0; i < block->length; i++) {
    block->data[i] = (rshift >= 32) ? 0 : fxp_shr_rnd_i32(block->data[i], (int)rshift);
  }
  block->exp2 = target_exp2;
}

/** Coarsens if necessary to keep `required` bits of headroom */
static inline void bfp_i32_ensure_headroom(bfp_i32 *block, int required) {
  int available = bfp_i32_headroom(block);
  if (available < required) {
    bfp_i32_coarsen(block, block->exp2 + required - available);
  }
}

// Calculated a shift between two exponents, clamping at 32 to avoid wrapping.
static inline uint8_t bfp_i32_alignment_shift(int common_exp2, int block_exp2) {
  int shift = common_exp2 - block_exp2;
  return (shift >= 32) ? 32 : (uint8_t)shift;
}

/**
 * Compute, without modifying either block, how to align them with headroom.
 * An all-zero block is exponent-neutral and never coarsens a nonzero block.
 */
static inline bfp_alignment
bfp_i32_alignment(const bfp_i32 *a, const bfp_i32 *b, uint8_t result_headroom) {
  bool a_is_zero = bfp_i32_is_zero(a);
  bool b_is_zero = bfp_i32_is_zero(b);

  int common_exp2;
  if (a_is_zero && !b_is_zero) {
    common_exp2 = b->exp2;
  } else if (!a_is_zero && b_is_zero) {
    common_exp2 = a->exp2;
  } else {
    common_exp2 = (a->exp2 > b->exp2) ? a->exp2 : b->exp2;
  }
  common_exp2 += (int)result_headroom;

  bfp_alignment alignment = {
      .common_exp2 = common_exp2,
      .a_rshift = a_is_zero ? 0 : bfp_i32_alignment_shift(common_exp2, a->exp2),
      .b_rshift = b_is_zero ? 0 : bfp_i32_alignment_shift(common_exp2, b->exp2),
  };
  return alignment;
}

/**
 * @brief Align two BFP views into a shared exponent
 *
 * @param first BFP view
 * @param second BFP view
 * @param result_headroom the amount of headroom bits to keep after alignment
 */
static inline void bfp_i32_align_pair(bfp_i32 *a, bfp_i32 *b, uint8_t result_headroom) {
  bfp_alignment alignment = bfp_i32_alignment(a, b, result_headroom);
  bfp_i32_coarsen(a, alignment.common_exp2);
  bfp_i32_coarsen(b, alignment.common_exp2);

  a->exp2 = alignment.common_exp2;
  b->exp2 = alignment.common_exp2;
}

#ifdef __cplusplus
}
#endif

#endif // HQLC_FXP_H
