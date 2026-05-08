#ifndef HQLC_QUANT_H
#define HQLC_QUANT_H

#include <stdint.h>

#include "fxp.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gain code: 7-bit, 8 codes per octave, bias at gain=1.0
#define QUANT_GAIN_BITS     7
#define QUANT_GAIN_Q        8
#define QUANT_GAIN_BIAS     27
#define QUANT_GAIN_MAX_CODE 127
#define QUANT_GAIN_RC_MAX   67  // GAIN_BIAS + GAIN_Q * 5

// Step exponent offset: E = 2*exp - gain_code - 59
// step = 2^(E/8), split into octave + fractional pow2 LUT
#define QUANT_EXP_OFFSET 59

// Q-format: Q31(spec) * Q28(inv_step) needs TOTAL_Q bits of shift to get Q8
#define QUANT_TOTAL_Q 51

// Deadzone: |scaled| < 0.65 → zero. Rounding bias = 1 - 0.65 = 0.35
#define QUANT_DZ_THRESH_Q8 ((int32_t)(0.65 * 256 + 1)) // ceil(0.65 * 256)
#define QUANT_DZ_BIAS_Q8   Q8(0.35)

// Centroid: MMSE-optimal reconstruction offset for Laplacian source
#define QUANT_CENTROID 0.15

// 2^(f/8) for f=0..7, Q30 — shared between quantizer and psy
extern const int32_t quant_pow2_eighth_q30[8];

/**
 * @brief Interpolate 20 coarse-band exponent indices to per-bin values.
 *
 * Output: bin_exp[PSY_ACTIVE_BINS]. Caller may reuse across quant/NF calls
 * for the same channel to avoid redundant computation.
 */
void quant_interp_bin_exp(const int32_t *exp_indices, int32_t *bin_exp);

/**
 * @brief Forward quantizer: MDCT spectrum → integer symbols.
 *
 * Per bin: scaled = |X| / step, then deadzone + round.
 * step = 2^((2*interp_exp - gain_code - 59) / 8).
 */
void quant_forward(const int32_t *spec_q31,
                   int loss_bits,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out);

/**
 * @brief Forward quantizer fused with NF estimation.
 *
 * Same as quant_forward but also computes the noise factor in the same pass,
 * avoiding a redundant interp_bin_exp + scale_to_q8 sweep.
 * Returns the 3-bit noise factor (0..7).
 */
int quant_forward_nf(const int32_t *spec_q31,
                     int loss_bits,
                     const int32_t *exp_indices,
                     int gain_code,
                     int16_t *quant_out);

/**
 * @brief Inverse quantizer: integer symbols → reconstructed spectrum.
 *
 * Per bin: x_hat = sign(q) * (|q| + 0.15) * step.
 * Output is Q31 BFP; loss_bits_out gives the exponent.
 * Uses flat per-band exponents (same as encoder quantizer).
 */
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   int32_t *spec_q31,
                   int *loss_bits_out);

/**
 * @brief Encode a floating-point gain to a 7-bit gain code.
 */
int quant_gain_encode(float gain);

// Noise fill seed
#define NF_SEED_BIAS 0x9E3779B9u

/**
 * @brief Fill runs of zeros with shaped pseudorandom noise.
 *
 * Runs of >4 consecutive zeros get filled at ±nf_amp * step.
 * Operates on the already-dequantized BFP spectrum.
 * Uses flat per-band exponents (same as encoder quantizer).
 */
void nf_run_length_fill(int16_t *quant,
                        const int32_t *exp_indices,
                        int gain_code,
                        int nf,
                        int32_t *spec_q31,
                        int *loss_bits_io);

#ifdef __cplusplus
}
#endif

#endif // HQLC_QUANT_H
