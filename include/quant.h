#ifndef HQLC_QUANT_H
#define HQLC_QUANT_H

#include <stdbool.h>
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
#define QUANT_GAIN_RC_MAX   75 // GAIN_BIAS + GAIN_Q * 6

// Step exponent offset: E = 2*exp - gain_code - 59
// step = 2^(E/8), split into octave + fractional pow2 LUT
#define QUANT_EXP_OFFSET 59

// Q-format: Q31(spec) * Q28(inv_step) needs TOTAL_Q bits of shift to get Q8
#define QUANT_TOTAL_Q 51

// Deadzone: |scaled| < DZ = zero, rounding bias = 1 - DZ. DZ = 0.65
// +1 makes the compare strict at boundary
#define QUANT_DZ_THRESH_Q8 (FXP_Q(0.65, 8) + 1)
#define QUANT_DZ_BIAS_Q8   FXP_Q8(0.35)

// Centroid: MMSE-optimal reconstruction offset for Laplacian source
#define QUANT_CENTROID 0.15

// Noise fill seed
#define NF_SEED_BIAS 0x9E3779B9u

// 2^(f/8) for f=0..7, Q30, shared between quantizer and psy
extern const int32_t quant_pow2_eighth_q30[8];

/**
 * @brief Quantize a spectrum.
 *
 * For each bin, scales by the quantizer step, applies the deadzone, and rounds
 * the result.
 *
 * @param spec_q31 Input spectrum in Q31 BFP format.
 * @param loss_bits BFP exponent for `spec_q31`.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param interp True to interpolate per-bin exponents, false to use band centers.
 * @param quant_out Destination for quantized coefficients.
 */
void quant_forward(const int32_t *spec_q31,
                   int loss_bits,
                   const int32_t *exp_indices,
                   int gain_code,
                   bool interp,
                   int16_t *quant_out);

/**
 * @brief Reconstruct a spectrum from quantized coefficients.
 *
 * Uses `x_hat = sign(q) * (abs(q) + 0.15) * step` for nonzero symbols.
 *
 * @param quant_in Quantized coefficients.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param interp True to interpolate per-bin exponents, false to use band centers.
 * @param spec_q31 Destination for reconstructed Q31 BFP spectrum.
 * @param loss_bits_out Receives the BFP exponent for `spec_q31`.
 */
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   bool interp,
                   int32_t *spec_q31,
                   int *loss_bits_out);

/**
 * @brief Encode a floating-point gain as a 7-bit gain code.
 *
 * @param gain Floating-point quantizer gain.
 * @return Encoded gain code.
 */
int quant_gain_encode(float gain);

/**
 * @brief Reconstruct a sparse noise floor from decoded zero occupancy.
 *
 * @param quant Quantized coefficients used to measure per-band zero occupancy.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param interp Use interpolated per-bin exponents when true
 * @param seed Seed for the noise generator.
 * @param skip_bands Optional per-band skip mask (used to skip flagged M/S side bands)
 * @param spec_q31 Dequantized Q31 BFP spectrum to update.
 * @param loss_bits_io BFP exponent for `spec_q31`, updated if renormalized
 */
void noise_fill(const int16_t *quant,
                const int32_t *exp_indices,
                int gain_code,
                bool interp,
                uint32_t seed,
                const bool *skip_bands,
                int32_t *spec_q31,
                int *loss_bits_io);

#ifdef __cplusplus
}
#endif

#endif // HQLC_QUANT_H
