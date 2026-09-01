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
#define QUANT_GAIN_RC_MAX   75 // GAIN_BIAS + GAIN_Q * 6

// Quantizer rounding bias = 1 - deadzone, shared with the rate probe
#define QUANT_DZ_BIAS_Q8 FXP_Q8(0.35)

// Noise fill seed
#define NF_SEED_BIAS 0x9E3779B9u

// First coarse band eligible for noise fill
#define NF_FIRST_BAND 10

/** Scaling needed to convert a BFP spectrum magnitude to quantizer Q8. */
typedef struct {
  int32_t multiplier_q28;
  int product_rshift;
} quant_scale;

/**
 * @brief Derive the forward-quantizer scale for one envelope exponent.
 */
quant_scale quant_forward_scale(int exp_index, int gain_code, int spectrum_exp);

/**
 * @brief Quantize a spectrum.
 *
 * For each bin, scales by the quantizer step, applies the deadzone, and rounds
 * the result.
 *
 * @param spectrum Input Q31 BFP spectrum.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param quant_out Destination for quantized coefficients.
 */
void quant_forward(const bfp_i32 *spectrum,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out);

/**
 * @brief Reconstruct a spectrum from quantized coefficients.
 *
 * Uses `x_hat = sign(q) * (abs(q) + 0.15) * step` for nonzero symbols.
 *
 * @param quant_in Quantized coefficients.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param spectrum Destination for the reconstructed Q31 BFP spectrum.
 */
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   bfp_i32 *spectrum);

/**
 * @brief Transmitted NF refinement mask, used to skip NF where it overshoots
 *
 * Encoder side, mirrors the decoder NF logic so the estimates match exactly
 *
 * @param spectrum Encoder-side spectrum (post TNS, pre quantization)
 * @param quant Quantized coefficients
 * @param exp_indices Per-band exponent indices
 * @param gain_code Quantizer gain code
 * @param mask Destination, one flag per coarse band. Pre nf-band flags will stay false
 */
void noise_fill_refinement_mask(const bfp_i32 *spectrum,
                                const int16_t *quant,
                                const int32_t *exp_indices,
                                int gain_code,
                                bool *mask);

/**
 * @brief Reconstruct a sparse noise floor from decoded zero occupancy
 *
 * @param quant Quantized coefficients used to measure per-band zero occupancy.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param seed Seed for the noise generator.
 * @param skip_bands Optional per-band skip mask (flagged M/S side bands and the nf refinement mask)
 * @param spectrum Dequantized Q31 BFP spectrum to update.
 */
void noise_fill(const int16_t *quant,
                const int32_t *exp_indices,
                int gain_code,
                uint32_t seed,
                const bool *skip_bands,
                bfp_i32 *spectrum);

#ifdef __cplusplus
}
#endif

#endif // HQLC_QUANT_H
