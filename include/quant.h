#ifndef HQLC_QUANT_H
#define HQLC_QUANT_H

#include <stdbool.h>
#include <stdint.h>

#include "fxp.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gain code: 7-bit, 8 codes per octave, bias at gain=1.0
#define QUANT_GAIN_BITS   7
#define QUANT_GAIN_Q      8
#define QUANT_GAIN_BIAS   27
#define QUANT_GAIN_RC_MAX 75 // GAIN_BIAS + GAIN_Q * 6

// Quantizer rounding bias = 1 - deadzone, shared with the rate probe
#define QUANT_DZ_BIAS_Q8 FXP_Q8(0.35)

// Noise fill seeds, bias ^ channel offset
#define NF_SEED_BIAS 0x9E3779B9u
#define NF_SEED_CH   0x51EDu

// FNV prime for multiply by envelope index
// See https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
#define NF_SEED_MUL  0x01000193u

// Per-band noise fill mode, transmitted from nf_first_band() up
typedef enum {
  NF_FILL = 0, // a cleared mask fills everywhere
  NF_ATT,      // attenuated fill (used for lower bitrates)
  NF_OFF,
} nf_mode;

// From this gain code up (around 90kbps) killed bands never get filled
#define NF_ATT_GAIN_MAX 49

// First coarse band eligible for noise fill, ~2 kHz at low rates, ~3.5 kHz above
#define NF_FIRST_BAND     10
#define NF_FIRST_BAND_LOW 7

static inline int nf_first_band(int gain_code) {
  return gain_code < NF_ATT_GAIN_MAX ? NF_FIRST_BAND_LOW : NF_FIRST_BAND;
}

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
 * @brief Transmitted NF kill mask, used to skip NF where it overshoots
 *
 * Encoder side, mirrors the decoder NF logic so the estimates match exactly
 *
 * @param spectrum Encoder-side spectrum (post TNS, pre quantization)
 * @param quant Quantized coefficients
 * @param exp_indices Per-band exponent indices
 * @param gain_code Quantizer gain code
 * @param mask Destination, one NF mode per coarse band. Pre nf-band modes stay NF_FILL
 */
void noise_fill_calculate_mask(const bfp_i32 *spectrum,
                               const int16_t *quant,
                               const int32_t *exp_indices,
                               int gain_code,
                               nf_mode *mask);

/**
 * @brief Reconstruct a sparse noise floor from decoded zero occupancy
 *
 * @param quant Quantized coefficients used to measure per-band zero occupancy.
 * @param exp_indices Per-band exponent indices.
 * @param gain_code Quantizer gain code.
 * @param seed Seed for the noise generator.
 * @param band_mode Optional per-band NF mode (NF_OFF for flagged M/S side bands and
 * killed bands, NF_ATT for the attenuated fill)
 * @param spectrum Dequantized Q31 BFP spectrum to update.
 */
void noise_fill_apply(const int16_t *quant,
                      const int32_t *exp_indices,
                      int gain_code,
                      uint32_t seed,
                      const nf_mode *band_mode,
                      bfp_i32 *spectrum);

#ifdef __cplusplus
}
#endif

#endif // HQLC_QUANT_H
