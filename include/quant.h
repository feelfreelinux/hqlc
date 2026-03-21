#ifndef HQLC_QUANT_H
#define HQLC_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Params of the quantized gain codes
#define QUANT_GAIN_BITS     7
#define QUANT_GAIN_Q        8  // codes per octave
#define QUANT_GAIN_BIAS     48 // code for gain=1.0
#define QUANT_GAIN_MAX_CODE 127
#define QUANT_GAIN_RC_MAX   72 // GAIN_BIAS + GAIN_Q * 3

// Combined exponent offset: 2*PSY_EXP_INDEX_BIAS - QUANT_GAIN_BIAS.
// E = 2*exp_idx - gain_code - QUANT_EXP_OFFSET encodes the quantizer
// step size as step = 2^(E/8) * BW[b].
#define QUANT_EXP_OFFSET 38

// Quantizer Q-format shift: Q31(spec) + Q28(inv_step) - Q8(output) = 51
#define QUANT_TOTAL_Q 51

// Deadzone: coefficients below 0.65 quantize to zero, rounding bias = 1 - 0.65 = 0.35
#define QUANT_DZ_THRESH_Q8 ((int32_t)(0.65 * 256 + 1)) // ceil(0.65 * 256)
#define QUANT_DZ_BIAS_Q8   Q8(0.35)

// 2^(f/8) for f=0..7, in Q30
extern const int32_t quant_pow2_eighth_q30[8];

// 1 / BW[b] in Q28, where BW[b] = log2(bin_count[b]) / log2(81), clipped to [0.10, 1.0]
extern const int32_t quant_inv_bw_q28[20];

/**
 * @brief Quantize spectral coefficients to int symbols
 *
 * @param spec_q31    MDCT coefficients
 * @param loss_bits   BFP exponent of the coeffs
 * @param exp_indices Exponent indices from the psy
 * @param gain_code   Gain code to quantize under
 * @param quant_out   Output for the quantized symbols
 */
void quant_forward(const int32_t *spec_q31,
                   int loss_bits,
                   const int32_t *exp_indices,
                   int gain_code,
                   int16_t *quant_out);

/**
 * @brief Reconstruct spectral coefficients from quantized symbols
 *
 * @param quant_in      Quantized symbol
 * @param exp_indices   Exponent indices
 * @param gain_code     Gain code to dequant from
 * @param spec_q31      Output reconstructed spectral coefficients
 * @param loss_bits_out Output BFP exponent for the reconstructed signal
 * @param work          int64_t * HQLC_FRAME_SAMPLES scratch buffer.
 */
void quant_inverse(const int16_t *quant_in,
                   const int32_t *exp_indices,
                   int gain_code,
                   int32_t *spec_q31,
                   int *loss_bits_out,
                   int64_t *work);

/**
 * @brief Encode a floating-point gain value to a gain code.
 *
 * @param gain Linear gain value.
 * @return Gain code
 */
int quant_gain_encode(float gain);

// Noise fill seed bias
#define NF_SEED_BIAS 0x9E3779B9u

/**
 * @brief Compute noise-fill amplitude, in the BFP loss domain
 *
 * @param exp_idx  Band exponent index
 * @param loss_bits BFP exponent for the frame
 * @return NF amplitude in Q31
 */
int32_t nf_compute_amp_q31(int exp_idx, int loss_bits);

/**
 * @brief Fill a spectral band with L1-normalized LCG noise
 *
 * @param spec        Spectral coefficient buffer
 * @param start       First bin of the band
 * @param end         Last bin of the band
 * @param nf_amp_q31  Noise-fill amplitude in Q31
 * @param seed        LCG seed value
 */
void nf_fill_band(int32_t *spec, int start, int end, int32_t nf_amp_q31, uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif // HQLC_QUANT_H
