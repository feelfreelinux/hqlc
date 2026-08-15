#ifndef HQLC_MDCT_TABLES_H
#define HQLC_MDCT_TABLES_H

#include <stdint.h>

#define MDCT_KBD_WINDOW_HALF  512
#define MDCT_PRE_TWIDDLE_LEN  512
#define MDCT_POST_TWIDDLE_LEN 512
#define MDCT_FFT_TWIDDLE_LEN  256

// Lookup table for the half-size KBD window, mirrored across the center to save memory
extern const int32_t kbd_window_half_q31[MDCT_KBD_WINDOW_HALF];

// Lookup tables for MDCT pre/post twiddle factors and FFT twiddle factors
extern const int32_t lut_pre_twiddle_q31[MDCT_PRE_TWIDDLE_LEN];
extern const int32_t lut_post_twiddle_q31[MDCT_POST_TWIDDLE_LEN];
extern const int32_t lut_fft_twiddle_q31[MDCT_FFT_TWIDDLE_LEN];

// Digit-reversal swap pairs for the 256-point radix-4 FFT.
// 120 pairs stored as (i, j).
#define MDCT_DIGIT_REV_PAIRS 120
extern const uint8_t lut_digit_rev[MDCT_DIGIT_REV_PAIRS * 2];

#endif // HQLC_MDCT_TABLES_H
