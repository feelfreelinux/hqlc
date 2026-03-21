#pragma once
#include <stdint.h>

#define RANS_LUT_NBINS 32
#define RANS_MAX_SYM   16

extern const uint16_t rans_lut_freq[RANS_LUT_NBINS][RANS_MAX_SYM];
extern const uint32_t rans_lut_rcp[RANS_LUT_NBINS][RANS_MAX_SYM];
extern const uint32_t rans_lut_alpha_edges_q16[RANS_LUT_NBINS + 1];
extern const int32_t rans_k_bw_q16[20];
extern const uint8_t log2_frac_q8[128];
