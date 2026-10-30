#ifndef HQLC_ENTROPY_TABLES_H
#define HQLC_ENTROPY_TABLES_H

#include <stdint.h>

#include "entropy.h"

// Exponent rANS models: three model types x four position groups, with
// centered magnitudes 0..11 and one escape slot.
#define RANS_EXP_NBANDS  20
#define RANS_EXP_NTABLES 12
#define RANS_EXP_NSLOTS  13

// Coefficient rANS probability model.
extern const uint16_t rans_cf[RANS_COEF_NTABLES][RANS_MAX_SYM + 1];
extern const uint32_t rans_rcp[RANS_COEF_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_cost_q8[RANS_COEF_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_log2_sigma_q8[RANS_COEF_NPAIRS];

extern const uint8_t rans_exp_group[RANS_EXP_NBANDS];
extern const int16_t rans_exp_center[RANS_EXP_NTABLES];
extern const uint16_t rans_exp_cf[RANS_EXP_NTABLES][RANS_EXP_NSLOTS + 1];
extern const uint32_t rans_exp_rcp[RANS_EXP_NTABLES][RANS_EXP_NSLOTS];
extern const int16_t rans_exp_cost_q8[RANS_EXP_NTABLES][RANS_EXP_NSLOTS];

#define RANS_NF_FLAG_NCTX 400
#define RANS_NF_ATT_NCTX  9

extern const uint16_t rans_nf_flag_lut[16];
extern const uint8_t rans_nf_flag_q4[RANS_NF_FLAG_NCTX / 2];
extern const uint16_t rans_nf_att_freq[RANS_NF_ATT_NCTX];

#endif // HQLC_ENTROPY_TABLES_H
