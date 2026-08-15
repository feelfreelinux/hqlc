#include "entropy.h"
#include "entropy_tables.h"
#include "fxp.h"
#include "hqlc.h"
#include "hqlc_bench.h"
#include "ms.h"
#include "psy.h"
#include "quant.h"

#include <string.h>

// rANS renormalization, emit bytes until state < f << RANS_RENORM_SHIFT, keeping state
// in [RANS_L, RANS_L*256]. 14 = RANS_L_BITS(16) + BYTE_BITS(8) - RANS_M_BITS(10).
#define RANS_RENORM_SHIFT 14
#define RANS_BYTE_BITS    8

// Sign coding uses fixed probability split, done with no tables
#define RANS_SIGN_FREQ         (RANS_M / 2)
#define RANS_SIGN_SLOT_SHIFT   (RANS_M_BITS - 1)
#define RANS_SIGN_SLOT_MASK    (RANS_SIGN_FREQ - 1)
#define RANS_SIGN_RENORM_UPPER ((uint32_t)RANS_SIGN_FREQ << RANS_RENORM_SHIFT)

// Index definition for the rANS exponent models
typedef enum {
  RANS_EXP_DPCM = 0,
  RANS_EXP_CROSS_CHANNEL,
  RANS_EXP_CROSS_CHANNEL_MS,
} rans_exp_model;

// Sign encode for the fixed {M/2, M/2} split, shift/mask only, no tables
static inline void rans_enc_sign(hqlc_rans_enc *enc, uint8_t sign) {
  uint32_t state = enc->state;

  if (state >= RANS_SIGN_RENORM_UPPER) {
    enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
    state >>= RANS_BYTE_BITS;
    if (state >= RANS_SIGN_RENORM_UPPER) {
      enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
      state >>= RANS_BYTE_BITS;
    }
  }

  enc->state = ((state >> RANS_SIGN_SLOT_SHIFT) << RANS_M_BITS) +
               (state & RANS_SIGN_SLOT_MASK) + (sign ? RANS_SIGN_FREQ : 0u);
}

// General symbol encode, the reciprocal avoids division
static inline void
rans_enc_sym(hqlc_rans_enc *enc, uint16_t f, uint32_t rcp, uint16_t cf) {
  uint32_t upper = (uint32_t)f << RANS_RENORM_SHIFT;
  uint32_t state = enc->state;

  if (state >= upper) {
    enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
    state >>= 8;
    if (state >= upper) {
      enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
      state >>= 8;
    }
  }

  uint32_t q = (uint32_t)(((uint64_t)state * rcp) >> 32);
  uint32_t r = state - q * f;
  if (r >= f) {
    q++;
    r -= f;
  }

  enc->state = (q << RANS_M_BITS) + r + cf;
}

// Pull one renorm byte, flagging (and returning 0) past the buffer end so a
// corrupt or truncated stream can never read out of bounds.
static inline uint8_t rans_dec_byte(hqlc_rans_dec *dec) {
  if (dec->pos < dec->len) {
    return dec->buf[dec->pos++];
  }
  dec->overrun = true;
  return 0;
}

// Sign decode for the fixed {M/2, M/2} split
static inline uint8_t rans_dec_sign(hqlc_rans_dec *dec) {
  uint32_t state = dec->state;
  uint32_t slot = state & (RANS_M - 1);
  uint8_t s = (uint8_t)(slot >> RANS_SIGN_SLOT_SHIFT); // 0 = low half, 1 = high half

  // Inverse update: state = (M/2)*floor(state/M) + slot%(M/2)
  dec->state =
      ((state >> RANS_M_BITS) << RANS_SIGN_SLOT_SHIFT) + (slot & RANS_SIGN_SLOT_MASK);

  // Renorm (at most 2 bytes)
  if (dec->state < RANS_L) {
    dec->state = (dec->state << 8) | rans_dec_byte(dec);
    if (dec->state < RANS_L) {
      dec->state = (dec->state << 8) | rans_dec_byte(dec);
    }
  }
  return s;
}

static inline uint8_t rans_dec_sym(hqlc_rans_dec *dec, const uint16_t *cf) {
  uint32_t state = dec->state;
  uint32_t slot = state & (RANS_M - 1);

  // Linear scan: symbol 0 is most probable, so 1-2 iters typical
  // cf[RANS_MAX_SYM] == RANS_M > slot guarantees termination
  int s = 0;
  while (cf[s + 1] <= slot) {
    s++;
  }

  uint16_t f = cf[s + 1] - cf[s];
  dec->state = (uint32_t)f * (state >> RANS_M_BITS) + slot - cf[s];

  // Renorm
  if (dec->state < RANS_L) {
    dec->state = (dec->state << 8) | rans_dec_byte(dec);
    if (dec->state < RANS_L) {
      dec->state = (dec->state << 8) | rans_dec_byte(dec);
    }
  }
  return (uint8_t)s;
}

static inline int rans_exp_table_index(rans_exp_model model, int band) {
  return (int)model * 4 + rans_exp_group[band];
}

static int rans_exp_symbol_cost_q8(int table_index, int32_t value) {
  int32_t d = value - rans_exp_center[table_index];
  int mag = (d < 0) ? -d : d;
  int cost_q8 = (mag != 0) ? FXP_Q8(1.0) : 0; // sign
  if (mag < RANS_EXP_NSLOTS - 1) {
    return cost_q8 + rans_exp_cost_q8[table_index][mag];
  }
  int nbits = rans_eg0_nbits(mag - (RANS_EXP_NSLOTS - 1));
  int escape_cost_q8 = fxp_rescale_i32(2 * nbits + 1, 0, 8);
  return cost_q8 + rans_exp_cost_q8[table_index][RANS_EXP_NSLOTS - 1] + escape_cost_q8;
}

// Code a single rANS exponent symbol, reverse order to decoder (so decoder reads mag,
// escape, sign)
static void rans_exp_encode_symbol(hqlc_rans_enc *enc, int table_index, int32_t value) {
  const uint16_t *cf = rans_exp_cf[table_index];
  int32_t d = value - rans_exp_center[table_index];
  int mag = (d < 0) ? -d : d;
  int sym = (mag < RANS_EXP_NSLOTS - 1) ? mag : (RANS_EXP_NSLOTS - 1);

  // Code sign
  if (mag != 0) {
    rans_enc_sign(enc, (d > 0) ? 0 : 1);
  }
  if (mag >= RANS_EXP_NSLOTS - 1) {
    // Code the escape value using 50/50 coder
    int overflow = mag - (RANS_EXP_NSLOTS - 1);
    int nbits = rans_eg0_nbits(overflow);
    int val = overflow + 1;
    for (int i = 0; i < nbits; i++) {
      rans_enc_sign(enc, (uint8_t)((val >> i) & 1));
    }
    rans_enc_sign(enc, 1);
    for (int j = 0; j < nbits; j++) {
      rans_enc_sign(enc, 0);
    }
  }

  // Code the magnitude (contains the escape symbol too)
  rans_enc_sym(enc, cf[sym + 1] - cf[sym], rans_exp_rcp[table_index][sym], cf[sym]);
}

static int32_t rans_exp_decode_symbol(hqlc_rans_dec *dec, int table_index) {
  int sym = rans_dec_sym(dec, rans_exp_cf[table_index]);
  int32_t mag = sym;
  if (sym == RANS_EXP_NSLOTS - 1) {
    int nbits = 0;
    while (rans_dec_sign(dec) == 0 && nbits < 24 && !dec->overrun) {
      nbits++;
    }
    int32_t val = 1;
    for (int i = 0; i < nbits; i++) {
      val = (val << 1) | rans_dec_sign(dec);
    }
    mag = (RANS_EXP_NSLOTS - 1) + val - 1;
  }
  int32_t d = mag;
  if (mag != 0 && rans_dec_sign(dec)) {
    d = -mag;
  }
  return rans_exp_center[table_index] + d;
}

static int rans_exp_channel_cost_q8(const int32_t *exp_indices,
                                    int ch,
                                    const bool *ms_flags,
                                    bool *dpcm_selected) {
  const int32_t *e0 = exp_indices;
  const int32_t *e1 = &exp_indices[ch * PSY_N_BANDS];

  // Estimate both DPCM and cross channel delta cost
  int delta_cost_q8 = 0;
  int dpcm_cost_q8 = 0;

  int32_t prev = 0;
  for (int b = 0; b < PSY_N_BANDS; b++) {
    // M/S channels get a different rANS model
    rans_exp_model model =
        (ms_flags && ms_flags[b]) ? RANS_EXP_CROSS_CHANNEL_MS : RANS_EXP_CROSS_CHANNEL;
    delta_cost_q8 +=
        rans_exp_symbol_cost_q8(rans_exp_table_index(model, b), e1[b] - e0[b]);

    // DPCM cost model is same regardless of M/S
    dpcm_cost_q8 +=
        rans_exp_symbol_cost_q8(rans_exp_table_index(RANS_EXP_DPCM, b), e1[b] - prev);
    prev = e1[b];
  }

  // Pick the cheaper model
  bool select_dpcm = dpcm_cost_q8 < delta_cost_q8;
  if (dpcm_selected) {
    *dpcm_selected = select_dpcm;
  }

  // One bit for the model decision
  return FXP_Q8(1.0) + (select_dpcm ? dpcm_cost_q8 : delta_cost_q8);
}

int find_best_rice_k(const int32_t *values, int n) {
  int best_k = 0;
  int best_cost = 0x7FFFFFFF;
  for (int k = 0; k < 7; k++) { // k=0..6 covers 6-bit exponent deltas
    int cost = 0;
    for (int i = 0; i < n; i++) {
      uint32_t u = zigzag_enc(values[i]);
      cost += (int)(u >> k) + 1 + k;
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_k = k;
    }
  }
  return best_k;
}

void bw_write_rice(hqlc_bitwriter *w, uint32_t val, int k) {
  uint32_t q = val >> k;

  // Unary prefix, q ones then a zero
  for (uint32_t i = 0; i < q; i++) {
    bw_write(w, 1, 1);
  }
  bw_write(w, 0, 1);

  // k-bit remainder
  if (k > 0) {
    bw_write(w, val & ((1u << k) - 1), k);
  }
}

uint32_t br_read_rice(hqlc_bitreader *r, int k) {
  uint32_t q = 0;
  while (br_read(r, 1) && r->pos < r->len) {
    q++;
  }
  uint32_t rem = (k > 0) ? br_read(r, k) : 0;
  return (q << k) | rem;
}

void bw_write_binary_rle(hqlc_bitwriter *w, const bool *flags, int n_flags, int rice_k) {
  if (n_flags <= 0) {
    return;
  }

  // Write the initial color bit
  bw_write(w, flags[0] & 1, 1);
  int pos = 0;
  while (pos < n_flags) {
    int run = 1;
    while (pos + run < n_flags && flags[pos + run] == flags[pos]) {
      run++;
    }
    if (pos + run < n_flags) {
      bw_write(w, 1, 1); // another run follows
      bw_write_rice(w, (uint32_t)(run - 1), rice_k);
    }
    pos += run;
  }
  bw_write(w, 0, 1); // terminator
}

void br_read_binary_rle(hqlc_bitreader *r, bool *flags, int n_flags, int rice_k) {
  if (n_flags <= 0) {
    return;
  }

  // Read the initial color bit
  uint8_t cur = (uint8_t)br_read(r, 1);
  int pos = 0;
  for (int guard = 0; guard < n_flags && pos < n_flags; guard++) {
    if (!br_read(r, 1)) {
      break;
    }
    uint32_t run = br_read_rice(r, rice_k) + 1;
    for (uint32_t i = 0; i < run && pos < n_flags; i++) {
      flags[pos++] = cur;
    }
    cur ^= 1;
  }
  while (pos < n_flags) {
    flags[pos++] = cur;
  }
}

#ifdef HQLC_TRAIN_TABLES
// Optional rANS training histogram, used to adjust the entropy tables
uint64_t rans_train_hist[RANS_COEF_NTABLES][RANS_MAX_SYM];
int rans_train_enabled = 0;

void rans_train_reset(void) {
  memset(rans_train_hist, 0, sizeof(rans_train_hist));
  rans_train_enabled = 1;
}

#define RANS_TRAIN_COUNT(tidx, sym)     \
  do {                                  \
    if (rans_train_enabled) {           \
      rans_train_hist[(tidx)][(sym)]++; \
    }                                   \
  } while (0)
#else
#define RANS_TRAIN_COUNT(tidx, sym) \
  do {                              \
  } while (0)
#endif

void rans_enc_init(hqlc_rans_enc *enc, uint8_t *buf, size_t cap) {
  enc->state = RANS_L;
  enc->buf = buf;
  enc->cap = cap;
  enc->pos = cap; // write cursor starts at end
}

void rans_dec_init(hqlc_rans_dec *dec, const uint8_t *buf, size_t len) {
  dec->buf = buf;
  dec->len = len;
  dec->pos = 0;
  dec->state = 0;
  dec->overrun = false;
  for (int i = 0; i < 3 && dec->pos < len; i++) {
    dec->state = (dec->state << 8) | buf[dec->pos++];
  }
}

size_t rans_enc_flush(hqlc_rans_enc *enc) {
  // Emit the 3-byte final state (renorm keeps state < 1 << 24), then
  // left-align the stream to buf[0]
  uint32_t state = enc->state;
  enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
  state >>= 8;
  enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
  state >>= 8;
  enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);

  size_t len = enc->cap - enc->pos;
  memmove(enc->buf, enc->buf + enc->pos, len);
  enc->pos = 0;
  return len;
}

int rans_alpha_bin(int band, int gain_code) {
  // log2(alpha) = log2(gain) + log2(sigma_pair)
  // log2(gain) = (gc - GAIN_BIAS) / 8
  int pair = band >> 1;
  return rans_alpha_bin_from_la((gain_code - QUANT_GAIN_BIAS) * 32 +
                                rans_log2_sigma_q8[pair]);
}

int rans_activity_bin(const int16_t *quant, int band) {
  if (band == 0) {
    return 0;
  }
  int s = psy_band_edges[band - 1];
  int e = psy_band_edges[band];
  int nz = 0;
  for (int i = s; i < e; i++) {
    if (quant[i] != 0) {
      nz++;
    }
  }
  return rans_activity_from(nz, e - s);
}

int rans_exp_payload_cost_q8(const int32_t *exp_indices, int n_ch, const bool *ms_flags) {
  if (!exp_indices) {
    return 0;
  }

  int total_q8 = 0;
  for (int ch = 1; ch < n_ch; ch++) {
    total_q8 += rans_exp_channel_cost_q8(exp_indices, ch, ms_flags, NULL);
  }
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int32_t prev = (b > 0) ? exp_indices[b - 1] : 0;
    total_q8 += rans_exp_symbol_cost_q8(rans_exp_table_index(RANS_EXP_DPCM, b),
                                        exp_indices[b] - prev);
  }
  return total_q8;
}

size_t rans_encode_payload(const int16_t *quant,
                           int n_ch,
                           int gain_code,
                           const bool *ms_flags,
                           const int32_t *exp_indices,
                           uint8_t *out,
                           size_t out_cap) {
  hqlc_rans_enc enc;
  rans_enc_init(&enc, out, out_cap);

  // rANS encodes in reverse order, the decoder reads forward
  for (int ch = n_ch - 1; ch >= 0; ch--) {
    const int16_t *ch_q = &quant[ch * HQLC_FRAME_SAMPLES];

    for (int b = PSY_N_BANDS - 1; b >= 0; b--) {
      // Per-band table from alpha + activity of the previous band
      HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_RANS_TBL);
      int abin = rans_alpha_bin(b, gain_code);
      if (ch == 1 && ms_flags && ms_flags[b]) {
        abin -= MS_RANS_ALPHA_SHIFT;
        if (abin < 0) {
          abin = 0;
        }
      }
      int act = rans_activity_bin(ch_q, b);
      int tidx = rans_table_idx(abin, act);
      const uint16_t *cf = rans_cf[tidx];
      const uint32_t *rcp = rans_rcp[tidx];
      HQLC_BENCH_END(HQLC_BENCH_ENC_RANS_TBL);

      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];

      // We mostly code zeros, so keep it as a constant loaded here
      uint16_t f0 = cf[1]; // cf[0] == 0
      uint32_t rcp0 = rcp[0];

      for (int i = e - 1; i >= s; i--) {
        int16_t v = ch_q[i];
        if (v == 0) {
          rans_enc_sym(&enc, f0, rcp0, 0);
          RANS_TRAIN_COUNT(tidx, 0);
          continue;
        }

        int mag = (v < 0) ? -v : v;
        uint8_t sym =
            (mag < RANS_MAX_SYM - 1) ? (uint8_t)mag : (uint8_t)(RANS_MAX_SYM - 1);
        RANS_TRAIN_COUNT(tidx, sym);

        rans_enc_sign(&enc, (v > 0) ? 0 : 1);

        // Handle the escape (when mag > 15). uses the 50/50 binary coder
        if (mag >= RANS_MAX_SYM - 1) {
          int overflow = mag - (RANS_MAX_SYM - 1);
          int nbits = rans_eg0_nbits(overflow);
          int val = overflow + 1;
          for (int bit_idx = 0; bit_idx < nbits; bit_idx++) {
            rans_enc_sign(&enc, (val >> bit_idx) & 1);
          }
          rans_enc_sign(&enc, 1);
          for (int j = 0; j < nbits; j++) {
            rans_enc_sign(&enc, 0);
          }
        }

        rans_enc_sym(&enc, cf[sym + 1] - cf[sym], rcp[sym], cf[sym]);
      }
    }
  }

  // Exponents use the same rANS stream as the payload
  // First band always uses thes the DPCM coding (and appropriate rANS model)
  // Bands ch1+ select between a cross-channel coding deltas or a DPCM, depending on whats
  // cheaper The first bit in ch+ choses the right coding model
  if (exp_indices) {
    for (int ch = n_ch - 1; ch >= 1; ch--) {
      const int32_t *e1 = &exp_indices[ch * PSY_N_BANDS];
      bool dpcm_selected;
      rans_exp_channel_cost_q8(exp_indices, ch, ms_flags, &dpcm_selected);
      for (int b = PSY_N_BANDS - 1; b >= 0; b--) {
        if (dpcm_selected) {
          int32_t pv = (b > 0) ? e1[b - 1] : 0;
          rans_exp_encode_symbol(
              &enc, rans_exp_table_index(RANS_EXP_DPCM, b), e1[b] - pv);
        } else {
          // Select the right rANS model - M/S coded bands have a different distribution
          // after all
          rans_exp_model model = (ms_flags && ms_flags[b]) ? RANS_EXP_CROSS_CHANNEL_MS
                                                           : RANS_EXP_CROSS_CHANNEL;
          rans_exp_encode_symbol(
              &enc, rans_exp_table_index(model, b), e1[b] - exp_indices[b]);
        }
      }
      rans_enc_sign(&enc, dpcm_selected ? 1 : 0);
    }
    for (int b = PSY_N_BANDS - 1; b >= 0; b--) {
      int32_t pv = (b > 0) ? exp_indices[b - 1] : 0;
      rans_exp_encode_symbol(
          &enc, rans_exp_table_index(RANS_EXP_DPCM, b), exp_indices[b] - pv);
    }
  }

  return rans_enc_flush(&enc);
}

bool rans_decode_payload(const uint8_t *data,
                         size_t len,
                         int16_t *quant_out,
                         int n_ch,
                         int gain_code,
                         const bool *ms_flags,
                         int32_t *exp_indices) {
  if (len == 0) {
    memset(quant_out, 0, (size_t)n_ch * HQLC_FRAME_SAMPLES * sizeof(int16_t));
    return true;
  }

  hqlc_rans_dec dec;
  rans_dec_init(&dec, data, len);

  // Exponents come first in the stream (see rans_encode_payload)
  if (exp_indices) {
    int32_t prev = 0;
    for (int b = 0; b < PSY_N_BANDS; b++) {
      prev = fxp_clamp_i32(
          prev + rans_exp_decode_symbol(&dec, rans_exp_table_index(RANS_EXP_DPCM, b)),
          PSY_EXP_INDEX_MIN,
          PSY_EXP_INDEX_MAX);
      exp_indices[b] = prev;
    }
    for (int ch = 1; ch < n_ch; ch++) {
      bool dpcm_selected = rans_dec_sign(&dec);
      int32_t *e1 = &exp_indices[ch * PSY_N_BANDS];
      int32_t prev1 = 0;
      for (int b = 0; b < PSY_N_BANDS; b++) {
        if (dpcm_selected) {
          prev1 = fxp_clamp_i32(prev1 + rans_exp_decode_symbol(
                                            &dec, rans_exp_table_index(RANS_EXP_DPCM, b)),
                                PSY_EXP_INDEX_MIN,
                                PSY_EXP_INDEX_MAX);
          e1[b] = prev1;
        } else {
          rans_exp_model model = (ms_flags && ms_flags[b]) ? RANS_EXP_CROSS_CHANNEL_MS
                                                           : RANS_EXP_CROSS_CHANNEL;
          int32_t delta = rans_exp_decode_symbol(&dec, rans_exp_table_index(model, b));
          e1[b] =
              fxp_clamp_i32(exp_indices[b] + delta, PSY_EXP_INDEX_MIN, PSY_EXP_INDEX_MAX);
        }
      }
    }
    if (dec.overrun) {
      return false;
    }
  }

  for (int ch = 0; ch < n_ch; ch++) {
    int16_t *ch_q = &quant_out[ch * HQLC_FRAME_SAMPLES];

    for (int b = 0; b < PSY_N_BANDS; b++) {
      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];

      // Per-band table from alpha + activity of the already-decoded prev band
      HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_RANS_DEC);
      int abin = rans_alpha_bin(b, gain_code);
      if (ch == 1 && ms_flags && ms_flags[b]) {
        abin -= MS_RANS_ALPHA_SHIFT;
        if (abin < 0) {
          abin = 0;
        }
      }
      int act = rans_activity_bin(ch_q, b);
      int tidx = rans_table_idx(abin, act);
      const uint16_t *cf = rans_cf[tidx];
      HQLC_BENCH_END(HQLC_BENCH_DEC_RANS_DEC);

      for (int i = s; i < e; i++) {
        uint8_t sym = rans_dec_sym(&dec, cf);
        int mag = sym;

        if (sym >= RANS_MAX_SYM - 1) {
          // Unary prefix, bounded so a corrupt stream can't spin forever
          int nbits = 0;
          while (rans_dec_sign(&dec) == 0 && nbits < 24 && !dec.overrun) {
            nbits++;
          }
          int val = 1;
          for (int j = 0; j < nbits; j++) {
            val = (val << 1) | rans_dec_sign(&dec);
          }
          mag = (RANS_MAX_SYM - 1) + val - 1;
        }

        if (mag > 0) {
          uint8_t sign_val = rans_dec_sign(&dec);
          ch_q[i] = sign_val ? (int16_t)(-mag) : (int16_t)mag;
        } else {
          ch_q[i] = 0;
        }
      }
    }
  }

  return !dec.overrun;
}
