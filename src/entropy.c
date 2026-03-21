#include "entropy.h"
#include "entropy_tables.h"
#include "fxp.h"
#include "hqlc.h"
#include "psy.h"
#include "quant.h"

#include <string.h>

// Byte-streaming rANS renormalization threshold.
// Before encoding a symbol with frequency f, we emit bytes until
// state < f << RANS_RENORM_SHIFT. This keeps state in [RANS_L, RANS_L*256)
// after byte emission, preventing uint32 overflow in the encode step.
// Derived from: RANS_L_BITS + BYTE_BITS - RANS_M_BITS = 16 + 8 - 10 = 14
#define RANS_RENORM_SHIFT    14
#define RANS_BYTE_BITS       8
#define RANS_COST_ONE_BIT_Q8 (1 << 8)
// Sign coding uses a fixed equi-probable {M/2, M/2} distribution.
// All sign encode/decode is done via shifts and masks, no table lookups.
#define RANS_SIGN_FREQ         (RANS_M / 2)
#define RANS_SIGN_SLOT_SHIFT   (RANS_M_BITS - 1)
#define RANS_SIGN_SLOT_MASK    (RANS_SIGN_FREQ - 1)
#define RANS_SIGN_RENORM_UPPER ((uint32_t)RANS_SIGN_FREQ << RANS_RENORM_SHIFT)
#define RANS_MAX_COST_Q8       (RANS_M_BITS * RANS_COST_ONE_BIT_Q8)

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

void rans_enc_init(hqlc_rans_enc *enc, uint8_t *buf, size_t cap) {
  enc->state = RANS_L;
  enc->buf = buf;
  enc->cap = cap;
  enc->pos = cap; // write cursor starts at end
}

void rans_enc_put(hqlc_rans_enc *enc,
                  uint8_t sym,
                  const uint16_t *freq,
                  const uint16_t *cf,
                  const uint32_t *rcp) {
  uint16_t f = freq[sym];
  uint32_t upper = (uint32_t)f << RANS_RENORM_SHIFT;
  uint32_t state = enc->state;

  while (state >= upper) {
    enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
    state >>= 8;
  }

  uint32_t q = (uint32_t)(((uint64_t)state * rcp[sym]) >> 32);
  uint32_t r = state - q * f;
  if (r >= f) {
    q++;
    r -= f;
  }

  enc->state = (q << RANS_M_BITS) + r + cf[sym];
}

// Specialized sign encode for the fixed {RANS_M/2, RANS_M/2} split used by the sign bit.
// This path is fully shift/mask based, so it avoids general freq/cf/rcp table lookups.
static inline void rans_enc_sign(hqlc_rans_enc *enc, uint8_t sign) {
  uint32_t state = enc->state;

  // Renorm threshold for the fixed sign distribution {RANS_M/2, RANS_M/2}.
  if (state >= RANS_SIGN_RENORM_UPPER) {
    enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
    state >>= RANS_BYTE_BITS;
    if (state >= RANS_SIGN_RENORM_UPPER) {
      enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
      state >>= RANS_BYTE_BITS;
    }
  }

  // For the fixed {RANS_SIGN_FREQ, RANS_SIGN_FREQ} split:
  //   q = state / RANS_SIGN_FREQ = state >> RANS_SIGN_SLOT_SHIFT
  //   r = state % RANS_SIGN_FREQ = state & RANS_SIGN_SLOT_MASK
  enc->state = ((state >> RANS_SIGN_SLOT_SHIFT) << RANS_M_BITS) +
               (state & RANS_SIGN_SLOT_MASK) + (sign ? RANS_SIGN_FREQ : 0u);
}

// Inlined general encode for mag / overflow
static inline void rans_enc_sym(hqlc_rans_enc *enc,
                                uint8_t sym,
                                const uint16_t *freq,
                                const uint16_t *cf,
                                const uint32_t *rcp) {
  uint16_t f = freq[sym];
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

  uint32_t q = (uint32_t)(((uint64_t)state * rcp[sym]) >> 32);
  uint32_t r = state - q * f;
  if (r >= f) {
    q++;
    r -= f;
  }

  enc->state = (q << RANS_M_BITS) + r + cf[sym];
}

size_t rans_enc_flush(hqlc_rans_enc *enc) {
  // Flush the encoder state to the buffer, writes 4 bytes
  uint32_t state = enc->state;
  enc->buf[--enc->pos] = (uint8_t)(state & 0xFF);
  state >>= 8;
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

void rans_dec_init(hqlc_rans_dec *dec, const uint8_t *buf, size_t len) {
  dec->buf = buf;
  dec->len = len;
  dec->pos = 0;
  dec->state = 0;
  for (int i = 0; i < 4 && dec->pos < len; i++) {
    dec->state = (dec->state << 8) | buf[dec->pos++];
  }
}

uint8_t
rans_dec_get(hqlc_rans_dec *dec, const uint16_t *freq, const uint16_t *cf, int nsym) {
  uint32_t slot = dec->state & (RANS_M - 1);

  int lo = 0, hi = nsym;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (cf[mid] <= slot) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  uint8_t s = (uint8_t)lo;

  uint16_t f = freq[s];
  dec->state = (uint32_t)f * (dec->state >> RANS_M_BITS) + slot - cf[s];

  while (dec->state < RANS_L && dec->pos < dec->len) {
    dec->state = (dec->state << 8) | dec->buf[dec->pos++];
  }

  return s;
}

// Specialized sign decode for the fixed {RANS_SIGN_FREQ, RANS_SIGN_FREQ} split.
static inline uint8_t rans_dec_sign(hqlc_rans_dec *dec) {
  uint32_t state = dec->state;
  uint32_t slot = state & (RANS_M - 1);
  uint8_t s =
      (uint8_t)(slot >> RANS_SIGN_SLOT_SHIFT); // 0 if slot < RANS_SIGN_FREQ, 1 otherwise

  // Inverse update for the fixed sign distribution:
  //   state = RANS_SIGN_FREQ * floor(state / RANS_M) + (slot % RANS_SIGN_FREQ)
  dec->state =
      ((state >> RANS_M_BITS) << RANS_SIGN_SLOT_SHIFT) + (slot & RANS_SIGN_SLOT_MASK);

  // Renorm (unrolled, at most 2 bytes)
  if (dec->state < RANS_L) {
    dec->state = (dec->state << 8) | dec->buf[dec->pos++];
    if (dec->state < RANS_L) {
      dec->state = (dec->state << 8) | dec->buf[dec->pos++];
    }
  }
  return s;
}

static inline uint8_t
rans_dec_sym(hqlc_rans_dec *dec, const uint16_t *freq, const uint16_t *cf) {
  uint32_t state = dec->state;
  uint32_t slot = state & (RANS_M - 1);

  // Linear scan — symbol 0 is most probable (Laplacian), so this
  // typically terminates in 1-2 iterations vs 4 for binary search.
  // cf[RANS_MAX_SYM] == RANS_M > any slot, so loop always terminates.
  int s = 0;
  while (cf[s + 1] <= slot) {
    s++;
  }

  uint16_t f = freq[s];
  dec->state = (uint32_t)f * (state >> RANS_M_BITS) + slot - cf[s];

  // Renorm
  if (dec->state < RANS_L) {
    dec->state = (dec->state << 8) | dec->buf[dec->pos++];
    if (dec->state < RANS_L) {
      dec->state = (dec->state << 8) | dec->buf[dec->pos++];
    }
  }
  return (uint8_t)s;
}

// Estimate coding cost of a symbol at a frequency, essentially a log2 approximation with
// a LUT
static inline int16_t rans_freq_cost_q8(uint16_t freq_val) {
  if (freq_val == 0) {
    return RANS_MAX_COST_Q8; // impossible symbol: clamp to the max finite cost
  }
  int n = 31 - __builtin_clz(freq_val);
  int idx;
  if (n >= 7) {
    idx = (freq_val >> (n - 7)) & 0x7F;
  } else {
    idx = (freq_val << (7 - n)) & 0x7F;
  }
  // Ideal symbol cost is log2(RANS_M / freq), in bits.
  // Since RANS_M = 2^RANS_M_BITS and this function returns Q8 fractional bits:
  //   cost_q8 = (RANS_M_BITS - log2(freq)) * RANS_COST_ONE_BIT_Q8
  // We approximate log2(freq) in Q8 as:
  //   n * RANS_COST_ONE_BIT_Q8 + log2_frac_q8[idx],
  // where n = floor(log2(freq)).
  return (int16_t)(RANS_MAX_COST_Q8 - (n * RANS_COST_ONE_BIT_Q8 + log2_frac_q8[idx]));
}

// Sign coding uses the fixed uniform split {RANS_M/2, RANS_M/2} over total mass RANS_M.
// Specialized rans_enc_sign / rans_dec_sign use shift-based arithmetic directly, so no
// freq/cf/rcp tables are needed at runtime.
void rans_build_band_tables(int gain_code, hqlc_rans_band_tables tables[RANS_N_PAIRS]) {
  // inv_gain = 2^((QUANT_GAIN_BIAS - gain_code) / 8)
  // alpha[b] = K * BW[b] * inv_gain
  int gc = gain_code;

  // Clamp to range where LUT alpha stays in bounds
  // the 21 and 67 are derived from the gain_code range
  gc = fxp_clamp_i32(gc, 21, 67);

  int neg_E = QUANT_GAIN_BIAS - gc;
  int int_part = (neg_E >= 0) ? (neg_E / 8) : ((neg_E - 7) / 8);
  int frac = neg_E - 8 * int_part;
  int32_t inv_gain_m_q30 = quant_pow2_eighth_q30[frac];

  // shift to go from Q16 * Q30 = Q46 down to Q16. That'd be 30, but inv_gain has extra
  // 2^int_part factor, so shift = 30 - int_part.
  int shift = 30 - int_part;

  for (int pi = 0; pi < RANS_N_PAIRS; pi++) {
    int b0 = 2 * pi;
    int b1 = 2 * pi + 1;

    // Per-band alpha in Q16
    uint32_t a0_q16, a1_q16;
    if (shift > 0) {
      a0_q16 = (uint32_t)(((int64_t)rans_k_bw_q16[b0] * inv_gain_m_q30) >> shift);
      a1_q16 = (uint32_t)(((int64_t)rans_k_bw_q16[b1] * inv_gain_m_q30) >> shift);
    } else {
      a0_q16 = (uint32_t)(((int64_t)rans_k_bw_q16[b0] * inv_gain_m_q30) << (-shift));
      a1_q16 = (uint32_t)(((int64_t)rans_k_bw_q16[b1] * inv_gain_m_q30) << (-shift));
    }

    // Pair alpha = BW-weighted mean of band alphas
    // pair_alpha = (a0 * bw0 + a1 * bw1) / (bw0 + bw1)
    // All in Q16, bw values are small integers
    int bw0 = psy_band_edges[b0 + 1] - psy_band_edges[b0];
    int bw1 = psy_band_edges[b1 + 1] - psy_band_edges[b1];
    uint32_t pair_alpha_q16 =
        (uint32_t)((uint64_t)a0_q16 * bw0 + (uint64_t)a1_q16 * bw1) /
        (uint32_t)(bw0 + bw1);

    // Binary search alpha_edges for bin index
    int bin = 0;
    {
      int lo = 0, hi = RANS_LUT_NBINS;
      while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (rans_lut_alpha_edges_q16[mid] <= pair_alpha_q16) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      bin = lo;
      if (bin >= RANS_LUT_NBINS) {
        bin = RANS_LUT_NBINS - 1;
      }
    }

    // One table per pair (paired bands share the same distribution)
    hqlc_rans_band_tables *t = &tables[pi];

    // Copy freq + precomputed rcp from LUT, build cf[], compute cost_q8[]
    t->cf[0] = 0;
    for (int s = 0; s < RANS_MAX_SYM; s++) {
      t->freq[s] = rans_lut_freq[bin][s];
      t->cf[s + 1] = t->cf[s] + t->freq[s];
      t->cost_q8[s] = rans_freq_cost_q8(t->freq[s]);
      t->rcp[s] = rans_lut_rcp[bin][s];
    }
  }
}

int32_t rans_coeff_cost_q8(const hqlc_rans_band_tables *tbl, int16_t value) {
  int mag = (value < 0) ? -value : value;
  int32_t c;

  if (mag < RANS_MAX_SYM - 1) {
    c = tbl->cost_q8[mag];
  } else {
    c = tbl->cost_q8[RANS_MAX_SYM - 1]; // ESC symbol
    // EG(0) for the overflow value
    int overflow = mag - (RANS_MAX_SYM - 1);
    int nbits = 0;
    {
      int tmp = overflow + 1;
      while (tmp > 1) {
        tmp >>= 1;
        nbits++;
      }
    }
    c += (2 * nbits + 1) * RANS_COST_ONE_BIT_Q8; // EG(0): nbits+1 unary + nbits binary
  }
  if (value != 0) {
    c += RANS_COST_ONE_BIT_Q8; // 1 sign bit
  }
  return c;
}

size_t rans_encode_coeffs(const int16_t *quant,
                          const uint8_t *nf_mask,
                          int n_ch,
                          const hqlc_rans_band_tables *tables,
                          uint8_t *out,
                          size_t out_cap) {
  hqlc_rans_enc enc;
  rans_enc_init(&enc, out, out_cap);

  // Due to nature of rANS, we encode in reverse order. Backward channels, bands and bins.
  // For each coefficient: put sign, then overflow, then magnitude (reversed).
  // Decoder reads forward naturally
  for (int ch = n_ch - 1; ch >= 0; ch--) {
    for (int b = PSY_N_BANDS - 1; b >= 0; b--) {
      if (nf_mask[ch * PSY_N_BANDS + b]) {
        continue;
      }

      const hqlc_rans_band_tables *tbl = &tables[b >> 1];
      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];

      for (int i = e - 1; i >= s; i--) {
        int16_t v = quant[ch * HQLC_FRAME_SAMPLES + i];
        int mag = (v < 0) ? -v : v;
        uint8_t sym =
            (mag < RANS_MAX_SYM - 1) ? (uint8_t)mag : (uint8_t)(RANS_MAX_SYM - 1);

        // Sign (if nonzero, put last so it's decoded first)
        if (v != 0) {
          rans_enc_sign(&enc, (v > 0) ? 0 : 1);
        }

        // Overflow: EG(0) coding via sign channel
        if (mag >= RANS_MAX_SYM - 1) {
          int overflow = mag - (RANS_MAX_SYM - 1);
          int nbits = 0;
          {
            int tmp = overflow + 1;
            while (tmp > 1) {
              tmp >>= 1;
              nbits++;
            }
          }
          int val = overflow + 1;

          for (int bit_idx = 0; bit_idx < nbits; bit_idx++) {
            rans_enc_sign(&enc, (val >> bit_idx) & 1);
          }
          rans_enc_sign(&enc, 1);
          for (int j = 0; j < nbits; j++) {
            rans_enc_sign(&enc, 0);
          }
        }

        // Magnitude symbol
        rans_enc_sym(&enc, sym, tbl->freq, tbl->cf, tbl->rcp);
      }
    }
  }

  return rans_enc_flush(&enc);
}

bool rans_decode_coeffs(const uint8_t *data,
                        size_t len,
                        int16_t *quant_out,
                        const uint8_t *nf_mask,
                        int n_ch,
                        const hqlc_rans_band_tables *tables) {
  if (len == 0) {
    memset(quant_out, 0, (size_t)n_ch * HQLC_FRAME_SAMPLES * sizeof(int16_t));
    return true;
  }

  hqlc_rans_dec dec;
  rans_dec_init(&dec, data, len);

  for (int ch = 0; ch < n_ch; ch++) {
    for (int b = 0; b < PSY_N_BANDS; b++) {
      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];

      if (nf_mask[ch * PSY_N_BANDS + b]) {
        for (int i = s; i < e; i++) {
          quant_out[ch * HQLC_FRAME_SAMPLES + i] = 0;
        }
        continue;
      }

      const hqlc_rans_band_tables *tbl = &tables[b >> 1];

      for (int i = s; i < e; i++) {
        // Decode magnitude
        // We use a linear scan, as naturally its ordered by frequency
        // Quite convinient, isnt it?
        uint8_t sym = rans_dec_sym(&dec, tbl->freq, tbl->cf);
        int mag = sym;

        // If ESC (15), decode overflow via EG(0)
        if (sym >= RANS_MAX_SYM - 1) {
          int nbits = 0;
          while (rans_dec_sign(&dec) == 0) {
            nbits++;
          }
          int val = 1;
          for (int j = 0; j < nbits; j++) {
            val = (val << 1) | rans_dec_sign(&dec);
          }
          mag = (RANS_MAX_SYM - 1) + val - 1;
        }

        // Decode sign (if nonzero)
        if (mag > 0) {
          uint8_t sign_val = rans_dec_sign(&dec);
          quant_out[ch * HQLC_FRAME_SAMPLES + i] =
              sign_val ? (int16_t)(-mag) : (int16_t)mag;
        } else {
          quant_out[ch * HQLC_FRAME_SAMPLES + i] = 0;
        }
      }
    }
  }

  return true;
}
