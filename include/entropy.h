#ifndef HQLC_ENTROPY_H
#define HQLC_ENTROPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// M=1024 (10-bit), provides enough precision without blowing up the LUTs
#define RANS_M         1024
#define RANS_M_BITS    10
#define RANS_L         (1u << 16)
#define RANS_MAX_SYM   16 // 0-14 magnitudes + 15 ESC
#define RANS_N_PAIRS   10

// Division-pipeline: 12 alpha x 4 activity = 48 tables
#define RANS_ALPHA_NBINS   12
#define RANS_ACT_NBINS     4
#define RANS_DIV_NTABLES   48
#define RANS_ALPHA_LO_Q8   (-1271)
#define RANS_ALPHA_HI_Q8   1375
#define RANS_ALPHA_RANGE_Q8 2647

// 48 division-pipeline rANS tables
extern const uint16_t rans_div_freq[RANS_DIV_NTABLES][RANS_MAX_SYM];
extern const uint32_t rans_div_rcp[RANS_DIV_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_div_cost_q8[RANS_DIV_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_log2_sigma_q8[RANS_N_PAIRS];

// Zigzag: map signed to unsigned (negatives → odd, non-negatives → even).
static inline uint32_t zigzag_enc(int32_t v) {
  return (v < 0) ? (uint32_t)((-v << 1) - 1) : (uint32_t)(v << 1);
}

// Inverse zigzag: unsigned back to signed.
static inline int32_t zigzag_dec(uint32_t u) {
  return (u & 1) ? -(int32_t)((u + 1) >> 1) : (int32_t)(u >> 1);
}

// Struct for the bitwriter (MSB-first)
typedef struct {
  uint8_t *buf; /**< buffer for the bitstream */
  size_t cap;   /**< Buffer capacity in bytes */
  size_t pos;   /**< Completed bytes written */
  int free;     /**< Free bits in buf[pos]: 8 = empty, 0 = full */
} hqlc_bitwriter;

static inline void bw_init(hqlc_bitwriter *w, uint8_t *buf, size_t cap) {
  w->buf = buf;
  w->cap = cap;
  w->pos = 0;
  w->free = 8;
  if (cap > 0) {
    buf[0] = 0;
  }
}

// Write the low n bits of val, MSB-first.
static inline void bw_write(hqlc_bitwriter *w, uint32_t val, int n) {
  while (n > 0) {
    int take = (n < w->free) ? n : w->free;
    w->buf[w->pos] |= (uint8_t)((val >> (n - take)) << (w->free - take));
    w->free -= take;
    n -= take;
    if (w->free == 0) {
      w->pos++;
      if (w->pos < w->cap) {
        w->buf[w->pos] = 0;
      }
      w->free = 8;
    }
  }
}

// Write val with Rice coding: unary(val >> k) + k-bit remainder.
void bw_write_rice(hqlc_bitwriter *w, uint32_t val, int k);

// Find the Rice parameter k (0..6) that minimizes total coded size.
int find_best_rice_k(const int32_t *values, int n);

// Pad the current byte with zeros and advance to the next byte boundary.
static inline void bw_flush(hqlc_bitwriter *w) {
  if (w->free < 8) {
    w->pos++;
  }
  w->free = 8;
  if (w->pos < w->cap) {
    w->buf[w->pos] = 0;
  }
}

// Total bits written, including a partial current byte.
static inline size_t bw_bits(const hqlc_bitwriter *w) {
  return w->pos * 8 + (8 - w->free);
}

// Completed bytes written; call after bw_flush().
static inline size_t bw_bytes(const hqlc_bitwriter *w) {
  return w->pos;
}

// Bit reader state
typedef struct {
  const uint8_t *buf; /**< Pointer to the buffer to read from */
  size_t len;         /**< buffer length in bytes */
  size_t pos;         /**< current byte index */
  int rem;            /**< remaining bits in buf[pos]: 8 = full byte */
} hqlc_bitreader;

static inline void br_init(hqlc_bitreader *r, const uint8_t *buf, size_t len) {
  r->buf = buf;
  r->len = len;
  r->pos = 0;
  r->rem = 8;
}

// Read n bits (1..25), right-aligned. Pads with zero bits past end-of-buffer.
static inline uint32_t br_read(hqlc_bitreader *r, int n) {
  uint32_t val = 0;
  while (n > 0) {
    if (r->pos >= r->len) {
      return val << n; // pad with zeros on overread
    }
    int take = (n < r->rem) ? n : r->rem;
    val = (val << take) | ((r->buf[r->pos] >> (r->rem - take)) & ((1u << take) - 1));
    r->rem -= take;
    n -= take;
    if (r->rem == 0) {
      r->pos++;
      r->rem = 8;
    }
  }
  return val;
}

// Read a Rice-coded value: unary(q) + k-bit remainder.
uint32_t br_read_rice(hqlc_bitreader *r, int k);

// Total bits consumed.
static inline size_t br_bits(const hqlc_bitreader *r) {
  return r->pos * 8 + (8 - r->rem);
}

// rANS encoder state. Writes backward from the buffer end (rANS emits in
// reverse; the decoder reads forward).
typedef struct {
  uint32_t state; /**< 4 byte state of the encoder */
  uint8_t *buf;   /**< Output buffer */
  size_t cap;     /**< Buffer capacity in bytes */
  size_t pos;     /**< Write position, goes backwards from cap */
} hqlc_rans_enc;

// rANS decoder state
typedef struct {
  uint32_t state;
  const uint8_t *buf;
  size_t len;
  size_t pos;  // read cursor
  bool overrun; // set if a read ran past the end (corrupt/truncated input)
} hqlc_rans_dec;

// Per-band rANS tables (one of the 48 precomputed alpha x activity sets).
typedef struct {
  uint16_t freq[RANS_MAX_SYM];   // symbol frequencies
  uint16_t cf[RANS_MAX_SYM + 1]; // cumulative frequencies
  int16_t cost_q8[RANS_MAX_SYM]; // cost per symbol, Q8 bits
  uint32_t rcp[RANS_MAX_SYM];    // reciprocals for division-free encode
} hqlc_rans_band_tables;

// Initialize an rANS encoder writing into [buf, buf+cap).
void rans_enc_init(hqlc_rans_enc *enc, uint8_t *buf, size_t cap);

// Flush the encoder; returns the encoded byte count.
size_t rans_enc_flush(hqlc_rans_enc *enc);

// Initialize an rANS decoder reading from [buf, buf+len).
void rans_dec_init(hqlc_rans_dec *dec, const uint8_t *buf, size_t len);

// rANS symbol cost in Q8 fractional bits, from a symbol frequency.
int16_t rans_freq_cost_q8(uint16_t freq_val);

// Alpha bin [0..11] for a band, from its gain code and pair sigma.
int rans_alpha_bin(int band, int gain_code);

// Activity bin [0..3] from the previous band's nonzero fraction.
int rans_activity_bin(const int16_t *quant, int band);

// Estimated rANS cost of a quantized coefficient, in Q8 fractional bits.
int32_t rans_coeff_cost_q8(const hqlc_rans_band_tables *tbl, int16_t value);

// Encode quantized coefficients to a byte buffer; returns the byte count.
size_t rans_encode_coeffs(const int16_t *quant,
                          const uint8_t *nf_mask,
                          int n_ch,
                          int gain_code,
                          uint8_t *out,
                          size_t out_cap);

// Decode quantized coefficients from a byte buffer. Returns false if the
// stream was corrupt or truncated (a read ran past the end).
bool rans_decode_coeffs(const uint8_t *data,
                        size_t len,
                        int16_t *quant_out,
                        const uint8_t *nf_mask,
                        int n_ch,
                        int gain_code);

#ifdef __cplusplus
}
#endif

#endif // HQLC_ENTROPY_H
