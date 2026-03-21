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
#define RANS_LUT_NBINS 32
#define RANS_N_PAIRS   10

// Zigzag encoding, mapping signed integers to unsigned for variable-length coding
static inline uint32_t zigzag_enc(int32_t v) {
  // negative numbers are mapped to odd values, positive numbers to even values
  return (v < 0) ? (uint32_t)((-v << 1) - 1) : (uint32_t)(v << 1);
}

// inverse of zigzag encoding, mapping unsigned integers back to signed for
// variable-length coding
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

/**
 * @brief Write n bits from the lower n bits of val.
 *
 * @param w   Bit writer state
 * @param val Value whose lower n bits are written
 * @param n   Number of bits to write
 */
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

/**
 * @brief Write a value using Rice coding, unary(q) + k-bit remainder
 *
 * @param w   Bit writer state
 * @param val Value to encode
 * @param k   Rice parameter
 */
void bw_write_rice(hqlc_bitwriter *w, uint32_t val, int k);

// Finds an optimal rice K for a set of values
int find_best_rice_k(const int32_t *values, int n);

/**
 * @brief Pad current byte with zeros and advance to the next byte boundary
 *
 * @param w Bit writer state
 */
static inline void bw_flush(hqlc_bitwriter *w) {
  if (w->free < 8) {
    w->pos++;
  }
  w->free = 8;
  if (w->pos < w->cap) {
    w->buf[w->pos] = 0;
  }
}

/**
 * @brief Return total bits written, including any partial current byte
 *
 * @param w Bit writer state
 * @return Number of bits written
 */
static inline size_t bw_bits(const hqlc_bitwriter *w) {
  return w->pos * 8 + (8 - w->free);
}

/**
 * @brief Return completed bytes written. Call after bw_flush()
 *
 * @param w Bit writer state
 * @return Number of complete bytes written
 */
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

/**
 * @brief Read n bits (1..25), returned right-aligned
 *
 * Returns zero bits if the reader runs past the end of the buffer.
 *
 * @param r Bit reader state
 * @param n Number of bits to read
 * @return Read bits in the lower n bits of the return value
 */
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

/**
 * @brief Read a Rice-coded value: unary(q) + k-bit remainder
 *
 * @param r Bit reader state
 * @param k Rice parameter
 * @return Decoded value
 */
uint32_t br_read_rice(hqlc_bitreader *r, int k);

/**
 * @brief Return total bits consumed
 *
 * @param r Bit reader state
 * @return Number of bits consumed
 */
static inline size_t br_bits(const hqlc_bitreader *r) {
  return r->pos * 8 + (8 - r->rem);
}

// rANS encoder state
// Note: This writes backward from the end of the buffer, inverse of the decoder due to
// the nature of rANS encoding
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
  size_t pos; // read cursor
} hqlc_rans_dec;

// rANS band precomputed tables definition
typedef struct {
  uint16_t freq[RANS_MAX_SYM];   /**< Symbol frequencies */
  uint16_t cf[RANS_MAX_SYM + 1]; /**< Cumulative frequencies */
  int16_t cost_q8[RANS_MAX_SYM]; /**< Cost per symbol */
  uint32_t rcp[RANS_MAX_SYM];    /**< Reciprocals for division-free encode, fixed point
                                    optimization */
} hqlc_rans_band_tables;

/**
 * @brief Initialize an rANS encoder
 *
 * @param enc Encoder state to initialize
 * @param buf Output buffer
 * @param cap Buffer capacity in bytes
 */
void rans_enc_init(hqlc_rans_enc *enc, uint8_t *buf, size_t cap);

/**
 * @brief Encode one symbol into the rANS stream. Essentially unused, as the codec uses
 * specialized versions - but kept for API.
 *
 * @param enc  Encoder state
 * @param sym  Symbol to encode
 * @param freq Symbol frequency table
 * @param cf   Cumulative frequency table
 * @param rcp  Reciprocal table for division free encoding
 */
void rans_enc_put(hqlc_rans_enc *enc,
                  uint8_t sym,
                  const uint16_t *freq,
                  const uint16_t *cf,
                  const uint32_t *rcp);

/**
 * @brief Flush the rANS encoder and return the encoded byte count
 *
 * @param enc Encoder state
 * @return Number of bytes written
 */
size_t rans_enc_flush(hqlc_rans_enc *enc);

/**
 * @brief Initialize a rANS decoder
 *
 * @param dec Decoder state to initialize
 * @param buf Input buffer
 * @param len Buffer length in bytes
 */
void rans_dec_init(hqlc_rans_dec *dec, const uint8_t *buf, size_t len);

/**
 * @brief Decode one symbol from the rANS stream
 *
 * @param dec  Decoder state
 * @param freq Symbol frequency table
 * @param cf   Cumulative frequency table
 * @param nsym Number of symbols
 * @return Decoded symbol
 */
uint8_t
rans_dec_get(hqlc_rans_dec *dec, const uint16_t *freq, const uint16_t *cf, int nsym);

/**
 * @brief Build per-band rANS frequency tables for a given gain code
 *
 * @param gain_code Global gain code
 * @param tables    Output array of RANS_N_PAIRS band tables
 */
void rans_build_band_tables(int gain_code, hqlc_rans_band_tables tables[RANS_N_PAIRS]);

/**
 * @brief Estimate the rANS coding cost of a quantized coefficient in Q8 bits
 *
 * @param tbl   Band table for this coefficient's band
 * @param value Quantized coefficient
 * @return Estimated cost in Q8 fractional bits
 */
int32_t rans_coeff_cost_q8(const hqlc_rans_band_tables *tbl, int16_t value);

/**
 * @brief Encode quantized spectral coefficients to a byte buffer
 *
 * @param quant   Quantized coeffs
 * @param nf_mask Noise-fill mask per band
 * @param n_ch    Number of channels
 * @param tables  Per-band rANS tables
 * @param out     Output byte buffer
 * @param out_cap Output buffer capacity in bytes
 * @return Number of bytes written
 */
size_t rans_encode_coeffs(const int16_t *quant,
                          const uint8_t *nf_mask,
                          int n_ch,
                          const hqlc_rans_band_tables *tables,
                          uint8_t *out,
                          size_t out_cap);

/**
 * @brief Decode quantized spectral coefficients from a byte buffer
 *
 * @param data     Input byte buffer
 * @param len      Input buffer length in bytes
 * @param quant_out Output quantized coefficients
 * @param nf_mask  Noise-fill mask per band
 * @param n_ch     Number of channels
 * @param tables   Per-band rANS tables
 * @return true on success, false on error
 */
bool rans_decode_coeffs(const uint8_t *data,
                        size_t len,
                        int16_t *quant_out,
                        const uint8_t *nf_mask,
                        int n_ch,
                        const hqlc_rans_band_tables *tables);

#ifdef __cplusplus
}
#endif

#endif // HQLC_ENTROPY_H
