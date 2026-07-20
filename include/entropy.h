#ifndef HQLC_ENTROPY_H
#define HQLC_ENTROPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// M=1024 (10-bit), provides enough precision without blowing up the LUTs
#define RANS_M       1024
#define RANS_M_BITS  10
#define RANS_L       (1u << 16)
#define RANS_MAX_SYM 16 // 0-14 magnitudes + 15 ESC
#define RANS_N_PAIRS 10

// rANS probability tables: 12 alpha x 4 activity = 48 tables
#define RANS_ALPHA_NBINS    12
#define RANS_ACT_NBINS      4
#define RANS_NTABLES        48
#define RANS_ALPHA_LO_Q8    (-1271)
#define RANS_ALPHA_RANGE_Q8 2647

// 48 rANS probability tables in cumulative form, freq[s] = cf[s + 1] - cf[s]
extern const uint16_t rans_cf[RANS_NTABLES][RANS_MAX_SYM + 1];
extern const uint32_t rans_rcp[RANS_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_cost_q8[RANS_NTABLES][RANS_MAX_SYM];
extern const int16_t rans_log2_sigma_q8[RANS_N_PAIRS];

/**
 * @brief Convert a signed value to zigzag form.
 *
 * Negative values map to odd integers and non-negative values map to even
 * integers.
 *
 * @param v Signed value to encode.
 * @return Zigzag-encoded unsigned value.
 */
static inline uint32_t zigzag_enc(int32_t v) {
  return (v < 0) ? (uint32_t)((-v << 1) - 1) : (uint32_t)(v << 1);
}

/**
 * @brief Convert a zigzag value back to signed form.
 *
 * @param u Zigzag-encoded unsigned value.
 * @return Decoded signed value.
 */
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

/**
 * @brief Initialize an MSB-first bit writer.
 *
 * @param w Bit writer state to initialize.
 * @param buf Output buffer.
 * @param cap Output buffer capacity in bytes.
 */
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
 * @brief Write bits to an MSB-first bitstream.
 *
 * Writes the low `n` bits of `val`, most-significant bit first.
 *
 * @param w Bit writer state.
 * @param val Value containing the bits to write.
 * @param n Number of low bits to write.
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
 * @brief Write a Rice-coded unsigned value.
 *
 * Encodes `val` as `unary(val >> k)` followed by the `k`-bit remainder.
 *
 * @param w Bit writer state.
 * @param val Value to encode.
 * @param k Rice parameter.
 */
void bw_write_rice(hqlc_bitwriter *w, uint32_t val, int k);

/**
 * @brief Find the Rice parameter that minimizes coded size.
 *
 * @param values Signed values to analyze.
 * @param n Number of values.
 * @return Best Rice parameter in the range 0..6.
 */
int find_best_rice_k(const int32_t *values, int n);

/**
 * @brief Pad with zero bits to the next byte boundary.
 *
 * @param w Bit writer state.
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
 * @brief Return the number of completed bytes written.
 *
 * Call this after bw_flush() if a partial byte may be pending.
 *
 * @param w Bit writer state.
 * @return Number of completed bytes in the output buffer.
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

/**
 * @brief Initialize an MSB-first bit reader.
 *
 * @param r Bit reader state to initialize.
 * @param buf Input buffer.
 * @param len Input buffer length in bytes.
 */
static inline void br_init(hqlc_bitreader *r, const uint8_t *buf, size_t len) {
  r->buf = buf;
  r->len = len;
  r->pos = 0;
  r->rem = 8;
}

/**
 * @brief Read bits from an MSB-first bitstream.
 *
 * Reads `n` bits and returns them right-aligned. Reads past the end of the
 * buffer are padded with zero bits.
 *
 * @param r Bit reader state.
 * @param n Number of bits to read, from 1 to 25.
 * @return Right-aligned bits read from the stream.
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
 * @brief Read a Rice-coded unsigned value.
 *
 * Decodes `unary(q)` followed by a `k`-bit remainder.
 *
 * @param r Bit reader state.
 * @param k Rice parameter.
 * @return Decoded value.
 */
uint32_t br_read_rice(hqlc_bitreader *r, int k);

/**
 * @brief Return the number of bits consumed.
 *
 * @param r Bit reader state.
 * @return Total bits read from the stream.
 */
static inline size_t br_bits(const hqlc_bitreader *r) {
  return r->pos * 8 + (8 - r->rem);
}

// rANS encoder state. Writes backward from the buffer end (reversed in decoder)
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
  size_t pos;   // read cursor
  bool overrun; // set if a read ran past the end (corrupt/truncated input)
} hqlc_rans_dec;

/**
 * @brief Initialize an rANS encoder.
 *
 * @param enc Encoder state to initialize.
 * @param buf Output buffer.
 * @param cap Output buffer capacity in bytes.
 */
void rans_enc_init(hqlc_rans_enc *enc, uint8_t *buf, size_t cap);

/**
 * @brief Flush an rANS encoder.
 *
 * @param enc Encoder state to flush.
 * @return Number of encoded bytes written.
 */
size_t rans_enc_flush(hqlc_rans_enc *enc);

/**
 * @brief Initialize an rANS decoder.
 *
 * @param dec Decoder state to initialize.
 * @param buf Encoded input buffer.
 * @param len Encoded input length in bytes.
 */
void rans_dec_init(hqlc_rans_dec *dec, const uint8_t *buf, size_t len);

/**
 * @brief Map a Q8 log2(alpha) value to an rANS alpha bin.
 *
 * The linear mapping is clamped to the valid 0..11 bin range.
 *
 * @param log2_alpha_q8 log2(alpha) in Q8 format.
 * @return Alpha bin in the range 0..11.
 */
static inline int rans_alpha_bin_from_la(int32_t log2_alpha_q8) {
  int32_t bin =
      (log2_alpha_q8 - RANS_ALPHA_LO_Q8) * RANS_ALPHA_NBINS / RANS_ALPHA_RANGE_Q8;
  if (bin < 0) {
    bin = 0;
  }
  if (bin >= RANS_ALPHA_NBINS) {
    bin = RANS_ALPHA_NBINS - 1;
  }
  return (int)bin;
}

/**
 * @brief Map a band's nonzero fraction to an activity bin.
 *
 * Thresholds on `nz / w` are: less than 0.1, less than 0.3, less than 0.6,
 * and 0.6 or greater.
 *
 * @param nz Number of nonzero values.
 * @param w Band width.
 * @return Activity bin in the range 0..3.
 */
static inline int rans_activity_from(int nz, int w) {
  int nz10 = nz * 10;
  if (nz10 < w) {
    return 0;
  }
  if (nz10 < 3 * w) {
    return 1;
  }
  if (nz10 < 6 * w) {
    return 2;
  }
  return 3;
}

/**
 * @brief Build a probability table index from alpha and activity bins.
 *
 * @param alpha_bin Alpha bin.
 * @param activity Activity bin.
 * @return Clamped rANS table index.
 */
static inline int rans_table_idx(int alpha_bin, int activity) {
  int tidx = alpha_bin * RANS_ACT_NBINS + activity;
  if (tidx < 0) {
    tidx = 0;
  }
  if (tidx >= RANS_NTABLES) {
    tidx = RANS_NTABLES - 1;
  }
  return tidx;
}

/**
 * @brief Count the bits needed for an EG(0) escape body.
 *
 * @param overflow Overflow value to encode.
 * @return Number of EG(0) body bits.
 */
static inline int rans_eg0_nbits(int overflow) {
  int nbits = 0;
  for (int tmp = overflow + 1; tmp > 1; tmp >>= 1) {
    nbits++;
  }
  return nbits;
}

/**
 * @brief Compute the alpha bin for one band.
 *
 * @param band Band index.
 * @param gain_code Quantizer gain code.
 * @return Alpha bin in the range 0..11.
 */
int rans_alpha_bin(int band, int gain_code);

/**
 * @brief Compute the activity bin from the previous band's nonzero fraction.
 *
 * @param quant Quantized coefficients.
 * @param band Band index.
 * @return Activity bin in the range 0..3.
 */
int rans_activity_bin(const int16_t *quant, int band);

/**
 * @brief Encode quantized coefficients with rANS.
 *
 * @param quant Quantized coefficients to encode.
 * @param n_ch Number of channels.
 * @param gain_code Quantizer gain code.
 * @param out Output buffer.
 * @param out_cap Output buffer capacity in bytes.
 * @return Number of encoded bytes written.
 */
size_t rans_encode_coeffs(
    const int16_t *quant, int n_ch, int gain_code, uint8_t *out, size_t out_cap);

/**
 * @brief Decode quantized coefficients from an rANS byte stream.
 *
 * @param data Encoded byte stream.
 * @param len Encoded byte stream length in bytes.
 * @param quant_out Destination for decoded coefficients.
 * @param n_ch Number of channels.
 * @param gain_code Quantizer gain code.
 * @return True on success, or false if the stream is corrupt or truncated.
 */
bool rans_decode_coeffs(
    const uint8_t *data, size_t len, int16_t *quant_out, int n_ch, int gain_code);

#ifdef __cplusplus
}
#endif

#endif // HQLC_ENTROPY_H
