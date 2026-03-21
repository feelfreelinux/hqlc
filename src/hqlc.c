#include "hqlc.h"

#include <string.h>

#include "entropy.h"
#include "entropy_tables.h"
#include "fxp.h"
#include "mdct.h"
#include "pcm.h"
#include "psy.h"
#include "quant.h"
#include "tns.h"

#include "hqlc_bench.h"

#ifdef HQLC_BENCH
hqlc_bench_ctx *hqlc_bench = NULL;
#endif

// Encoder scratch layout. Holds all temporary data needed across stages
// It is allocated by the caller and passed to the encoder functions
typedef struct {
  // Spectral coefficients (Q31)
  int32_t spec_q31[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];

  // Per-band exponent indices
  int32_t exp_indices[HQLC_MAX_CHANNELS * PSY_N_BANDS];
  // Quantized coefficients
  int16_t quant[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];
  // noise fill mask
  uint8_t nf_mask[HQLC_MAX_CHANNELS * PSY_N_BANDS];

  // rANS band tables
  hqlc_rans_band_tables rans_tables[RANS_N_PAIRS];

  // Per-stage temporaries (only one active at a time)
  union {
    struct {
      // MDCT scratch buffer
      int64_t mdct_work[MDCT_SCRATCH_BYTES / sizeof(int64_t)];
      // single ch band energy, reused per iter
      uint64_t band_energy[PSY_N_BANDS];
      uint32_t band_peak[PSY_N_BANDS];
      int32_t smr_q4[PSY_N_BANDS];
    } analysis;
    struct {
      // Temporary buffer for the entropy coded bits
      uint8_t rans_tmp[HQLC_MAX_FRAME_BYTES];
    } coding;
  } stage;
} hqlc_enc_scratch;

// Decoder scratch layout
typedef struct {
  // Received quantized coeffs
  int16_t quant_buf[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];
  // Exponent indices for each band
  int32_t exp_indices[HQLC_MAX_CHANNELS * PSY_N_BANDS];

  // Noise fill mask
  uint8_t nf_mask[HQLC_MAX_CHANNELS * PSY_N_BANDS];

  // Per-stage temporaries
  union {
    struct {
      // RANS band tables
      hqlc_rans_band_tables tables[RANS_N_PAIRS];
    } entropy;
    struct {
      union {
        int64_t dequant_tmp[HQLC_FRAME_SAMPLES];
        int32_t windowed[HQLC_BLOCK_SAMPLES];
      };
      int32_t spec_q31[HQLC_FRAME_SAMPLES];
      int64_t mdct_work[MDCT_SCRATCH_BYTES / sizeof(int64_t)];
    } synthesis;
  } stage;
} hqlc_dec_scratch;

// Declaration of the actual encode struct, internal - public header just defines a fwd
// decl
struct hqlc_encoder {
  uint8_t channels;
  uint32_t sample_rate;
  hqlc_mode mode;
  uint32_t bitrate;
  int gain_code;
  // Previous frame's raw PCM (max stereo 24-bit, zero-init = silence)
  uint8_t prev_pcm[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS * 3];
  // RC state (MODE_RC only)
  int prev_gain_code;
  int32_t ema_gain_q8; // EMA of gain_code in Q8 fixed-point
  int prev_side_bits;
  int32_t res_bits;
  int cached_tbl_gc; // gain code whose rANS tables are cached in scratch
};

struct hqlc_decoder {
  uint8_t channels;
  uint32_t sample_rate;
  uint32_t frame_idx; // NF seed requires frame index
  mdct_ola_state ola[HQLC_MAX_CHANNELS];
};

size_t hqlc_encoder_size(void) {
  return sizeof(hqlc_encoder);
}
size_t hqlc_encoder_scratch_size(void) {
  return sizeof(hqlc_enc_scratch);
}

size_t hqlc_decoder_size(void) {
  return sizeof(hqlc_decoder);
}
size_t hqlc_decoder_scratch_size(void) {
  return sizeof(hqlc_dec_scratch);
}

hqlc_error hqlc_encoder_init(hqlc_encoder *enc, const hqlc_encoder_config *cfg) {
  if (!enc || !cfg) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (cfg->sample_rate != HQLC_SAMPLE_RATE) {
    return HQLC_ERR_UNSUPPORTED_RATE;
  }
  if (cfg->channels < 1 || cfg->channels > HQLC_MAX_CHANNELS) {
    return HQLC_ERR_UNSUPPORTED_CHANNELS;
  }

  memset(enc, 0, sizeof(*enc));

  switch (cfg->mode) {
  case HQLC_MODE_RC:
    if (cfg->bitrate == 0) {
      return HQLC_ERR_INVALID_ARG;
    }
    enc->bitrate = cfg->bitrate;
    enc->prev_gain_code = quant_gain_encode(0.5f); // -6dB: conservative startup gain
    enc->ema_gain_q8 = enc->prev_gain_code << 8;
    enc->prev_side_bits = 150; // ~19 bytes: typical side info for first probe
    enc->res_bits = 0;
    enc->cached_tbl_gc = -1;
    break;
  case HQLC_MODE_FIXED:
    if (cfg->gain <= 0.0f) {
      return HQLC_ERR_INVALID_ARG;
    }
    enc->gain_code = quant_gain_encode(cfg->gain);
    enc->cached_tbl_gc = -1;
    break;
  default:
    return HQLC_ERR_INVALID_ARG;
  }

  enc->channels = cfg->channels;
  enc->sample_rate = cfg->sample_rate;
  enc->mode = cfg->mode;
  return HQLC_OK;
}

// log2(x) in Q8 for x > 0, using a lookup table for fractional parts
static int32_t log2_q8_u32(uint32_t x) {
  if (x == 0) {
    return 0;
  }
  int msb = 31 - __builtin_clz(x);
  int frac_idx;
  if (msb >= 7) {
    frac_idx = (int)((x >> (msb - 7)) & 0x7F);
  } else {
    frac_idx = (int)((x << (7 - msb)) & 0x7F);
  }
  return msb * 256 + (int32_t)log2_frac_q8[frac_idx];
}

// Estimate how many bits a frame would cost at a given gain_code, without
// actually writing the quantized output. This mirrors the quant_forward logic
// (same exponent decomposition, same inv_step mantissa, same shift paths) but
// skips the deadzone test and sign — it only needs approximate magnitudes to
// sum rANS symbol costs.
//
// The three shift paths handle different numeric ranges:
//   total_shift >= 64:  all coefficients quantize to zero (skip)
//   total_shift >= 32:  common case, only high 32 bits of the product needed
//   total_shift >  0:   full 64-bit product (uncommon)
//   total_shift <= 0:   saturation (extremely rare)
static int probe_frame_bits(const int32_t *spec_q31,
                            const int *loss_bits,
                            const int32_t *exp_indices,
                            const uint8_t *nf_mask,
                            int gain_code,
                            int n_ch,
                            const hqlc_rans_band_tables *tables) {
  int32_t total_q8 = 0; // accumulated cost in Q8 fractional bits

  for (int ch = 0; ch < n_ch; ch++) {
    const int32_t *ch_spec = &spec_q31[ch * HQLC_FRAME_SAMPLES];
    const int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];
    const uint8_t *ch_nf = &nf_mask[ch * PSY_N_BANDS];
    int lb = loss_bits[ch];

    HQLC_BENCH_BEGIN();
    for (int b = 0; b < PSY_N_BANDS; b++) {
      if (ch_nf[b]) {
        continue; // noise-filled bands cost zero coefficient bits
      }

      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];

      // Compute inv_step mantissa and shift — same decomposition as quant_forward
      int E = 2 * (int)ch_exp[b] - gain_code - QUANT_EXP_OFFSET;
      int neg_E = -E;
      int int_part = (neg_E >= 0) ? (neg_E / 8) : ((neg_E - 7) / 8);
      int frac = neg_E - 8 * int_part;
      int32_t inv_step_m =
          (int32_t)((int64_t)quant_pow2_eighth_q30[frac] * quant_inv_bw_q28[b] >> 30);
      int total_shift = QUANT_TOTAL_Q - lb - int_part;

      const hqlc_rans_band_tables *tbl = &tables[b >> 1];

      if (total_shift >= 64) {
        continue; // everything quantizes to zero
      } else if (total_shift >= 32) {
        // Common path, high-word multiply
        int small_shift = total_shift - 32;
        for (int i = s; i < e; i++) {
          int32_t abs_spec = fxp_abs_i32(ch_spec[i]);
          int32_t hi = (int32_t)((int64_t)abs_spec * inv_step_m >> 32);
          int32_t q = ((hi >> small_shift) + QUANT_DZ_BIAS_Q8) >> 8;
          if (q > 0) {
            total_q8 += rans_coeff_cost_q8(tbl, (int16_t)q);
          }
        }
      } else if (total_shift > 0) {
        // Full 64-bit product path
        for (int i = s; i < e; i++) {
          int32_t abs_spec = fxp_abs_i32(ch_spec[i]);
          int32_t scaled_q8 = (int32_t)((int64_t)abs_spec * inv_step_m >> total_shift);
          int32_t q = (scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8;
          if (q > 0) {
            total_q8 += rans_coeff_cost_q8(tbl, (int16_t)q);
          }
        }
      } else {
        // Saturation path, any nonzero coeff costs max
        for (int i = s; i < e; i++) {
          if (ch_spec[i] != 0) {
            total_q8 += rans_coeff_cost_q8(tbl, (int16_t)32767);
          }
        }
      }
    }
    HQLC_BENCH_END(HQLC_BENCH_ENC_RC_QUANT);
  }

  // Convert Q8 bit count to integer bits (round to nearest)
  return (total_q8 + 128) >> 8;
}

// Build (or reuse cached) rANS tables for a gain code.
static void ensure_rans_tables(hqlc_encoder *enc, int gc, hqlc_rans_band_tables *tables) {
  if (enc->cached_tbl_gc != gc) {
    HQLC_BENCH_BEGIN();
    rans_build_band_tables(gc, tables);
    HQLC_BENCH_END(HQLC_BENCH_ENC_RC_TABLE);
    enc->cached_tbl_gc = gc;
  }
}

// Estimate how many gain-code steps to adjust, based on the ratio of
// target bits to probed bits. Returns a signed delta in gain-code units.
static int estimate_gain_delta(int effective_target, int probed_bits) {
  uint32_t b_clamped = (uint32_t)(probed_bits > 0 ? probed_bits : 1);
  int32_t log2_ratio_q8 =
      log2_q8_u32((uint32_t)effective_target) - log2_q8_u32(b_clamped);
  // QUANT_GAIN_Q codes per octave to delta = log2_ratio_q8 / 32 with rounding
  if (log2_ratio_q8 >= 0) {
    return (log2_ratio_q8 + 16) >> 5;
  }
  return -((-log2_ratio_q8 + 16) >> 5);
}

// Compute the maximum downward slew (in gain-code steps) based on how far
// the current gain is above the long-term EMA.
static int compute_slew_limit(hqlc_encoder *enc) {
  int32_t oct_above_q8 = (enc->prev_gain_code << 8) - enc->ema_gain_q8;
  if (oct_above_q8 < 0) {
    oct_above_q8 = 0;
  }
  if (oct_above_q8 > 3072) { // > 1.5 octaves
    return QUANT_GAIN_Q * 3;
  }
  if (oct_above_q8 > 1024) { // > 0.5 octaves
    return QUANT_GAIN_Q * 2;
  }
  return QUANT_GAIN_Q;
}

// Select the gain code for this frame. In RC mode, does a 2-probe search
// to find the gain code closest to the target bitrate. In fixed mode,
// just returns the configured gain code. Also ensures the rANS tables
// are built for the chosen gain before returning.
static int select_gain(hqlc_encoder *enc,
                       const int32_t *spec_q31,
                       const int *loss_bits,
                       const int32_t *exp_indices,
                       const uint8_t *nf_mask,
                       int n_ch,
                       hqlc_rans_band_tables *rans_tables,
                       bool *quiet_frame_out,
                       int *target_bpf_out) {
  *quiet_frame_out = false;
  *target_bpf_out = 0;

  if (enc->mode != HQLC_MODE_RC) {
    // Fixed gain — just use the configured code
    ensure_rans_tables(enc, enc->gain_code, rans_tables);
    return enc->gain_code;
  }

  // Rate-controlled mode, does a 2-probe gain search
  int target_bpf = (int)((int64_t)enc->bitrate * HQLC_FRAME_SAMPLES / HQLC_SAMPLE_RATE);
  *target_bpf_out = target_bpf;
  int tol = target_bpf / 50;
  if (tol < 8) {
    tol = 8;
  }

  // Dampen reservoir borrow to avoid extreme rate swings
  int borrow = fxp_clamp_i32(enc->res_bits, -target_bpf, target_bpf);

  // Effective target = nominal + half the damped borrow
  int effective_target = target_bpf + borrow / 2;
  effective_target = fxp_clamp_i32(effective_target, target_bpf / 4, target_bpf * 3);

  // First probe, at previous frame's gain code
  int gc0 = enc->prev_gain_code;
  if (gc0 > QUANT_GAIN_RC_MAX) {
    gc0 = QUANT_GAIN_RC_MAX;
  }
  ensure_rans_tables(enc, gc0, rans_tables);
  int b0 = probe_frame_bits(
               spec_q31, loss_bits, exp_indices, nf_mask, gc0, n_ch, rans_tables) +
           enc->prev_side_bits;

  int err0 = b0 - effective_target;
  if (err0 < 0) {
    err0 = -err0;
  }
  if (err0 <= tol || b0 <= 0) {
    return gc0; // close enough
  }

  // Second probe, at a delta from the previous gain code
  int delta = estimate_gain_delta(effective_target, b0);

  // ensure slew rate
  int slew_dn = compute_slew_limit(enc);
  delta = fxp_clamp_i32(delta, -slew_dn, QUANT_GAIN_Q);
  if (delta == 0) {
    delta = (b0 < effective_target) ? 1 : -1;
  }

  int gc1 = fxp_clamp_i32(gc0 + delta, 0, QUANT_GAIN_RC_MAX);
  ensure_rans_tables(enc, gc1, rans_tables);
  int b1 = probe_frame_bits(
               spec_q31, loss_bits, exp_indices, nf_mask, gc1, n_ch, rans_tables) +
           enc->prev_side_bits;

  // if raising the gain barely changes the bit count, the signal is near-silent — hold
  // the gain to prevent inflation.
  if (gc1 > gc0 && b0 < effective_target && (b1 - b0) < tol * (gc1 - gc0) / 2) {
    *quiet_frame_out = true;
    ensure_rans_tables(enc, gc0, rans_tables);
    return gc0;
  }

  // Pick whichever probe is closer to the target
  int e0 = b0 - effective_target;
  if (e0 < 0) {
    e0 = -e0;
  }
  int e1 = b1 - effective_target;
  if (e1 < 0) {
    e1 = -e1;
  }
  int chosen = (e1 < e0) ? gc1 : gc0;
  ensure_rans_tables(enc, chosen, rans_tables);
  return chosen;
}

hqlc_error hqlc_encode_frame(hqlc_encoder *enc,
                             const uint8_t *pcm,
                             hqlc_pcm_format fmt,
                             uint8_t *out,
                             size_t out_cap,
                             size_t *out_len,
                             void *scratch) {
  if (!enc || !pcm || !out || !out_len || !scratch) {
    return HQLC_ERR_INVALID_ARG;
  }

  int n_ch = enc->channels;
  int gain_code;
  size_t bps = (fmt == HQLC_PCM16) ? 2 : 3;
  size_t frame_pcm_bytes = (size_t)HQLC_FRAME_SAMPLES * n_ch * bps;

  // Scratch layout
  hqlc_enc_scratch *s = (hqlc_enc_scratch *)scratch;
  int32_t *spec_q31 = s->spec_q31;
  int32_t *exp_indices = s->exp_indices;
  int16_t *quant = s->quant;
  uint8_t *nf_mask = s->nf_mask;

  int loss_bits[HQLC_MAX_CHANNELS];
  tns_info tns[HQLC_MAX_CHANNELS];

  // Per channel main pipeline. Does MDCT -> TNS -> PSYCHO -> QUANT
  for (int ch = 0; ch < n_ch; ch++) {
    int32_t *ch_spec = &spec_q31[ch * HQLC_FRAME_SAMPLES];
    int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];
    uint8_t *ch_nf = &nf_mask[ch * PSY_N_BANDS];

    // MDCT forward
    HQLC_BENCH_BEGIN();
    // We use previous PCM + current PCM directly, no block assembly. Technically, in MDCT
    // this is a continous region for the overlap
    hqlc_error err = mdct_forward(enc->prev_pcm,
                                  pcm,
                                  frame_pcm_bytes,
                                  fmt,
                                  n_ch,
                                  ch,
                                  ch_spec,
                                  HQLC_FRAME_SAMPLES,
                                  s->stage.analysis.mdct_work,
                                  MDCT_SCRATCH_BYTES,
                                  &loss_bits[ch]);
    HQLC_BENCH_END(HQLC_BENCH_ENC_MDCT);
    if (err != HQLC_OK) {
      return err;
    }

    // TNS analysis
    HQLC_BENCH_BEGIN();
    tns[ch].order = 0;
    if (tns_detect_transient(enc->prev_pcm, pcm, fmt, n_ch, ch)) {
      // Calculate TNS reflection coefficients if transient detected
      tns_analyze(ch_spec, &tns[ch]);
      if (tns[ch].order > 0) {
        // Apply TNS filter in place, returns the loss bits adjustment
        loss_bits[ch] += tns_fir_safe(ch_spec, tns[ch].k_q30, tns[ch].order);
      }
    }

    HQLC_BENCH_END(HQLC_BENCH_ENC_TNS);

    // Band analysis
    HQLC_BENCH_BEGIN();
    psy_band_analysis(ch_spec,
                      loss_bits[ch],
                      ch_exp,
                      s->stage.analysis.band_energy,
                      s->stage.analysis.band_peak);

    // Calculate the spreading envelope, used for NF tier 2.
    // Only used for noise fill - the exponents already provide a sufficient perceptual
    // model.
    psy_spreading_envelope(ch_exp, s->stage.analysis.smr_q4);
    for (int b = 0; b < PSY_N_BANDS; b++) {
      ch_nf[b] =
          (ch_exp[b] <= PSY_NF_EXP_MAX &&
           psy_nf_crest_below(
               b, s->stage.analysis.band_energy[b], s->stage.analysis.band_peak[b])) ||
          (ch_exp[b] <= PSY_NF_EXP_MAX_TIER2 &&
           s->stage.analysis.smr_q4[b] <= PSY_NF_SMR_THRESHOLD_Q4);
    }
    HQLC_BENCH_END(HQLC_BENCH_ENC_PSY);
  }

  // Save current frame as prev for next call, needed for the MDCT overlap
  memcpy(enc->prev_pcm, pcm, frame_pcm_bytes);

  // Gain selection + rANS table setup
  hqlc_rans_band_tables *rans_tables = s->rans_tables;
  uint8_t *rans_tmp = s->stage.coding.rans_tmp;
  bool quiet_frame = false;
  int target_bpf = 0;

  // run the gain selection algorithm, will do two probes in case of RC mode
  gain_code = select_gain(enc,
                          spec_q31,
                          loss_bits,
                          exp_indices,
                          nf_mask,
                          n_ch,
                          rans_tables,
                          &quiet_frame,
                          &target_bpf);

  // Quantize + NF zero at chosen gain
  HQLC_BENCH_BEGIN();
  for (int ch = 0; ch < n_ch; ch++) {
    int16_t *ch_quant = &quant[ch * HQLC_FRAME_SAMPLES];
    uint8_t *ch_nf = &nf_mask[ch * PSY_N_BANDS];

    // Run the quantizer with the chosen gain code
    quant_forward(&spec_q31[ch * HQLC_FRAME_SAMPLES],
                  loss_bits[ch],
                  &exp_indices[ch * PSY_N_BANDS],
                  gain_code,
                  ch_quant);

    for (int b = 0; b < PSY_N_BANDS; b++) {
      if (ch_nf[b]) {
        // Zero out all the coeffs of the NF band
        int s = psy_band_edges[b];
        int e = psy_band_edges[b + 1];
        for (int i = s; i < e; i++) {
          ch_quant[i] = 0;
        }
      }
    }
  }
  HQLC_BENCH_END(HQLC_BENCH_ENC_RC_FINAL);

  // Write side information bitstream
  HQLC_BENCH_BEGIN();
  hqlc_bitwriter bw;
  bw_init(&bw, out, out_cap);

  // Gain code (7 bits)
  bw_write(&bw, (uint32_t)gain_code, QUANT_GAIN_BITS);

  // TNS per CH flag + optional order and LAR indices
  for (int ch = 0; ch < n_ch; ch++) {
    if (tns[ch].order == 0) {
      // No TNS, just write a 0 flag
      bw_write(&bw, 0, 1);
    } else {
      // TNS on, write the 1 flag, order + the LAR indices
      bw_write(&bw, 1, 1);
      bw_write(&bw, (uint32_t)(tns[ch].order - 1), 3);
      for (int i = 0; i < tns[ch].order; i++) {
        bw_write(&bw, (uint32_t)(tns[ch].q_lar[i] + TNS_LAR_HALF), TNS_K_BITS);
      }
    }
  }

  // DPCM codec exponents for ch0
  {
    int32_t deltas[PSY_N_BANDS];
    int32_t prev = 0;
    for (int b = 0; b < PSY_N_BANDS; b++) {
      deltas[b] = exp_indices[b] - prev;
      prev = exp_indices[b];
    }
    int k = find_best_rice_k(deltas, PSY_N_BANDS);
    bw_write(&bw, (uint32_t)k, 3);
    for (int b = 0; b < PSY_N_BANDS; b++) {
      bw_write_rice(&bw, zigzag_enc(deltas[b]), k);
    }
  }

  // Ch1+ exponents - calculated as delta from ch0
  for (int ch = 1; ch < n_ch; ch++) {
    int32_t deltas[PSY_N_BANDS];
    for (int b = 0; b < PSY_N_BANDS; b++) {
      deltas[b] = exp_indices[ch * PSY_N_BANDS + b] - exp_indices[b];
    }
    int k = find_best_rice_k(deltas, PSY_N_BANDS);
    bw_write(&bw, (uint32_t)k, 3);
    for (int b = 0; b < PSY_N_BANDS; b++) {
      bw_write_rice(&bw, zigzag_enc(deltas[b]), k);
    }
  }

  // NF masks (1 bit per band per channel)
  for (int ch = 0; ch < n_ch; ch++) {
    for (int b = 0; b < PSY_N_BANDS; b++) {
      bw_write(&bw, nf_mask[ch * PSY_N_BANDS + b], 1);
    }
  }

  // Flush + byte aligns
  bw_flush(&bw);
  size_t side_bytes = bw_bytes(&bw);

  // rANS encode coefficients
  size_t rans_len = rans_encode_coeffs(
      quant, nf_mask, n_ch, rans_tables, rans_tmp, HQLC_MAX_FRAME_BYTES);

  // Assemble output: side info + rANS stream
  if (side_bytes + rans_len > out_cap) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  memcpy(out + side_bytes, rans_tmp, rans_len);
  *out_len = side_bytes + rans_len;
  HQLC_BENCH_END(HQLC_BENCH_ENC_ENTROPY);

  // RC state update
  if (enc->mode == HQLC_MODE_RC) {
    int frame_bits = (int)(*out_len * 8);
    // Only update res_bits for non-quiet frames
    if (!quiet_frame) {
      enc->res_bits += target_bpf - frame_bits;
      enc->res_bits = fxp_clamp_i32(enc->res_bits, -(2 * target_bpf), 2 * target_bpf);
      // EMA: ema += (new - ema) / 16  (alpha ≈ 0.06)
      enc->ema_gain_q8 += ((gain_code << 8) - enc->ema_gain_q8) >> 4;
    }
    enc->prev_gain_code = gain_code;
    enc->prev_side_bits = (int)(side_bytes * 8) + 32; // +32: rANS state flush overhead
  }

  return HQLC_OK;
}

hqlc_error hqlc_decoder_init(hqlc_decoder *dec, uint8_t channels, uint32_t sample_rate) {
  if (!dec) {
    return HQLC_ERR_INVALID_ARG;
  }
  if (sample_rate != HQLC_SAMPLE_RATE) {
    return HQLC_ERR_UNSUPPORTED_RATE;
  }
  if (channels < 1 || channels > HQLC_MAX_CHANNELS) {
    return HQLC_ERR_UNSUPPORTED_CHANNELS;
  }

  memset(dec, 0, sizeof(*dec));
  dec->channels = channels;
  dec->sample_rate = sample_rate;
  for (int ch = 0; ch < channels; ch++) {
    mdct_ola_init(&dec->ola[ch]);
  }

  return HQLC_OK;
}

void hqlc_decoder_reset(hqlc_decoder *dec) {
  if (!dec) {
    return;
  }
  for (int ch = 0; ch < dec->channels; ch++) {
    mdct_ola_init(&dec->ola[ch]);
  }
}

hqlc_error hqlc_decode_frame(hqlc_decoder *dec,
                             const uint8_t *payload,
                             size_t payload_len,
                             uint8_t *pcm_out,
                             hqlc_pcm_format fmt,
                             void *scratch) {
  if (!dec || !payload || !pcm_out || !scratch) {
    return HQLC_ERR_INVALID_ARG;
  }

  int n_ch = dec->channels;

  // Scratch layout
  hqlc_dec_scratch *s = (hqlc_dec_scratch *)scratch;
  int16_t *quant_buf = s->quant_buf;
  int32_t *exp_indices = s->exp_indices;
  uint8_t *nf_mask = s->nf_mask;

  // Read side information
  HQLC_BENCH_BEGIN();
  hqlc_bitreader br;
  br_init(&br, payload, payload_len);

  // Read the gain code
  int gain_code = (int)br_read(&br, QUANT_GAIN_BITS);

  // TNS per channel
  tns_info tns[HQLC_MAX_CHANNELS];
  for (int ch = 0; ch < n_ch; ch++) {
    tns[ch].order = 0;
    uint32_t active = br_read(&br, 1);
    if (active) {
      int order = (int)br_read(&br, 3) + 1;
      tns[ch].order = order;
      for (int i = 0; i < order; i++) {
        int q = (int)br_read(&br, TNS_K_BITS) - TNS_LAR_HALF;
        tns[ch].q_lar[i] = (int8_t)q;
        tns[ch].k_q30[i] = tns_dequant_k(q);
      }
    }
  }

  // DPCM ch0 exponents
  {
    int k = (int)br_read(&br, 3);
    int32_t prev = 0;
    for (int b = 0; b < PSY_N_BANDS; b++) {
      uint32_t u = br_read_rice(&br, k);
      exp_indices[b] = prev + zigzag_dec(u);
      prev = exp_indices[b];
    }
  }

  // CH1+ exponents are encoded as deltas from ch0
  for (int ch = 1; ch < n_ch; ch++) {
    int k = (int)br_read(&br, 3);
    for (int b = 0; b < PSY_N_BANDS; b++) {
      uint32_t u = br_read_rice(&br, k);
      exp_indices[ch * PSY_N_BANDS + b] = exp_indices[b] + zigzag_dec(u);
    }
  }

  // NF masks
  for (int ch = 0; ch < n_ch; ch++) {
    for (int b = 0; b < PSY_N_BANDS; b++) {
      nf_mask[ch * PSY_N_BANDS + b] = (uint8_t)br_read(&br, 1);
    }
  }

  // Byte-align to find rANS stream start
  size_t bits_used = br_bits(&br);
  int pad = (int)((8 - bits_used % 8) % 8);
  if (pad) {
    br_read(&br, pad);
  }
  size_t rans_start = br_bits(&br) / 8;

  // rANS decode coefficients
  rans_build_band_tables(gain_code, s->stage.entropy.tables);

  const uint8_t *rans_data = payload + rans_start;
  size_t rans_len = (rans_start < payload_len) ? payload_len - rans_start : 0;

  memset(quant_buf, 0, (size_t)n_ch * HQLC_FRAME_SAMPLES * sizeof(int16_t));

  if (rans_len > 0) {
    rans_decode_coeffs(
        rans_data, rans_len, quant_buf, nf_mask, n_ch, s->stage.entropy.tables);
  }
  HQLC_BENCH_END(HQLC_BENCH_DEC_ENTROPY);

  for (int ch = 0; ch < n_ch; ch++) {
    int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];
    int16_t *ch_quant = &quant_buf[ch * HQLC_FRAME_SAMPLES];
    uint8_t *ch_nf = &nf_mask[ch * PSY_N_BANDS];
    int32_t *spec_q31 = s->stage.synthesis.spec_q31;

    // Inverse quantize
    HQLC_BENCH_BEGIN();
    int loss_bits;
    quant_inverse(ch_quant,
                  ch_exp,
                  gain_code,
                  spec_q31,
                  &loss_bits,
                  s->stage.synthesis.dequant_tmp);

    // NF synthesis: fill noise-filled bands with shaped noise
    for (int b = 0; b < PSY_N_BANDS; b++) {
      if (!ch_nf[b]) {
        continue;
      }
      int32_t amp = nf_compute_amp_q31(ch_exp[b], loss_bits);
      if (amp <= 0) {
        continue;
      }
      uint32_t seed = (dec->frame_idx * 2246822519u +
                       (uint32_t)(b + ch * PSY_N_BANDS) * 3266489917u + NF_SEED_BIAS);
      nf_fill_band(spec_q31, psy_band_edges[b], psy_band_edges[b + 1], amp, seed);
    }
    HQLC_BENCH_END(HQLC_BENCH_DEC_DEQUANT_NF);

    // TNS synthesis / inerse filter
    HQLC_BENCH_BEGIN();
    if (tns[ch].order > 0) {
      loss_bits += tns_iir_safe(spec_q31, tns[ch].k_q30, tns[ch].order);
    }
    HQLC_BENCH_END(HQLC_BENCH_DEC_TNS);

    // IMDCT inverse to 1024 windowed samples
    HQLC_BENCH_BEGIN();
    int32_t *windowed = s->stage.synthesis.windowed;
    int loss_time;
    hqlc_error err = mdct_inverse(spec_q31,
                                  HQLC_FRAME_SAMPLES,
                                  loss_bits,
                                  windowed,
                                  HQLC_BLOCK_SAMPLES,
                                  s->stage.synthesis.mdct_work,
                                  MDCT_SCRATCH_BYTES,
                                  &loss_time);
    if (err != HQLC_OK) {
      return err;
    }

    // OLA: overlap-add with previous frame's second half
    mdct_ola_state *ola = &dec->ola[ch];

    if (!ola->has_overlap) {
      ola->loss_td_bits = loss_time;
      ola->has_overlap = true;
    }

    // Align both halves to the worse (higher-loss) domain.
    // Pre-shift both by >>1 to prevent int32 overflow on add,
    // compensate by adding 1 to common_loss.
    int ola_loss = ola->loss_td_bits;
    int common_loss = (ola_loss > loss_time) ? ola_loss : loss_time;
    int ola_shift = (common_loss - ola_loss) + 1;
    int win_shift = (common_loss - loss_time) + 1;
    int adj_loss = common_loss + 1;

    if (fmt == HQLC_PCM16) {
      // PCM16 fast path with int64 OLA sum to prevent rare overflow.
      int16_t *out16 = (int16_t *)pcm_out;
      if (ola_shift >= 31 && win_shift >= 31) {
        for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
          out16[i * n_ch + ch] = 0;
        }
      } else if (adj_loss < 0) {
        int rsh = 16 - adj_loss;
        for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
          int32_t o =
              (ola_shift < 31) ? fxp_shr_rnd_i32(ola->overlap_q31[i], ola_shift) : 0;
          int32_t w = (win_shift < 31) ? fxp_shr_rnd_i32(windowed[i], win_shift) : 0;
          int32_t y = fxp_sat_i64_to_i32((int64_t)o + w);
          out16[i * n_ch + ch] = (int16_t)fxp_shr_rnd_i32(y, rsh);
        }
      } else {
        for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
          int32_t o =
              (ola_shift < 31) ? fxp_shr_rnd_i32(ola->overlap_q31[i], ola_shift) : 0;
          int32_t w = (win_shift < 31) ? fxp_shr_rnd_i32(windowed[i], win_shift) : 0;
          int32_t y = fxp_sat_i64_to_i32((int64_t)o + w);
          int32_t pcm_val = fxp_shl_sat_i32(y, adj_loss);
          int32_t pcm16 = (pcm_val >> 16) + ((pcm_val >> 15) & 1);
          out16[i * n_ch + ch] = pcm_clamp_i16(pcm16);
        }
      }
    } else {
      for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
        int32_t o =
            (ola_shift < 31) ? fxp_shr_rnd_i32(ola->overlap_q31[i], ola_shift) : 0;
        int32_t w = (win_shift < 31) ? fxp_shr_rnd_i32(windowed[i], win_shift) : 0;
        int32_t y = fxp_sat_i64_to_i32((int64_t)o + w);

        int32_t pcm_val;
        if (adj_loss > 0) {
          pcm_val = fxp_shl_sat_i32(y, adj_loss);
        } else if (adj_loss < 0) {
          pcm_val = fxp_shr_rnd_i32(y, -adj_loss);
        } else {
          pcm_val = y;
        }

        pcm_store_q31(pcm_out, fmt, i * n_ch + ch, pcm_val);
      }
    }

    // Store new overlap, normalized to maximize int32 headroom
    int32_t *new_overlap = &windowed[HQLC_FRAME_SAMPLES];
    uint32_t ola_or = 0;
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      ola_or |= (uint32_t)fxp_abs_i32(new_overlap[i]);
    }
    int hr = (ola_or == 0) ? 31 : __builtin_clz(ola_or) - 1;
    for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
      ola->overlap_q31[i] = new_overlap[i] << hr;
    }
    ola->loss_td_bits = loss_time - hr;
    HQLC_BENCH_END(HQLC_BENCH_DEC_IMDCT_OLA);
  }

  dec->frame_idx++;
  return HQLC_OK;
}
