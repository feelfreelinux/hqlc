#include "hqlc.h"

#include <math.h>
#include <string.h>

#include "entropy.h"
#include "entropy_tables.h"
#include "fxp.h"
#include "mdct.h"
#include "ms.h"
#include "psy.h"
#include "quant.h"
#include "tns.h"

#include "hqlc_bench.h"

#ifdef HQLC_BENCH
hqlc_bench_ctx *hqlc_bench = NULL;
#endif

// Boost for transient frames, letting it use more bits
// 1/4 = 25% boost in the rate controller
#define RC_TRANSIENT_BOOST 4

#define GAIN_CODE_MAX ((1 << QUANT_GAIN_BITS) - 1)

// The public fixed-mode gain is converted once at the encoder boundary. The
// codec pipeline and bitstream use gain codes exclusively.
static int gain_code_from_float(float gain) {
  float code = log2f(gain) * QUANT_GAIN_Q;
  int rounded = (int)(code >= 0.0f ? code + 0.5f : code - 0.5f);
  return fxp_clamp_i32(rounded + QUANT_GAIN_BIAS, 0, GAIN_CODE_MAX);
}

// Interpolated quantizer steps on tonal frames, flat on TNS (transient)
// frames. The TNS flag is transmitted, so both sides gate identically.
static inline bool env_interp_active(int tns_order) {
  return tns_order == 0;
}

// Encoder scratch, allocated by the caller
typedef struct {
  // Spectral coefficients (Q31)
  int32_t spec_q31[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];

  // Per-band exponent indices
  int32_t exp_indices[HQLC_MAX_CHANNELS * PSY_N_BANDS];
  // Quantized coefficients
  int16_t quant[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];

  // Per-stage temporaries (only one active at a time)
  union {
    struct {
      int64_t mdct_work[MDCT_SCRATCH_BYTES / sizeof(int64_t)];
    } analysis;
    struct {
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

  // Per-stage temporaries
  union {
    struct {
      int32_t spec_q31[HQLC_MAX_CHANNELS * HQLC_FRAME_SAMPLES];
      int64_t mdct_work[MDCT_SCRATCH_BYTES / sizeof(int64_t)];
    } synthesis;
  } stage;
} hqlc_dec_scratch;

// Internal encoder struct
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
  int prev_overhead_bits;
  int32_t res_bits;
  int tilt_step_q7; // per-fine-band tilt increment in EXP_Q7

  tns_detect_state tns_det[HQLC_MAX_CHANNELS];
  uint8_t tns_hang[HQLC_MAX_CHANNELS]; // frames of TNS eligibility left post-attack

  // Per-band M/S eligibility flags
  bool ms_flags[PSY_N_BANDS];
};

struct hqlc_decoder {
  uint8_t channels;
  uint32_t sample_rate;
  uint32_t frame_count; // varies the noise fill seed per frame
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
    enc->prev_gain_code = QUANT_GAIN_BIAS + 4 * QUANT_GAIN_Q;
    enc->ema_gain_q8 = fxp_rescale_i32(enc->prev_gain_code, 0, 8);
    // Conservative typical fixed side info + rANS flush overhead. The exact
    // measured overhead replaces this after the first encoded frame.
    enc->prev_overhead_bits = 64;
    enc->res_bits = 0;
    enc->tilt_step_q7 = psy_tilt_step_q7(psy_tilt_for_bitrate(cfg->bitrate));
    break;
  case HQLC_MODE_FIXED:
    if (!(cfg->gain > 0.0f)) {
      return HQLC_ERR_INVALID_ARG;
    }
    enc->gain_code = gain_code_from_float(cfg->gain);
    enc->tilt_step_q7 = psy_tilt_step_q7(psy_tilt_for_bitrate(96000));
    break;
  default:
    return HQLC_ERR_INVALID_ARG;
  }

  enc->channels = cfg->channels;
  enc->sample_rate = cfg->sample_rate;
  enc->mode = cfg->mode;
  return HQLC_OK;
}

// Cost (Q8 bits) of the EG(0) escape for magnitudes past the rANS alphabet:
// (nbits+1) unary + nbits binary, matching rans_encode_payload
static inline int32_t eg0_overflow_cost_q8(int q) {
  int bit_count = 2 * rans_eg0_nbits(q - (RANS_MAX_SYM - 1)) + 1;
  return fxp_rescale_i32(bit_count, 0, 8);
}

// Estimate the rANS payload cost (bits) at a candidate gain_code without quantizing. Uses
// flat per-band steps, as the interp quantizer does not affect the cost too much.
static int probe_frame_bits(const bfp_i32 *spectra,
                            const int32_t *exp_indices,
                            int gain_code,
                            int n_ch,
                            const bool *ms_flags) {
  int32_t total_q8 = 0;

  for (int ch = 0; ch < n_ch; ch++) {
    const int32_t *ch_spec = spectra[ch].data;
    const int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];

    HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_RC_QUANT);
    int prev_nz = 0, prev_w = 1;

    // Precompute alpha bins (gain_code constant, only sigma varies per pair)
    int32_t log2_gain_q8 = (gain_code - QUANT_GAIN_BIAS) * 32;
    int8_t abins[PSY_N_BANDS];
    for (int b = 0; b < PSY_N_BANDS; b++) {
      abins[b] =
          (int8_t)rans_alpha_bin_from_la(log2_gain_q8 + rans_log2_sigma_q8[b >> 1]);
    }

    for (int b = 0; b < PSY_N_BANDS; b++) {
      int s = psy_band_edges[b];
      int e = psy_band_edges[b + 1];
      int w = e - s;

      quant_scale scale =
          quant_forward_scale((int)ch_exp[b], gain_code, spectra[ch].exp2);

      int abin = abins[b];
      if (ch == 1 && ms_flags && ms_flags[b]) {
        // Account for coarsened S channel
        abin -= MS_RANS_ALPHA_SHIFT;
        if (abin < 0) {
          abin = 0;
        }
      }
      int act = (b > 0) ? rans_activity_from(prev_nz, prev_w) : 0;
      const int16_t *cost = rans_cost_q8[rans_table_idx(abin, act)];

      int nz = 0;
      HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_PROBE_BAND);
      if (scale.product_rshift >= 64) {
        // everything quantizes to zero
      } else if (scale.product_rshift >= 32) {
        int small_shift = scale.product_rshift - 32;
        for (int i = s; i < e; i++) {
          int32_t abs_spec = fxp_abs_i32(ch_spec[i]);
          int32_t hi = fxp_mul_rshift_i32(abs_spec, scale.multiplier_q28, 32);
          int32_t q = ((hi >> small_shift) + QUANT_DZ_BIAS_Q8) >> 8;
          if (q > 0) {
            if (q < RANS_MAX_SYM - 1) {
              total_q8 += cost[q] + FXP_Q8(1.0); // symbol + 1 sign bit
            } else {
              total_q8 +=
                  cost[RANS_MAX_SYM - 1] + FXP_Q8(1.0) + eg0_overflow_cost_q8(q);
            }
            nz++;
          }
        }
      } else if (scale.product_rshift > 0) {
        for (int i = s; i < e; i++) {
          int32_t abs_spec = fxp_abs_i32(ch_spec[i]);
          int32_t scaled_q8 = (int32_t)((int64_t)abs_spec * scale.multiplier_q28 >>
                                        scale.product_rshift);
          int32_t q = (scaled_q8 + QUANT_DZ_BIAS_Q8) >> 8;
          if (q > 0) {
            if (q < RANS_MAX_SYM - 1) {
              total_q8 += cost[q] + FXP_Q8(1.0); // symbol + 1 sign bit
            } else {
              total_q8 +=
                  cost[RANS_MAX_SYM - 1] + FXP_Q8(1.0) + eg0_overflow_cost_q8(q);
            }
            nz++;
          }
        }
      } else {
        // Step so fine every nonzero bin escapes. RC never picks gains this
        // fine, so a flat ESC cost is close enough
        for (int i = s; i < e; i++) {
          if (ch_spec[i] != 0) {
            total_q8 += cost[RANS_MAX_SYM - 1] + FXP_Q8(1.0);
            nz++;
          }
        }
      }
      HQLC_BENCH_END(HQLC_BENCH_ENC_PROBE_BAND);
      // Also count zero's as cost (symbol 0 per bin)
      total_q8 += (int32_t)cost[0] * (w - nz);
      prev_nz = nz;
      prev_w = w;
    }
    HQLC_BENCH_END(HQLC_BENCH_ENC_RC_QUANT);
  }

  // Exponents share the rANS payload with coefficients; estimate their cost
  // for this frame rather than carrying the previous frame's estimate.
  total_q8 += rans_exp_payload_cost_q8(exp_indices, n_ch, ms_flags);

  return fxp_rescale_i32(total_q8, 8, 0);
}

// Estimate how many gain-code steps to adjust, based on the ratio of
// target bits to probed bits. Returns a signed delta in gain-code units.
static int estimate_gain_delta(int effective_target, int probed_bits) {
  uint32_t t_clamped = (uint32_t)(effective_target > 0 ? effective_target : 1);
  uint32_t b_clamped = (uint32_t)(probed_bits > 0 ? probed_bits : 1);
  int32_t log2_ratio_q8 = fxp_log2_q8_u64(t_clamped) - fxp_log2_q8_u64(b_clamped);
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
// just returns the configured gain code.
static int select_gain(hqlc_encoder *enc,
                       const bfp_i32 *spectra,
                       const int32_t *exp_indices,
                       int n_ch,
                       const bool *ms_flags,
                       bool transient,
                       bool *quiet_frame_out,
                       int *target_bpf_out) {
  *quiet_frame_out = false;
  *target_bpf_out = 0;

  if (enc->mode != HQLC_MODE_RC) {
    return enc->gain_code;
  }

  int target_bpf = (int)((int64_t)enc->bitrate * HQLC_FRAME_SAMPLES / HQLC_SAMPLE_RATE);
  *target_bpf_out = target_bpf;
  int tol = target_bpf / 50;
  if (tol < 8) {
    tol = 8;
  }

  // Attacks get a larger share of the budget, resevoir absorbs it later
  int borrow = fxp_clamp_i32(enc->res_bits, -target_bpf, target_bpf);
  int boost = transient ? target_bpf / RC_TRANSIENT_BOOST : 0;
  int effective_target = target_bpf + borrow / 2 + boost;
  effective_target = fxp_clamp_i32(effective_target, target_bpf / 4, target_bpf * 3);

  int gc0 = enc->prev_gain_code;
  if (gc0 > QUANT_GAIN_RC_MAX) {
    gc0 = QUANT_GAIN_RC_MAX;
  }
  int b0 = probe_frame_bits(spectra, exp_indices, gc0, n_ch, ms_flags) +
           enc->prev_overhead_bits;

  int err0 = b0 - effective_target;
  if (err0 < 0) {
    err0 = -err0;
  }
  if (err0 <= tol || b0 <= 0) {
    return gc0;
  }

  int delta = estimate_gain_delta(effective_target, b0);
  int slew_dn = compute_slew_limit(enc);
  delta = fxp_clamp_i32(delta, -slew_dn, QUANT_GAIN_Q);
  if (delta == 0) {
    delta = (b0 < effective_target) ? 1 : -1;
  }

  int gc1 = fxp_clamp_i32(gc0 + delta, 0, QUANT_GAIN_RC_MAX);
  int b1 = probe_frame_bits(spectra, exp_indices, gc1, n_ch, ms_flags) +
           enc->prev_overhead_bits;

  if (gc1 > gc0 && b0 < effective_target && (b1 - b0) < tol * (gc1 - gc0) / 2) {
    *quiet_frame_out = true;
    return gc0;
  }

  int e0 = b0 - effective_target;
  if (e0 < 0) {
    e0 = -e0;
  }
  int e1 = b1 - effective_target;
  if (e1 < 0) {
    e1 = -e1;
  }
  return (e1 < e0) ? gc1 : gc0;
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
  int32_t *exp_indices = s->exp_indices;
  int16_t *quant = s->quant;

  bfp_i32 spectra[HQLC_MAX_CHANNELS];
  for (int ch = 0; ch < n_ch; ch++) {
    spectra[ch] = bfp_i32_view(
        &s->spec_q31[ch * HQLC_FRAME_SAMPLES], HQLC_FRAME_SAMPLES, 0);
  }
  tns_info tns[HQLC_MAX_CHANNELS];
  memset(tns, 0, sizeof(tns));

  bool tns_eligible[HQLC_MAX_CHANNELS] = {false, false};
  bool tns_attack[HQLC_MAX_CHANNELS] = {false, false};

  for (int ch = 0; ch < n_ch; ch++) {
    int32_t *ch_spec = spectra[ch].data;

    // Perform TNS detection on raw spectrum before MDCT
    bool tr = tns_detect_transient(&enc->tns_det[ch], pcm, fmt, n_ch, ch);
    tns_eligible[ch] = tr || enc->tns_hang[ch] > 0;
    tns_attack[ch] = tr;

    // Update the 1 frame hangover
    enc->tns_hang[ch] = tr ? 1 : (enc->tns_hang[ch] > 0 ? enc->tns_hang[ch] - 1 : 0);

    HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_MDCT);
    hqlc_error err = mdct_forward(enc->prev_pcm,
                                  pcm,
                                  frame_pcm_bytes,
                                  fmt,
                                  n_ch,
                                  ch,
                                  &spectra[ch],
                                  s->stage.analysis.mdct_work,
                                  MDCT_SCRATCH_BYTES);
    HQLC_BENCH_END(HQLC_BENCH_ENC_MDCT);
    if (err != HQLC_OK) {
      return err;
    }
    for (int i = PSY_ACTIVE_BINS; i < HQLC_FRAME_SAMPLES; i++) {
      ch_spec[i] = 0;
    }
  }

  if (n_ch == 2) {
    // On transient frames, skip M/S entirely even if gate passes.
    // The sudden attack messes with steady state energy ratios,
    // so the gate flaps in and out
    if (tns_eligible[0] || tns_eligible[1]) {
      memset(enc->ms_flags, 0, sizeof(enc->ms_flags));
    } else {
      // Swaps L/R channels in place with M/S for eligible bands
      ms_encode(&spectra[0], &spectra[1], enc->ms_flags);
    }
  }

  // Perform TNS on eligible frames
  for (int ch = 0; ch < n_ch; ch++) {
    int32_t *ch_spec = spectra[ch].data;
    int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];

    HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_TNS);
    if (tns_eligible[ch]) {
      tns_analyze(ch_spec, !tns_attack[ch], &tns[ch]);
      if (tns[ch].order > 0) {
        tns_apply_analysis_filter(&spectra[ch], tns[ch].k_q30, tns[ch].order);
      }
    }
    HQLC_BENCH_END(HQLC_BENCH_ENC_TNS);

    HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_PSY);
    psy_fine_band_exponents(
        &spectra[ch], enc->tilt_step_q7, tns_eligible[ch], ch_exp);
    HQLC_BENCH_END(HQLC_BENCH_ENC_PSY);
  }

  // Coarsen the envelope for S channel
  if (n_ch == 2) {
    ms_apply_side_exp_bias(&exp_indices[0], &exp_indices[PSY_N_BANDS], enc->ms_flags);
  }

  // Save current frame as prev for next call, needed for the MDCT overlap
  memcpy(enc->prev_pcm, pcm, frame_pcm_bytes);

  uint8_t *rans_tmp = s->stage.coding.rans_tmp;
  bool quiet_frame = false;
  int target_bpf = 0;

  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_SELECT_GAIN);
  gain_code = select_gain(enc,
                          spectra,
                          exp_indices,
                          n_ch,
                          enc->ms_flags,
                          tns_eligible[0] || (n_ch > 1 && tns_eligible[1]),
                          &quiet_frame,
                          &target_bpf);
  HQLC_BENCH_END(HQLC_BENCH_ENC_SELECT_GAIN);

  // Quantize per channel
  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_QUANT_FWD);
  for (int ch = 0; ch < n_ch; ch++) {
    quant_forward(&spectra[ch],
                  &exp_indices[ch * PSY_N_BANDS],
                  gain_code,
                  env_interp_active(tns[ch].order),
                  &quant[ch * HQLC_FRAME_SAMPLES]);
  }
  HQLC_BENCH_END(HQLC_BENCH_ENC_QUANT_FWD);

  // Write side information bitstream
  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_SIDE_INFO);
  hqlc_bitwriter bw;
  bw_init(&bw, out, out_cap);

  // Gain code (7 bits)
  bw_write(&bw, (uint32_t)gain_code, QUANT_GAIN_BITS);

  // Per-band M/S flags, RLE coded
  if (n_ch == 2) {
    // Rice k=1 for flag runs
    bw_write_binary_rle(&bw, enc->ms_flags, PSY_N_BANDS, 1);
  }

  // TNS per CH flag + optional order and LAR indices
  for (int ch = 0; ch < n_ch; ch++) {
    if (tns[ch].order == 0) {
      bw_write(&bw, 0, 1);
    } else {
      bw_write(&bw, 1, 1);
      bw_write(&bw, (uint32_t)(tns[ch].order - 1), 3);
      for (int i = 0; i < tns[ch].order; i++) {
        bw_write(&bw, (uint32_t)(tns[ch].q_lar[i] + TNS_LAR_HALF), TNS_K_BITS);
      }
    }
  }

  // Exponents are rANS-coded inside the coefficient stream

  bw_flush(&bw);
  size_t side_bytes = bw_bytes(&bw);
  HQLC_BENCH_END(HQLC_BENCH_ENC_SIDE_INFO);

  // rANS encode coefficients (+ exponents in the same stream)
  HQLC_BENCH_BEGIN(HQLC_BENCH_ENC_RANS_ENC);
  size_t rans_len = rans_encode_payload(quant,
                                       n_ch,
                                       gain_code,
                                       (n_ch == 2) ? enc->ms_flags : NULL,
                                       exp_indices,
                                       rans_tmp,
                                       HQLC_MAX_FRAME_BYTES);
  HQLC_BENCH_END(HQLC_BENCH_ENC_RANS_ENC);

  // Assemble output: side info + rANS stream
  if (side_bytes + rans_len > out_cap) {
    return HQLC_ERR_BUFFER_TOO_SMALL;
  }

  memcpy(out + side_bytes, rans_tmp, rans_len);
  *out_len = side_bytes + rans_len;

  // RC state update
  if (enc->mode == HQLC_MODE_RC) {
    int frame_bits = (int)(*out_len * 8);
    // Quiet frames bank reservoir credit
    enc->res_bits += target_bpf - frame_bits;
    enc->res_bits = fxp_clamp_i32(enc->res_bits, -(2 * target_bpf), 2 * target_bpf);

    // Don't update gain EMA for quiet frames
    if (!quiet_frame) {
      // Gain EMA, alpha = 1/16
      int32_t gain_code_q8 = fxp_rescale_i32(gain_code, 0, 8);
      enc->ema_gain_q8 += (gain_code_q8 - enc->ema_gain_q8) >> 4;
    }
    enc->prev_gain_code = gain_code;
    // +24: rANS state flush; exponent cost is estimated per candidate frame
    enc->prev_overhead_bits = (int)(side_bytes * 8) + 24;
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
  dec->frame_count = 0;
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

  // Read side information
  HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_ENTROPY);
  hqlc_bitreader br;
  br_init(&br, payload, payload_len);

  int gain_code = (int)br_read(&br, QUANT_GAIN_BITS);

  // Read the M/S flags
  bool ms_flags[PSY_N_BANDS];
  memset(ms_flags, 0, sizeof(ms_flags));
  if (n_ch == 2) {
    br_read_binary_rle(&br, ms_flags, PSY_N_BANDS, 1); // Rice k=1
  }

  // TNS per channel. A valid stream never codes order > TNS_MAX_ORDER or
  // LAR code 15 (q = 8) reject instead of overrunning the tns_info arrays.
  tns_info tns[HQLC_MAX_CHANNELS];
  for (int ch = 0; ch < n_ch; ch++) {
    tns[ch].order = 0;
    uint32_t active = br_read(&br, 1);
    if (active) {
      int order = (int)br_read(&br, 3) + 1;
      if (order > TNS_MAX_ORDER) {
        return HQLC_ERR_BITSTREAM_CORRUPT;
      }
      tns[ch].order = order;
      for (int i = 0; i < order; i++) {
        int q = (int)br_read(&br, TNS_K_BITS) - TNS_LAR_HALF;
        if (q > TNS_LAR_HALF) {
          return HQLC_ERR_BITSTREAM_CORRUPT;
        }
        tns[ch].q_lar[i] = (int8_t)q;
        tns[ch].k_q30[i] = tns_dequant_k(q);
      }
    }
  }

  // Exponents are decoded from the rANS stream

  // Byte-align to find rANS stream start
  size_t bits_used = br_bits(&br);
  int pad = (int)((8 - bits_used % 8) % 8);
  if (pad) {
    br_read(&br, pad);
  }
  size_t rans_start = br_bits(&br) / 8;

  // rANS coefficient decode
  const uint8_t *rans_data = payload + rans_start;
  size_t rans_len = (rans_start < payload_len) ? payload_len - rans_start : 0;

  memset(quant_buf, 0, (size_t)n_ch * HQLC_FRAME_SAMPLES * sizeof(int16_t));

  if (rans_len == 0 || !rans_decode_payload(rans_data,
                                           rans_len,
                                           quant_buf,
                                           n_ch,
                                           gain_code,
                                           (n_ch == 2) ? ms_flags : NULL,
                                           exp_indices)) {
    return HQLC_ERR_BITSTREAM_CORRUPT;
  }
  HQLC_BENCH_END(HQLC_BENCH_DEC_ENTROPY);

  bfp_i32 spectra[HQLC_MAX_CHANNELS];
  for (int ch = 0; ch < n_ch; ch++) {
    spectra[ch] = bfp_i32_view(&s->stage.synthesis.spec_q31[ch * HQLC_FRAME_SAMPLES],
                               HQLC_FRAME_SAMPLES,
                               0);
  }

  // Performs dequant + NF + TNS synthesis per channel
  for (int ch = 0; ch < n_ch; ch++) {
    int32_t *ch_exp = &exp_indices[ch * PSY_N_BANDS];
    int16_t *ch_quant = &quant_buf[ch * HQLC_FRAME_SAMPLES];
    bfp_i32 *spectrum = &spectra[ch];

    // Inverse quantize, interpolate mode on non TNS frames
    HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_DEQUANT);
    quant_inverse(ch_quant,
                  ch_exp,
                  gain_code,
                  env_interp_active(tns[ch].order),
                  spectrum);
    HQLC_BENCH_END(HQLC_BENCH_DEC_DEQUANT);

    // Noise fill, seeded per frame and channel so the texture differs across frames.
    HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_NF);
    uint32_t nf_seed =
        NF_SEED_BIAS ^ (dec->frame_count * 0x9E37u) ^ ((uint32_t)ch * 0x51EDu);

    // skip S bands (only ch 1 of M/S)
    const bool *skip = ch == 1 ? ms_flags : NULL;
    noise_fill(ch_quant,
               ch_exp,
               gain_code,
               env_interp_active(tns[ch].order),
               nf_seed,
               skip,
               spectrum);
    HQLC_BENCH_END(HQLC_BENCH_DEC_NF);

    // TNS synthesis / inverse filter
    HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_TNS);
    if (tns[ch].order > 0) {
      tns_apply_synthesis_filter(spectrum, tns[ch].k_q30, tns[ch].order);
    }
    HQLC_BENCH_END(HQLC_BENCH_DEC_TNS);
  }

  // Decode back to L/R for flagged bands
  if (n_ch == 2) {
    ms_decode(&spectra[0], &spectra[1], ms_flags);
  }

  // do the inverse mdct, OLA, write the PCM output
  for (int ch = 0; ch < n_ch; ch++) {
    bfp_i32 *spectrum = &spectra[ch];
    HQLC_BENCH_BEGIN(HQLC_BENCH_DEC_IMDCT_OLA);
    hqlc_error err = mdct_inverse_ola(spectrum,
                                      &dec->ola[ch],
                                      pcm_out,
                                      fmt,
                                      n_ch,
                                      ch,
                                      s->stage.synthesis.mdct_work,
                                      MDCT_SCRATCH_BYTES);
    if (err != HQLC_OK) {
      return err;
    }
    HQLC_BENCH_END(HQLC_BENCH_DEC_IMDCT_OLA);
  }

  dec->frame_count++;
  return HQLC_OK;
}
