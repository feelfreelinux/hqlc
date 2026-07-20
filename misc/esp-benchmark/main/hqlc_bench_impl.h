#ifndef HQLC_BENCH_IMPL_H
#define HQLC_BENCH_IMPL_H

// Full benchmarking implementation for HQLC_BENCH builds.
// This header is included by hqlc_bench.h when HQLC_BENCH is defined.
// The benchmark build must add this directory to the include path.

#include <stdint.h>
#include <stdio.h>

// Stage indices
enum {
  HQLC_BENCH_ENC_MDCT = 0,
  HQLC_BENCH_ENC_TNS,
  HQLC_BENCH_ENC_PSY,
  HQLC_BENCH_ENC_RC_TABLE,
  HQLC_BENCH_ENC_RC_QUANT,
  HQLC_BENCH_ENC_RC_COST,
  HQLC_BENCH_ENC_SELECT_GAIN,
  HQLC_BENCH_ENC_QUANT_FWD,
  HQLC_BENCH_ENC_NF_EST,
  HQLC_BENCH_ENC_ENTROPY,
  HQLC_BENCH_DEC_ENTROPY,
  HQLC_BENCH_DEC_DEQUANT,
  HQLC_BENCH_DEC_NF,
  HQLC_BENCH_DEC_TNS,
  HQLC_BENCH_DEC_IMDCT_OLA,
  HQLC_BENCH_MDCT_FOLD,
  HQLC_BENCH_MDCT_PRE_TW,
  HQLC_BENCH_MDCT_FFT,
  HQLC_BENCH_MDCT_POST_TW,
  HQLC_BENCH_ENC_SIDE_INFO,
  HQLC_BENCH_ENC_RANS_ENC,
  HQLC_BENCH_ENC_RANS_TBL,
  HQLC_BENCH_DEC_RANS_DEC,
  HQLC_BENCH_ENC_INTERP,
  HQLC_BENCH_ENC_QLOOP,
  HQLC_BENCH_ENC_PROBE_BAND,
  HQLC_BENCH_N_STAGES,
};

// Platform cycle counter
#if defined(__XTENSA__)
static inline uint32_t hqlc_bench_cycles(void) {
  uint32_t c;
  __asm__ __volatile__("rsr %0, ccount" : "=a"(c));
  return c;
}
#elif defined(__riscv)
static inline uint32_t hqlc_bench_cycles(void) {
  uint32_t c;
  __asm__ __volatile__("rdcycle %0" : "=r"(c));
  return c;
}
#else
static inline uint32_t hqlc_bench_cycles(void) {
  return 0;
}
#endif

// Per-stage statistics
typedef struct {
  const char *name;
  uint32_t count;
  uint32_t min;
  uint32_t max;
  uint64_t sum;
} hqlc_bench_stage;

typedef struct {
  hqlc_bench_stage stages[HQLC_BENCH_N_STAGES];
  // Per-stage start marks so nested BEGIN/END pairs don't clobber each other
  uint32_t marks[HQLC_BENCH_N_STAGES];
} hqlc_bench_ctx;

// Global pointer, set by benchmark harness before encode/decode calls.
extern hqlc_bench_ctx *hqlc_bench;

static inline void hqlc_bench_init(hqlc_bench_ctx *ctx) {
  static const char *const names[HQLC_BENCH_N_STAGES] = {
      "enc_mdct",     "enc_tns",      "enc_psy",       "enc_rc_table",
      "enc_rc_quant", "enc_rc_cost",  "enc_select_gain",
      "enc_quant_fwd", "enc_nf_est",   "enc_entropy",
      "dec_entropy",  "dec_dequant",  "dec_nf",       "dec_tns",
      "dec_imdct_ola",
      "mdct_fold",    "mdct_pre_tw",  "mdct_fft",      "mdct_post_tw",
      "enc_side_info", "enc_rans_enc", "enc_rans_tbl", "dec_rans_dec",
      "enc_interp",   "enc_qloop",    "enc_probe_band",
  };
  for (int i = 0; i < HQLC_BENCH_N_STAGES; i++) {
    ctx->stages[i].name = names[i];
    ctx->stages[i].count = 0;
    ctx->stages[i].min = UINT32_MAX;
    ctx->stages[i].max = 0;
    ctx->stages[i].sum = 0;
    ctx->marks[i] = 0;
  }
}

static inline void hqlc_bench_print(const hqlc_bench_ctx *ctx, int cpu_mhz) {
  printf("\n%-20s %10s %10s %10s %10s\n",
         "Stage", "Avg cy", "Min cy", "Max cy", "Avg us");
  for (int i = 0; i < HQLC_BENCH_N_STAGES; i++) {
    const hqlc_bench_stage *s = &ctx->stages[i];
    if (s->count == 0) {
      continue;
    }
    uint32_t avg = (uint32_t)(s->sum / s->count);
    uint32_t avg_us = avg / (uint32_t)cpu_mhz;
    printf("%-20s %10lu %10lu %10lu %10lu\n",
           s->name,
           (unsigned long)avg,
           (unsigned long)s->min,
           (unsigned long)s->max,
           (unsigned long)avg_us);
  }
  printf("\n");
}

// Instrumentation macros
#define HQLC_BENCH_BEGIN(stage)                        \
  do {                                                 \
    if (hqlc_bench)                                    \
      hqlc_bench->marks[stage] = hqlc_bench_cycles();  \
  } while (0)

#define HQLC_BENCH_END(stage)                                              \
  do {                                                                     \
    if (hqlc_bench) {                                                      \
      uint32_t _elapsed = hqlc_bench_cycles() - hqlc_bench->marks[stage];  \
      hqlc_bench_stage *_s = &hqlc_bench->stages[stage];                   \
      _s->count++;                                                         \
      _s->sum += _elapsed;                                                 \
      if (_elapsed < _s->min)                                              \
        _s->min = _elapsed;                                                \
      if (_elapsed > _s->max)                                              \
        _s->max = _elapsed;                                                \
    }                                                                      \
  } while (0)

#endif // HQLC_BENCH_IMPL_H
