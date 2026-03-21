#pragma once

#include <stdint.h>
#include <stdio.h>

// Xtensa cycle counter
static inline uint32_t cycles_now(void) {
  uint32_t c;
  __asm__ __volatile__("rsr %0, ccount" : "=a"(c));
  return c;
}

#define BENCH_MAX_STAGES 16

typedef struct {
  const char *name;
  uint32_t count;
  uint32_t min;
  uint32_t max;
  uint64_t sum;
} bench_stage;

typedef struct {
  bench_stage stages[BENCH_MAX_STAGES];
  int n_stages;
  uint32_t _mark; /* temp for begin/end pairs */
} bench_ctx;

static inline void bench_init(bench_ctx *ctx) {
  ctx->n_stages = 0;
  ctx->_mark = 0;
}

static inline int bench_add_stage(bench_ctx *ctx, const char *name) {
  int idx = ctx->n_stages++;
  ctx->stages[idx].name = name;
  ctx->stages[idx].count = 0;
  ctx->stages[idx].min = UINT32_MAX;
  ctx->stages[idx].max = 0;
  ctx->stages[idx].sum = 0;
  return idx;
}

static inline void bench_begin(bench_ctx *ctx) {
  ctx->_mark = cycles_now();
}

static inline void bench_end(bench_ctx *ctx, int stage_idx) {
  uint32_t elapsed = cycles_now() - ctx->_mark;
  bench_stage *s = &ctx->stages[stage_idx];
  s->count++;
  s->sum += elapsed;
  if (elapsed < s->min) {
    s->min = elapsed;
  }
  if (elapsed > s->max) {
    s->max = elapsed;
  }
}

static inline void bench_print(const bench_ctx *ctx, int cpu_mhz) {
  printf(
      "\n%-20s %10s %10s %10s %10s\n", "Stage", "Avg cy", "Min cy", "Max cy", "Avg us");
  printf("%-20s %10s %10s %10s %10s\n",
         "--------------------",
         "----------",
         "----------",
         "----------",
         "----------");
  for (int i = 0; i < ctx->n_stages; i++) {
    const bench_stage *s = &ctx->stages[i];
    if (s->count == 0) {
      continue;
    }
    uint32_t avg = (uint32_t)(s->sum / s->count);
    uint32_t avg_us = avg / cpu_mhz;
    printf("%-20s %10lu %10lu %10lu %10lu\n", s->name, avg, s->min, s->max, avg_us);
  }
  printf("\n");
}
