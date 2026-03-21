// PCM format conversion utilities
#ifndef HQLC_PCM_H
#define HQLC_PCM_H

#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load one interleaved PCM sample as Q31
static inline int32_t pcm_load_q31(const uint8_t *base, hqlc_pcm_format fmt, int idx) {
  if (fmt == HQLC_PCM16) {
    const int16_t *p = (const int16_t *)base;
    return (int32_t)p[idx] << 16;
  }
  const uint8_t *p = base + 3 * idx;
  int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
  if (v & 0x800000) {
    v |= (int32_t)0xFF000000;
  }
  return v << 8;
}

// Load one interleaved PCM sample in native range (no Q31 shift)
static inline int32_t pcm_load_native(const uint8_t *base, hqlc_pcm_format fmt, int idx) {
  if (fmt == HQLC_PCM16) {
    const int16_t *p = (const int16_t *)base;
    return (int32_t)p[idx];
  }
  const uint8_t *p = base + 3 * idx;
  int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
  if (v & 0x800000) {
    v |= (int32_t)0xFF000000;
  }
  return v;
}

// Store a Q31 value to interleaved PCM with rounding and clamping
static inline void
pcm_store_q31(uint8_t *base, hqlc_pcm_format fmt, int idx, int32_t val_q31) {
  if (fmt == HQLC_PCM16) {
    int16_t *p = (int16_t *)base;
    int32_t pcm16 = (val_q31 >> 16) + ((val_q31 >> 15) & 1);
    if (pcm16 > 32767) {
      pcm16 = 32767;
    }
    if (pcm16 < -32768) {
      pcm16 = -32768;
    }
    p[idx] = (int16_t)pcm16;
  } else {
    int32_t v = (val_q31 >> 8) + ((val_q31 >> 7) & 1);
    uint8_t *p = base + 3 * idx;
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
  }
}

// Clamp an int32 to int16 range
static inline int16_t pcm_clamp_i16(int32_t x) {
  if (x > 32767) {
    return 32767;
  }
  if (x < -32768) {
    return -32768;
  }
  return (int16_t)x;
}

#ifdef __cplusplus
}
#endif

#endif // HQLC_PCM_H
