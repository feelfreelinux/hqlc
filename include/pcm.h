// PCM format conversion utilities
#ifndef HQLC_PCM_H
#define HQLC_PCM_H

#include <stdint.h>

#include "fxp.h"
#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load one interleaved PCM sample as Q31.
 *
 * @param base PCM buffer base pointer.
 * @param fmt PCM sample format.
 * @param idx Interleaved sample index.
 * @return Sample converted to Q31.
 */
static inline int32_t pcm_load_q31(const uint8_t *base, hqlc_pcm_format fmt, int idx) {
  if (fmt == HQLC_PCM16) {
    const int16_t *p = (const int16_t *)base;
    return (int32_t)((uint32_t)p[idx] << 16);
  }
  const uint8_t *p = base + 3 * idx;
  int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16);
  if (v & 0x800000) {
    v |= (int32_t)0xFF000000;
  }
  return (int32_t)((uint32_t)v << 8);
}

/**
 * @brief Load one interleaved PCM sample without Q31 scaling.
 *
 * @param base PCM buffer base pointer.
 * @param fmt PCM sample format.
 * @param idx Interleaved sample index.
 * @return Sample in its native signed integer range.
 */
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

/**
 * @brief Store a Q31 value as interleaved PCM.
 *
 * The stored value is rounded to the target format. PCM16 output is clamped to
 * the signed 16-bit range.
 *
 * @param base PCM buffer base pointer.
 * @param fmt PCM sample format.
 * @param idx Interleaved sample index.
 * @param val_q31 Q31 sample value to store.
 */
static inline void
pcm_store_q31(uint8_t *base, hqlc_pcm_format fmt, int idx, int32_t val_q31) {
  if (fmt == HQLC_PCM16) {
    int16_t *p = (int16_t *)base;
    int32_t pcm16 = fxp_shr_rnd_i32(val_q31, 16);
    p[idx] = (int16_t)fxp_clamp_i32(pcm16, INT16_MIN, INT16_MAX);
  } else {
    int32_t v = fxp_shr_rnd_i32(val_q31, 8);
    uint8_t *p = base + 3 * idx;
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
  }
}

#ifdef __cplusplus
}
#endif

#endif // HQLC_PCM_H
