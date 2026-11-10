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

#ifdef __cplusplus
}
#endif

#endif // HQLC_PCM_H
