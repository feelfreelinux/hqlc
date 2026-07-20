#include "unity.h"

#include <math.h>
#include <string.h>

#include "hqlc.h"
#include "mdct.h"

void setUp(void) {}
void tearDown(void) {}

#define N      MDCT_N
#define FRAMES 6

static uint8_t scratch[MDCT_SCRATCH_BYTES] __attribute__((aligned(8)));

// Push FRAMES mono PCM16 frames through forward + inverse-OLA
static void transform_roundtrip(const int16_t *sig, int want, int16_t *out_frame) {
  mdct_ola_state ola;
  mdct_ola_init(&ola);
  int16_t prev[N] = {0};

  for (int f = 0; f < FRAMES; f++) {
    const int16_t *curr = &sig[f * N];
    int32_t spec[N];
    int loss;
    TEST_ASSERT_EQUAL_INT(HQLC_OK,
                          mdct_forward((const uint8_t *)prev,
                                       (const uint8_t *)curr,
                                       (size_t)N * sizeof(int16_t),
                                       HQLC_PCM16,
                                       1,
                                       0,
                                       spec,
                                       N,
                                       scratch,
                                       sizeof(scratch),
                                       &loss));
    int16_t dec[N];
    TEST_ASSERT_EQUAL_INT(HQLC_OK,
                          mdct_inverse_ola(spec,
                                           N,
                                           loss,
                                           &ola,
                                           (uint8_t *)dec,
                                           HQLC_PCM16,
                                           1,
                                           0,
                                           scratch,
                                           sizeof(scratch)));
    if (f == want) {
      memcpy(out_frame, dec, (size_t)N * sizeof(int16_t));
    }
    memcpy(prev, curr, (size_t)N * sizeof(int16_t));
  }
}

void test_mdct_reconstructs_sine(void) {
  int16_t sig[FRAMES * N];

  // Generate a sine wave input signal
  for (int i = 0; i < FRAMES * N; i++) {
    sig[i] = (int16_t)(16000.0 * sin(2.0 * M_PI * 1000.0 * i / 48000.0));
  }

  int16_t dec[N];
  transform_roundtrip(sig, 4, dec); // decoded frame 4 == input frame 3

  const int16_t *ref = &sig[3 * N];
  double sp = 0.0, np = 0.0;
  for (int i = 0; i < N; i++) {
    sp += (double)ref[i] * ref[i];
    np += (double)(ref[i] - dec[i]) * (ref[i] - dec[i]);
  }
  double snr = 10.0 * log10(sp / (np < 1.0 ? 1.0 : np));
  TEST_ASSERT_GREATER_THAN_FLOAT(60.0f, (float)snr);
}

void test_mdct_reconstructs_dc(void) {
  int16_t sig[FRAMES * N];
  for (int i = 0; i < FRAMES * N; i++) {
    sig[i] = 0x4000;
  }

  int16_t dec[N];
  transform_roundtrip(sig, 4, dec);

  for (int i = 0; i < N; i++) {
    TEST_ASSERT_INT16_WITHIN(4, 0x4000, dec[i]);
  }
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_mdct_reconstructs_sine);
  RUN_TEST(test_mdct_reconstructs_dc);
  return UNITY_END();
}
