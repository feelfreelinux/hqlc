#include "unity.h"

#include <string.h>

#include "hqlc.h"
#include "psy.h"
#include "quant.h"

void setUp(void) {}
void tearDown(void) {}

void test_forward_scale_decomposes_positive_inverse_step(void) {
  quant_scale scale = quant_forward_scale(30, 27, 5);

  TEST_ASSERT_EQUAL_INT32(319225354, scale.multiplier_q28);
  TEST_ASSERT_EQUAL_INT(43, scale.product_rshift);
}

void test_forward_scale_decomposes_negative_inverse_step(void) {
  quant_scale scale = quant_forward_scale(50, 27, 0);

  TEST_ASSERT_EQUAL_INT32(319225354, scale.multiplier_q28);
  TEST_ASSERT_EQUAL_INT(53, scale.product_rshift);
}

void test_inverse_quantizer_canonicalizes_an_empty_bfp_spectrum(void) {
  int16_t quant[HQLC_FRAME_SAMPLES] = {0};
  int32_t exp_indices[PSY_N_BANDS] = {0};
  int32_t spectrum_data[HQLC_FRAME_SAMPLES];
  memset(spectrum_data, 0x55, sizeof(spectrum_data));
  bfp_i32 spectrum = bfp_i32_view(spectrum_data, HQLC_FRAME_SAMPLES, 99);

  quant_inverse(quant, exp_indices, 27, false, &spectrum);

  int32_t zero[HQLC_FRAME_SAMPLES] = {0};
  TEST_ASSERT_EQUAL_MEMORY(zero, spectrum_data, sizeof(zero));
  TEST_ASSERT_EQUAL_INT(0, spectrum.exp2);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_forward_scale_decomposes_positive_inverse_step);
  RUN_TEST(test_forward_scale_decomposes_negative_inverse_step);
  RUN_TEST(test_inverse_quantizer_canonicalizes_an_empty_bfp_spectrum);
  return UNITY_END();
}
