#include "unity.h"

#include <string.h>

#include "hqlc.h"
#include "tns.h"

void setUp(void) {}
void tearDown(void) {}

void test_zero_coefficient_filters_preserve_spectrum_and_exponent(void) {
  int32_t spectrum_data[HQLC_FRAME_SAMPLES] = {0};
  int32_t original[HQLC_FRAME_SAMPLES];
  int32_t k_q30[] = {0};
  spectrum_data[HQLC_FRAME_SAMPLES / 2] = 0x40000000;
  spectrum_data[HQLC_FRAME_SAMPLES / 2 + 7] = -0x20000000;
  memcpy(original, spectrum_data, sizeof(original));

  bfp_i32 spectrum = bfp_i32_view(spectrum_data, HQLC_FRAME_SAMPLES, 7);
  tns_apply_analysis_filter(&spectrum, k_q30, 1);
  TEST_ASSERT_EQUAL_MEMORY(original, spectrum_data, sizeof(original));
  TEST_ASSERT_EQUAL_INT(7, spectrum.exp2);

  tns_apply_synthesis_filter(&spectrum, k_q30, 1);
  TEST_ASSERT_EQUAL_MEMORY(original, spectrum_data, sizeof(original));
  TEST_ASSERT_EQUAL_INT(7, spectrum.exp2);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_zero_coefficient_filters_preserve_spectrum_and_exponent);
  return UNITY_END();
}
