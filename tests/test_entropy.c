#include "unity.h"

#include <stdint.h>
#include <string.h>

#include "entropy.h"
#include "hqlc.h"
#include "psy.h"

void setUp(void) {}
void tearDown(void) {}

void test_zigzag_roundtrip(void) {
  for (int32_t v = -1000; v <= 1000; v++) {
    uint32_t u = zigzag_enc(v);
    TEST_ASSERT_EQUAL_INT32(v, zigzag_dec(u));
  }
}

void test_bitstream_roundtrip(void) {
  uint8_t buf[16];
  hqlc_bitwriter w;
  bw_init(&w, buf, sizeof(buf));

  // Write a mix of field sizes spanning byte boundaries
  bw_write(&w, 72, 7);
  bw_write(&w, 1, 1);
  bw_write(&w, 5, 3);
  bw_write(&w, 0xABCDE, 20);
  bw_flush(&w);

  hqlc_bitreader r;
  br_init(&r, buf, bw_bytes(&w));
  TEST_ASSERT_EQUAL_UINT32(72, br_read(&r, 7));
  TEST_ASSERT_EQUAL_UINT32(1, br_read(&r, 1));
  TEST_ASSERT_EQUAL_UINT32(5, br_read(&r, 3));
  TEST_ASSERT_EQUAL_UINT32(0xABCDE, br_read(&r, 20));
}

void test_rice_roundtrip(void) {
  uint8_t buf[256];
  hqlc_bitwriter w;
  bw_init(&w, buf, sizeof(buf));

  uint32_t vals[] = {0, 1, 2, 3, 7, 15, 31, 63, 100};
  int nvals = sizeof(vals) / sizeof(vals[0]);

  for (int k = 0; k <= 6; k++) {
    for (int j = 0; j < nvals; j++) {
      bw_write_rice(&w, vals[j], k);
    }
  }
  bw_flush(&w);

  hqlc_bitreader r;
  br_init(&r, buf, bw_bytes(&w));
  for (int k = 0; k <= 6; k++) {
    for (int j = 0; j < nvals; j++) {
      TEST_ASSERT_EQUAL_UINT32(vals[j], br_read_rice(&r, k));
    }
  }
}

void test_rans_highlevel_roundtrip(void) {
  int gain_code = 43; // gain=4.0 with GAIN_BIAS=27

  // Pattern with small magnitudes, overflow, and EG(0) values
  int16_t quant[HQLC_FRAME_SAMPLES];
  memset(quant, 0, sizeof(quant));
  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    int16_t pattern[] = {0, 1, -1, 2, -2, 3, -3, 0};
    for (int i = s; i < e; i++) {
      quant[i] = pattern[(i - s) % 8];
    }
  }
  // Add ESC + EG(0) values in band 10
  int s10 = psy_band_edges[10];
  quant[s10 + 0] = 15;  // ESC + EG(0) overflow=0
  quant[s10 + 1] = -20; // ESC + EG(0) overflow=5
  quant[s10 + 2] = -50; // ESC + EG(0) overflow=35
  quant[s10 + 3] = 100; // ESC + EG(0) overflow=85

  uint8_t out[4096];
  size_t len = rans_encode_coeffs(quant, 1, gain_code, out, sizeof(out));
  TEST_ASSERT_GREATER_THAN(0, len);

  int16_t decoded[HQLC_FRAME_SAMPLES];
  rans_decode_coeffs(out, len, decoded, 1, gain_code);

  for (int b = 0; b < PSY_N_BANDS; b++) {
    int s = psy_band_edges[b];
    int e = psy_band_edges[b + 1];
    for (int i = s; i < e; i++) {
      TEST_ASSERT_EQUAL_INT16(quant[i], decoded[i]);
    }
  }
}

void test_rans_corrupt_input_safe(void) {
  // The decoder must never read out of bounds, and must report corruption on corrupt or truncated bitstreams
  int gain_code = 43;
  int16_t quant[HQLC_FRAME_SAMPLES];
  for (int i = 0; i < HQLC_FRAME_SAMPLES; i++) {
    quant[i] = (int16_t)((i % 7) - 3);
  }

  uint8_t buf[4096];
  size_t len = rans_encode_coeffs(quant, 1, gain_code, buf, sizeof(buf));
  TEST_ASSERT_GREATER_THAN(4, len);

  int16_t decoded[HQLC_FRAME_SAMPLES];

  // Hard truncation: the decoder runs out of bytes and must report corruption.
  TEST_ASSERT_FALSE(rans_decode_coeffs(buf, 3, decoded, 1, gain_code));

  // Pure garbage at full length: must return without overreading or hanging.
  uint8_t junk[64];
  memset(junk, 0xA5, sizeof(junk));
  (void)rans_decode_coeffs(junk, sizeof(junk), decoded, 1, gain_code);
}

void test_rans_freq_tables_normalized(void) {
  // Every (alpha x activity) table must be monotone with cf[0] = 0 and
  // cf[MAX_SYM] = RANS_M, or the rANS renormalization is not exact and
  // decode desyncs.
  for (int t = 0; t < RANS_NTABLES; t++) {
    TEST_ASSERT_EQUAL_UINT32(0, rans_cf[t][0]);
    for (int s = 0; s < RANS_MAX_SYM; s++) {
      TEST_ASSERT_TRUE(rans_cf[t][s] <= rans_cf[t][s + 1]);
    }
    TEST_ASSERT_EQUAL_UINT32(RANS_M, rans_cf[t][RANS_MAX_SYM]);
  }
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_zigzag_roundtrip);
  RUN_TEST(test_bitstream_roundtrip);
  RUN_TEST(test_rice_roundtrip);
  RUN_TEST(test_rans_highlevel_roundtrip);
  RUN_TEST(test_rans_corrupt_input_safe);
  RUN_TEST(test_rans_freq_tables_normalized);
  return UNITY_END();
}
