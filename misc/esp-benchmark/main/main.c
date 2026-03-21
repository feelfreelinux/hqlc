#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "hqlc.h"
#include "hqlc_esp_codec.h"

#include "esp_audio_enc.h"
#include "esp_audio_dec.h"
#include "esp_audio_enc_default.h"
#include "esp_audio_dec_default.h"

#include "bench.h"
#include "test_pcm.h"

/* ── Shared buffers ── */

// Max PCM frame across codecs: AAC 1024 * 2ch * 2 = 4096
#define MAX_PCM_FRAME_BYTES 4096
#define MAX_ENC_OUT_BYTES   8192
#define MAX_FRAMES          1024
// 128 KB store for encoded frames (SBC at ~328kbps needs ~68KB for 800 frames)
#define ENC_STORE_BYTES     (128 * 1024)

static uint8_t frame_buf[MAX_PCM_FRAME_BYTES] __attribute__((aligned(4)));
static uint8_t enc_out[MAX_ENC_OUT_BYTES] __attribute__((aligned(4)));
static uint8_t dec_out[MAX_PCM_FRAME_BYTES] __attribute__((aligned(4)));

// Encoded bitstream store — malloc'd from PSRAM at runtime (too large for static DRAM)
static uint8_t *enc_store;
static uint16_t enc_sizes[MAX_FRAMES];

typedef struct {
    const char *name;
    esp_audio_type_t type;
    void *enc_cfg;
    uint32_t enc_cfg_sz;
    void *dec_cfg;
    uint32_t dec_cfg_sz;
    int target_bps;
} codec_entry_t;

typedef struct {
    const char *name;
    int ok;
    int n_frames;
    float frame_ms;
    float actual_kbps;
    uint32_t enc_avg_us;
    uint32_t dec_avg_us;
    uint32_t enc_total_us;
    uint32_t dec_total_us;
} codec_result_t;

static void bench_codec(const codec_entry_t *codec, int cpu_mhz,
                        codec_result_t *result) {
    result->name = codec->name;
    result->ok = 0;

    printf("\n--- %s @ %d kbps ---\n", codec->name, codec->target_bps / 1000);

    bench_ctx bench;
    bench_init(&bench);
    int s_enc = bench_add_stage(&bench, "encode");
    int s_dec = bench_add_stage(&bench, "decode");

    const uint8_t *pcm = (const uint8_t *)test_pcm;
    int in_size = 0, out_size = 0;
    int n_frames = 0;
    int frame_samples = 0;
    float frame_ms = 0;
    uint32_t total_encoded = 0;
    uint32_t store_offset = 0;
    esp_audio_err_t err;

    {
        esp_audio_enc_config_t enc_config = {
            .type = codec->type,
            .cfg = codec->enc_cfg,
            .cfg_sz = codec->enc_cfg_sz,
        };
        esp_audio_enc_handle_t enc_hd = NULL;
        err = esp_audio_enc_open(&enc_config, &enc_hd);
        if (err != ESP_AUDIO_ERR_OK) {
            printf("  ERROR: encoder open failed (%d)\n", err);
            return;
        }

        esp_audio_enc_get_frame_size(enc_hd, &in_size, &out_size);
        if (in_size == 0 || in_size > MAX_PCM_FRAME_BYTES) {
            printf("  ERROR: frame size %d out of range\n", in_size);
            esp_audio_enc_close(enc_hd);
            return;
        }

        int total_pcm_bytes =
            TEST_PCM_FRAMES * HQLC_FRAME_SAMPLES * TEST_PCM_CHANNELS * 2;
        n_frames = total_pcm_bytes / in_size;
        if (n_frames > MAX_FRAMES) n_frames = MAX_FRAMES;
        frame_samples = in_size / (TEST_PCM_CHANNELS * 2);
        frame_ms = (float)frame_samples / 48.0f;

        printf("  Frame: %d samples (%.2f ms), %d bytes PCM\n", frame_samples,
               frame_ms, in_size);
        printf("  Frames: %d\n", n_frames);

        for (int f = 0; f < n_frames; f++) {
            memcpy(frame_buf, pcm + f * in_size, in_size);

            esp_audio_enc_in_frame_t in_frame = {
                .buffer = frame_buf,
                .len = in_size,
            };
            esp_audio_enc_out_frame_t out_frame = {
                .buffer = enc_out,
                .len = MAX_ENC_OUT_BYTES,
                .encoded_bytes = 0,
            };

            bench_begin(&bench);
            err = esp_audio_enc_process(enc_hd, &in_frame, &out_frame);
            bench_end(&bench, s_enc);

            if (err != ESP_AUDIO_ERR_OK) {
                printf("  ERROR: encode frame %d failed (%d)\n", f, err);
                esp_audio_enc_close(enc_hd);
                return;
            }

            // Store encoded frame
            if (store_offset + out_frame.encoded_bytes > ENC_STORE_BYTES) {
                printf("  ERROR: encoded store overflow at frame %d\n", f);
                esp_audio_enc_close(enc_hd);
                return;
            }
            memcpy(enc_store + store_offset, enc_out, out_frame.encoded_bytes);
            enc_sizes[f] = (uint16_t)out_frame.encoded_bytes;
            store_offset += out_frame.encoded_bytes;
            total_encoded += out_frame.encoded_bytes;
        }

        esp_audio_enc_close(enc_hd);
    }

    printf("  Encoded: %u bytes total\n", (unsigned)total_encoded);

    {
        esp_audio_dec_cfg_t dec_config = {
            .type = codec->type,
            .cfg = codec->dec_cfg,
            .cfg_sz = codec->dec_cfg_sz,
        };
        esp_audio_dec_handle_t dec_hd = NULL;
        err = esp_audio_dec_open(&dec_config, &dec_hd);
        if (err != ESP_AUDIO_ERR_OK) {
            printf("  ERROR: decoder open failed (%d)\n", err);
            return;
        }

        uint32_t rd_offset = 0;
        for (int f = 0; f < n_frames; f++) {
            // Copy encoded frame from PSRAM to internal DRAM before timing
            memcpy(enc_out, enc_store + rd_offset, enc_sizes[f]);

            esp_audio_dec_in_raw_t raw = {
                .buffer = enc_out,
                .len = enc_sizes[f],
                .consumed = 0,
                .frame_recover = ESP_AUDIO_DEC_RECOVERY_NONE,
            };
            esp_audio_dec_out_frame_t dec_frame = {
                .buffer = dec_out,
                .len = MAX_PCM_FRAME_BYTES,
                .decoded_size = 0,
            };

            bench_begin(&bench);
            err = esp_audio_dec_process(dec_hd, &raw, &dec_frame);
            bench_end(&bench, s_dec);

            if (err != ESP_AUDIO_ERR_OK) {
                printf("  ERROR: decode frame %d failed (%d)\n", f, err);
                esp_audio_dec_close(dec_hd);
                return;
            }
            rd_offset += enc_sizes[f];
        }

        esp_audio_dec_close(dec_hd);
    }

    /* ── Results ── */
    {
        float actual_kbps = (float)total_encoded * 8.0f * 48000.0f /
                            (n_frames * (float)frame_samples) / 1000.0f;
        uint32_t enc_total_cy = (uint32_t)(bench.stages[s_enc].sum);
        uint32_t dec_total_cy = (uint32_t)(bench.stages[s_dec].sum);
        uint32_t enc_avg_cy = enc_total_cy / n_frames;
        uint32_t dec_avg_cy = dec_total_cy / n_frames;
        uint32_t enc_avg_us = enc_avg_cy / cpu_mhz;
        uint32_t dec_avg_us = dec_avg_cy / cpu_mhz;
        uint32_t enc_total_us = enc_total_cy / cpu_mhz;
        uint32_t dec_total_us = dec_total_cy / cpu_mhz;
        float frame_budget_us = frame_ms * 1000.0f;
        float audio_duration_ms = n_frames * frame_ms;

        printf("  Bitrate: %.1f kbps (%u bytes / %d frames)\n", actual_kbps,
               (unsigned)total_encoded, n_frames);

        bench_print(&bench, cpu_mhz);

        printf("  Audio duration: %.1f ms\n", audio_duration_ms);
        printf("  Encode total:   %lu us (%.1f ms)\n",
               (unsigned long)enc_total_us, enc_total_us / 1000.0f);
        printf("  Decode total:   %lu us (%.1f ms)\n",
               (unsigned long)dec_total_us, dec_total_us / 1000.0f);
        printf("  Frame budget: %.0f us (%.2f ms)\n", frame_budget_us,
               frame_ms);
        printf("  Encode:   %5.1f%% of realtime\n",
               100.0f * enc_avg_us / frame_budget_us);
        printf("  Decode:   %5.1f%% of realtime\n",
               100.0f * dec_avg_us / frame_budget_us);
        result->ok = 1;
        result->n_frames = n_frames;
        result->frame_ms = frame_ms;
        result->actual_kbps = actual_kbps;
        result->enc_avg_us = enc_avg_us;
        result->dec_avg_us = dec_avg_us;
        result->enc_total_us = enc_total_us;
        result->dec_total_us = dec_total_us;
    }
}

static hqlc_esp_enc_cfg_t hqlc_enc_cfg = {
    .channels = TEST_PCM_CHANNELS,
    .sample_rate = 48000,
    .bitrate = 96000,
};
static hqlc_esp_dec_cfg_t hqlc_dec_cfg = {
    .channels = TEST_PCM_CHANNELS,
    .sample_rate = 48000,
};

static esp_opus_enc_config_t opus_enc_cfg = {
    .sample_rate = 48000,
    .channel = TEST_PCM_CHANNELS,
    .bits_per_sample = 16,
    .bitrate = 96000,
    .frame_duration = ESP_OPUS_ENC_FRAME_DURATION_20_MS,
    .application_mode = ESP_OPUS_ENC_APPLICATION_AUDIO,
    .complexity = 5,
    .enable_fec = false,
    .enable_dtx = false,
    .enable_vbr = false,
};
static esp_opus_dec_cfg_t opus_dec_cfg = {
    .sample_rate = 48000,
    .channel = TEST_PCM_CHANNELS,
    .frame_duration = ESP_OPUS_DEC_FRAME_DURATION_20_MS,
    .self_delimited = false,
};

static esp_lc3_enc_config_t lc3_enc_cfg = {
    .sample_rate = 48000,
    .bits_per_sample = 16,
    .channel = TEST_PCM_CHANNELS,
    .frame_dms = 100,
    .nbyte = 60,
    .len_prefixed = false,
};
static esp_lc3_dec_cfg_t lc3_dec_cfg = {
    .sample_rate = 48000,
    .channel = TEST_PCM_CHANNELS,
    .bits_per_sample = 16,
    .frame_dms = 100,
    .nbyte = 60,
    .is_cbr = true,
    .len_prefixed = false,
    .enable_plc = false,
};

// AAC: min stereo@48kHz is 118kbps; use 128kbps
static esp_aac_enc_config_t aac_enc_cfg = {
    .sample_rate = 48000,
    .channel = TEST_PCM_CHANNELS,
    .bits_per_sample = 16,
    .bitrate = 128000,
    .adts_used = true,
};
// AAC decoder auto-detects from ADTS header (cfg=NULL)

// Opus complexity 1: same as above but minimal CPU
static esp_opus_enc_config_t opus_c1_enc_cfg = {
    .sample_rate = 48000,
    .channel = TEST_PCM_CHANNELS,
    .bits_per_sample = 16,
    .bitrate = 96000,
    .frame_duration = ESP_OPUS_ENC_FRAME_DURATION_20_MS,
    .application_mode = ESP_OPUS_ENC_APPLICATION_AUDIO,
    .complexity = 1,
    .enable_fec = false,
    .enable_dtx = false,
    .enable_vbr = false,
};

// SBC: joint stereo, bitpool 53 (~328 kbps), 16 blocks, 8 subbands
static esp_sbc_enc_config_t sbc_enc_cfg = {
    .sbc_mode = ESP_SBC_MODE_STD,
    .allocation_method = ESP_SBC_ALLOC_LOUDNESS,
    .ch_mode = ESP_SBC_CH_MODE_JOINT_STEREO,
    .sample_rate = 48000,
    .bits_per_sample = 16,
    .bitpool = 53,
    .block_length = 16,
    .sub_bands_num = 8,
};
static esp_sbc_dec_cfg_t sbc_dec_cfg = {
    .sbc_mode = ESP_SBC_MODE_STD,
    .ch_num = TEST_PCM_CHANNELS,
    .enable_plc = false,
};

#define N_CODECS 6
static codec_entry_t codecs[N_CODECS];
static codec_result_t results[N_CODECS];
static int n_codecs;

#define BENCH_TASK_STACK (48 * 1024)

static void bench_task(void *arg) {
    int cpu_mhz = (int)(intptr_t)arg;

    for (int i = 0; i < n_codecs; i++) {
        bench_codec(&codecs[i], cpu_mhz, &results[i]);
    }

    float audio_ms = TEST_PCM_FRAMES * HQLC_FRAME_SAMPLES / 48.0f;
    printf("\n=== Summary (audio: %.0f ms) ===\n", audio_ms);
    printf("%-10s %7s %6s %9s %9s %9s %9s %8s %8s\n",
           "Codec", "kbps", "Frames", "Enc ms", "Dec ms", "Enc/frm",
           "Dec/frm", "Enc RT%", "Dec RT%");
    printf("%-10s %7s %6s %9s %9s %9s %9s %8s %8s\n",
           "----------", "-------", "------", "---------", "---------",
           "---------", "---------", "----------", "----------");

    for (int i = 0; i < n_codecs; i++) {
        codec_result_t *r = &results[i];
        if (!r->ok) {
            printf("%-10s  (failed)\n", r->name);
            continue;
        }
        float budget_us = r->frame_ms * 1000.0f;
        float enc_rt = 100.0f * r->enc_avg_us / budget_us;
        float dec_rt = 100.0f * r->dec_avg_us / budget_us;
        printf("%-10s %7.1f %6d %9.1f %9.1f %7lu %7lu %7.1f%% %7.1f%%\n",
               r->name, r->actual_kbps, r->n_frames,
               r->enc_total_us / 1000.0f, r->dec_total_us / 1000.0f,
               (unsigned long)r->enc_avg_us, (unsigned long)r->dec_avg_us,
               enc_rt, dec_rt);
    }

    printf("\nNote: MP3 omitted (no encoder in esp_audio_codec)\n");
    printf("Note: AAC min stereo bitrate at 48kHz is 118kbps (using 128kbps)\n");

    printf("\nDone. Halting.\n");
    vTaskDelete(NULL);
}

void app_main(void) {
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);
    int cpu_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;

    printf("\n=== Multi-Codec ESP32 Benchmark ===\n");
    printf("  Target: %s, %d core(s), %d MHz\n", CONFIG_IDF_TARGET,
           chip_info.cores, cpu_mhz);
    printf("  Test clip: %dch, %d samples (%d HQLC frames)\n",
           TEST_PCM_CHANNELS, TEST_PCM_SAMPLES, TEST_PCM_FRAMES);
    printf("  HQLC encoder: %u bytes, scratch: %u bytes\n",
           (unsigned)hqlc_encoder_size(), (unsigned)hqlc_encoder_scratch_size());
    printf("  HQLC decoder: %u bytes, scratch: %u bytes\n",
           (unsigned)hqlc_decoder_size(), (unsigned)hqlc_decoder_scratch_size());
    printf("  HQLC allocs use internal DRAM (heap_caps)\n");

    // Allocate encoded frame store from PSRAM (too large for internal DRAM)
    enc_store = heap_caps_malloc(ENC_STORE_BYTES, MALLOC_CAP_SPIRAM);
    if (!enc_store) {
        printf("ERROR: failed to allocate enc_store from PSRAM\n");
        return;
    }

    // Register all codecs
    esp_audio_enc_register_default();
    esp_audio_dec_register_default();

    esp_audio_type_t hqlc_type = hqlc_esp_register();
    if (hqlc_type == ESP_AUDIO_TYPE_UNSUPPORT) {
        printf("ERROR: failed to register HQLC codec\n");
        return;
    }

    /* ── Build codec table ── */

    codecs[0] = (codec_entry_t){
        .name = "HQLC",        .type = hqlc_type,
        .enc_cfg = &hqlc_enc_cfg, .enc_cfg_sz = sizeof(hqlc_enc_cfg),
        .dec_cfg = &hqlc_dec_cfg, .dec_cfg_sz = sizeof(hqlc_dec_cfg),
        .target_bps = 96000,
    };
    codecs[1] = (codec_entry_t){
        .name = "Opus",        .type = ESP_AUDIO_TYPE_OPUS,
        .enc_cfg = &opus_enc_cfg, .enc_cfg_sz = sizeof(opus_enc_cfg),
        .dec_cfg = &opus_dec_cfg, .dec_cfg_sz = sizeof(opus_dec_cfg),
        .target_bps = 96000,
    };
    codecs[2] = (codec_entry_t){
        .name = "LC3",         .type = ESP_AUDIO_TYPE_LC3,
        .enc_cfg = &lc3_enc_cfg, .enc_cfg_sz = sizeof(lc3_enc_cfg),
        .dec_cfg = &lc3_dec_cfg, .dec_cfg_sz = sizeof(lc3_dec_cfg),
        .target_bps = 96000,
    };
    codecs[3] = (codec_entry_t){
        .name = "AAC",         .type = ESP_AUDIO_TYPE_AAC,
        .enc_cfg = &aac_enc_cfg, .enc_cfg_sz = sizeof(aac_enc_cfg),
        .dec_cfg = NULL,         .dec_cfg_sz = 0,
        .target_bps = 128000,
    };
    codecs[4] = (codec_entry_t){
        .name = "Opus c1",    .type = ESP_AUDIO_TYPE_OPUS,
        .enc_cfg = &opus_c1_enc_cfg, .enc_cfg_sz = sizeof(opus_c1_enc_cfg),
        .dec_cfg = &opus_dec_cfg,     .dec_cfg_sz = sizeof(opus_dec_cfg),
        .target_bps = 96000,
    };
    codecs[5] = (codec_entry_t){
        .name = "SBC",        .type = ESP_AUDIO_TYPE_SBC,
        .enc_cfg = &sbc_enc_cfg, .enc_cfg_sz = sizeof(sbc_enc_cfg),
        .dec_cfg = &sbc_dec_cfg, .dec_cfg_sz = sizeof(sbc_dec_cfg),
        .target_bps = 328000,
    };
    n_codecs = N_CODECS;
    memset(results, 0, sizeof(results));

    // Launch benchmark on a dedicated task with a large stack
    xTaskCreatePinnedToCore(bench_task, "bench", BENCH_TASK_STACK,
                            (void *)(intptr_t)cpu_mhz, 5, NULL, 0);
}
