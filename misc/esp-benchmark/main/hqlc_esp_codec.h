#pragma once

#include <stdint.h>
#include "esp_audio_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// HQLC encoder config for esp_audio_codec framework
typedef struct {
    uint8_t channels;
    uint32_t sample_rate;
    uint32_t bitrate;
} hqlc_esp_enc_cfg_t;

// HQLC decoder config for esp_audio_codec framework
typedef struct {
    uint8_t channels;
    uint32_t sample_rate;
} hqlc_esp_dec_cfg_t;

// Register HQLC encoder and decoder with esp_audio_codec.
// Returns the assigned custom type, or ESP_AUDIO_TYPE_UNSUPPORT on failure.
esp_audio_type_t hqlc_esp_register(void);

#ifdef __cplusplus
}
#endif
