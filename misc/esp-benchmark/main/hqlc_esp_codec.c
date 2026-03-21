#include "hqlc_esp_codec.h"
#include "hqlc.h"
#include "esp_audio_enc_reg.h"
#include "esp_audio_dec_reg.h"
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"

// Force internal DRAM for codec state and scratch (avoid PSRAM)
#define INTERNAL_MALLOC(sz) heap_caps_malloc(sz, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define INTERNAL_CALLOC(n, sz) heap_caps_calloc(n, sz, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)

typedef struct {
    hqlc_encoder *enc;
    void *scratch;
    uint8_t channels;
    uint32_t sample_rate;
    uint32_t bitrate;
} hqlc_enc_ctx_t;

static esp_audio_err_t hqlc_enc_get_frame_info(void *cfg,
                                                esp_audio_enc_frame_info_t *info) {
    hqlc_esp_enc_cfg_t *c = (hqlc_esp_enc_cfg_t *)cfg;
    if (!c || !info) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    info->in_frame_size = HQLC_FRAME_SAMPLES * c->channels * 2;
    info->in_frame_align = 4;
    info->out_frame_size = HQLC_MAX_FRAME_BYTES;
    info->out_frame_align = 1;
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_enc_open(void *cfg, uint32_t cfg_sz,
                                     void **enc_hd) {
    if (!cfg || !enc_hd) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_esp_enc_cfg_t *c = (hqlc_esp_enc_cfg_t *)cfg;

    hqlc_enc_ctx_t *ctx = INTERNAL_CALLOC(1, sizeof(*ctx));
    if (!ctx) return ESP_AUDIO_ERR_MEM_LACK;

    ctx->enc = INTERNAL_MALLOC(hqlc_encoder_size());
    ctx->scratch = INTERNAL_MALLOC(hqlc_encoder_scratch_size());
    if (!ctx->enc || !ctx->scratch) goto fail_mem;

    ctx->channels = c->channels;
    ctx->sample_rate = c->sample_rate;
    ctx->bitrate = c->bitrate;

    hqlc_encoder_config hcfg = {
        .channels = c->channels,
        .sample_rate = c->sample_rate,
        .mode = HQLC_MODE_RC,
        .bitrate = c->bitrate,
    };
    if (hqlc_encoder_init(ctx->enc, &hcfg) != HQLC_OK) goto fail_mem;

    *enc_hd = ctx;
    return ESP_AUDIO_ERR_OK;

fail_mem:
    free(ctx->scratch);
    free(ctx->enc);
    free(ctx);
    return ESP_AUDIO_ERR_MEM_LACK;
}

static esp_audio_err_t hqlc_enc_set_bitrate(void *enc_hd, int bitrate) {
    (void)enc_hd;
    (void)bitrate;
    return ESP_AUDIO_ERR_NOT_SUPPORT;
}

static esp_audio_err_t hqlc_enc_get_info(void *enc_hd,
                                         esp_audio_enc_info_t *info) {
    if (!enc_hd || !info) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_enc_ctx_t *ctx = (hqlc_enc_ctx_t *)enc_hd;
    info->sample_rate = ctx->sample_rate;
    info->channel = ctx->channels;
    info->bits_per_sample = 16;
    info->bitrate = ctx->bitrate;
    info->codec_spec_info = NULL;
    info->spec_info_len = 0;
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_enc_get_frame_size(void *enc_hd, int *in_size,
                                               int *out_size) {
    if (!enc_hd || !in_size || !out_size) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_enc_ctx_t *ctx = (hqlc_enc_ctx_t *)enc_hd;
    *in_size = HQLC_FRAME_SAMPLES * ctx->channels * 2;
    *out_size = HQLC_MAX_FRAME_BYTES;
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_enc_process(void *enc_hd,
                                        esp_audio_enc_in_frame_t *in_frame,
                                        esp_audio_enc_out_frame_t *out_frame) {
    if (!enc_hd || !in_frame || !out_frame) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_enc_ctx_t *ctx = (hqlc_enc_ctx_t *)enc_hd;

    int frame_bytes = HQLC_FRAME_SAMPLES * ctx->channels * 2;
    int n_frames = in_frame->len / frame_bytes;
    if (n_frames == 0) return ESP_AUDIO_ERR_DATA_LACK;

    uint32_t total_out = 0;
    for (int i = 0; i < n_frames; i++) {
        size_t out_len = 0;
        hqlc_error err = hqlc_encode_frame(
            ctx->enc, in_frame->buffer + i * frame_bytes, HQLC_PCM16,
            out_frame->buffer + total_out, out_frame->len - total_out, &out_len,
            ctx->scratch);
        if (err != HQLC_OK) return ESP_AUDIO_ERR_FAIL;
        total_out += out_len;
    }
    out_frame->encoded_bytes = total_out;
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_enc_reset(void *enc_hd) {
    (void)enc_hd;
    return ESP_AUDIO_ERR_NOT_SUPPORT;
}

static void hqlc_enc_close(void *enc_hd) {
    if (!enc_hd) return;
    hqlc_enc_ctx_t *ctx = (hqlc_enc_ctx_t *)enc_hd;
    free(ctx->scratch);
    free(ctx->enc);
    free(ctx);
}

typedef struct {
    hqlc_decoder *dec;
    void *scratch;
    uint8_t channels;
    uint32_t sample_rate;
} hqlc_dec_ctx_t;

static esp_audio_err_t hqlc_dec_open(void *cfg, uint32_t cfg_sz,
                                     void **decoder) {
    if (!cfg || !decoder) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_esp_dec_cfg_t *c = (hqlc_esp_dec_cfg_t *)cfg;

    hqlc_dec_ctx_t *ctx = INTERNAL_CALLOC(1, sizeof(*ctx));
    if (!ctx) return ESP_AUDIO_ERR_MEM_LACK;

    ctx->dec = INTERNAL_MALLOC(hqlc_decoder_size());
    ctx->scratch = INTERNAL_MALLOC(hqlc_decoder_scratch_size());
    if (!ctx->dec || !ctx->scratch) goto fail_mem;

    ctx->channels = c->channels;
    ctx->sample_rate = c->sample_rate;

    if (hqlc_decoder_init(ctx->dec, c->channels, c->sample_rate) != HQLC_OK)
        goto fail_mem;

    *decoder = ctx;
    return ESP_AUDIO_ERR_OK;

fail_mem:
    free(ctx->scratch);
    free(ctx->dec);
    free(ctx);
    return ESP_AUDIO_ERR_MEM_LACK;
}

static esp_audio_err_t hqlc_dec_decode(void *decoder,
                                       esp_audio_dec_in_raw_t *raw,
                                       esp_audio_dec_out_frame_t *frame,
                                       esp_audio_dec_info_t *info) {
    if (!decoder || !raw || !frame) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_dec_ctx_t *ctx = (hqlc_dec_ctx_t *)decoder;

    int frame_pcm_bytes = HQLC_FRAME_SAMPLES * ctx->channels * 2;
    if (frame->len < (uint32_t)frame_pcm_bytes) {
        frame->needed_size = frame_pcm_bytes;
        return ESP_AUDIO_ERR_BUFF_NOT_ENOUGH;
    }

    hqlc_error err = hqlc_decode_frame(ctx->dec, raw->buffer, raw->len,
                                       frame->buffer, HQLC_PCM16, ctx->scratch);
    if (err != HQLC_OK) return ESP_AUDIO_ERR_FAIL;

    raw->consumed = raw->len;
    frame->decoded_size = frame_pcm_bytes;

    if (info) {
        info->sample_rate = ctx->sample_rate;
        info->bits_per_sample = 16;
        info->channel = ctx->channels;
        info->frame_size = frame_pcm_bytes;
        info->bitrate = 0;
    }
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_dec_reset(void *decoder) {
    if (!decoder) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_dec_ctx_t *ctx = (hqlc_dec_ctx_t *)decoder;
    hqlc_decoder_reset(ctx->dec);
    return ESP_AUDIO_ERR_OK;
}

static esp_audio_err_t hqlc_dec_close(void *decoder) {
    if (!decoder) return ESP_AUDIO_ERR_INVALID_PARAMETER;
    hqlc_dec_ctx_t *ctx = (hqlc_dec_ctx_t *)decoder;
    free(ctx->scratch);
    free(ctx->dec);
    free(ctx);
    return ESP_AUDIO_ERR_OK;
}

/* ── Registration ── */
static const esp_audio_enc_ops_t hqlc_enc_ops = {
    .get_frame_info_by_cfg = hqlc_enc_get_frame_info,
    .open = hqlc_enc_open,
    .set_bitrate = hqlc_enc_set_bitrate,
    .get_info = hqlc_enc_get_info,
    .get_frame_size = hqlc_enc_get_frame_size,
    .process = hqlc_enc_process,
    .reset = hqlc_enc_reset,
    .close = hqlc_enc_close,
};

static const esp_audio_dec_ops_t hqlc_dec_ops = {
    .open = hqlc_dec_open,
    .decode = hqlc_dec_decode,
    .reset = hqlc_dec_reset,
    .close = hqlc_dec_close,
};

esp_audio_type_t hqlc_esp_register(void) {
    esp_audio_type_t type = esp_audio_enc_get_avail_type();
    if (type == ESP_AUDIO_TYPE_UNSUPPORT) return type;

    if (esp_audio_enc_register(type, &hqlc_enc_ops) != ESP_AUDIO_ERR_OK) {
        return ESP_AUDIO_TYPE_UNSUPPORT;
    }
    if (esp_audio_dec_register(type, &hqlc_dec_ops) != ESP_AUDIO_ERR_OK) {
        esp_audio_enc_unregister(type);
        return ESP_AUDIO_TYPE_UNSUPPORT;
    }
    return type;
}
