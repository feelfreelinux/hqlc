#ifndef HQLC_H
#define HQLC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define HQLC_FRAME_SAMPLES 512
#define HQLC_BLOCK_SAMPLES 1024
#define HQLC_MAX_CHANNELS  2
#define HQLC_SAMPLE_RATE   48000

// Upper bound on compressed frame size
#define HQLC_MAX_FRAME_BYTES 2048

// Error code enumeration
typedef enum {
  HQLC_OK = 0,                   /**< Success */
  HQLC_ERR_INVALID_ARG,          /**< NULL pointer or out-of-range parameter */
  HQLC_ERR_UNSUPPORTED_RATE,     /**< sample_rate != 48000 */
  HQLC_ERR_UNSUPPORTED_CHANNELS, /**< channels < 1 or > HQLC_MAX_CHANNELS */
  HQLC_ERR_BUFFER_TOO_SMALL,     /**< output buffer capacity insufficient */
  HQLC_ERR_BITSTREAM_CORRUPT,    /**< decoder encountered invalid bitstream */
} hqlc_error;

// PCM format enumeration
typedef enum {
  HQLC_PCM16 = 0, /**< 16-bit signed, little-endian */
  HQLC_PCM24 = 1, /**< 24-bit signed, little-endian, packed (3 bytes/sample) */
} hqlc_pcm_format;

/**
 * @brief Returns the number of bytes per interleaved frame for the given PCM
 * format
 *
 * @param channels Number of channels
 * @param fmt PCM format
 * @return Number of bytes per frame, or 0 if format is invalid
 */
static inline size_t hqlc_frame_pcm_bytes(uint8_t channels, hqlc_pcm_format fmt) {
  switch (fmt) {
  case HQLC_PCM16:
    return (size_t)HQLC_FRAME_SAMPLES * channels * 2;
  case HQLC_PCM24:
    return (size_t)HQLC_FRAME_SAMPLES * channels * 3;
  default:
    return 0;
  }
}

// Encoder mode enumeration
typedef enum {
  HQLC_MODE_RC = 0, /**< rate-controlled (target bitrate) */
  HQLC_MODE_FIXED,  /**< fixed gain                       */
} hqlc_mode;

// Encoder configuration
typedef struct {
  uint8_t channels;     /**< number of channels */
  uint32_t sample_rate; /**< sample rate in Hz (only 48000 supported) */
  hqlc_mode mode;       /**< encoder mode */
  union {
    uint32_t bitrate; /* HQLC_MODE_RC: target bits per second  */
    float gain;       /* HQLC_MODE_FIXED: quantizer gain value       */
  };
} hqlc_encoder_config;

// Fwd defs
typedef struct hqlc_encoder hqlc_encoder;
typedef struct hqlc_decoder hqlc_decoder;

// Returns size of hqlc_encoder struct in bytes
size_t hqlc_encoder_size(void);

// Returns size of hqlc_encoder scratch buffer in bytes
size_t hqlc_encoder_scratch_size(void);

// Returns size of hqlc_decoder struct in bytes
size_t hqlc_decoder_size(void);

// Returns size of hqlc_decoder scratch buffer in bytes
size_t hqlc_decoder_scratch_size(void);

/**
 * @brief Initialize an hqlc_encoder instance.
 *
 * @param enc Pointer to the hqlc_encoder struct to initialize, should be
 *            allocated by the caller with hqlc_encoder_size() bytes.
 * @param cfg Pointer to the encoder configuration.
 * @return hqlc_error HQLC_OK on success, or an error code on failure.
 */
hqlc_error hqlc_encoder_init(hqlc_encoder *enc, const hqlc_encoder_config *cfg);

/**
 * @brief Encode a single frame of PCM data
 *
 * @param enc Pointer to the hqlc_encoder struct.
 * @param pcm Pointer to the mono / interleaved PCM samples to encode
 * @param fmt The PCM format of the input samples.
 * @param out Destination buffer for the compressed frame.
 * @param out_cap Capacity of the `out` buffer in bytes.
 * @param out_len Receives the actual compressed frame size in bytes.
 * @param scratch Temporary workspace buffer (hqlc_encoder_scratch_size()
 * bytes).
 * @return hqlc_error HQLC_OK on success, or an error code on failure.
 */
hqlc_error hqlc_encode_frame(hqlc_encoder *enc,
                             const uint8_t *pcm,
                             hqlc_pcm_format fmt,
                             uint8_t *out,
                             size_t out_cap,
                             size_t *out_len,
                             void *scratch);

/**
 * @brief Initialize an hqlc_decoder instance.
 *
 * @param dec Pointer to the hqlc_decoder struct to initialize, should be
 *            allocated by the caller with hqlc_decoder_size() bytes.
 * @param channels Number of channels to decode.
 * @param sample_rate Sample rate of the output PCM.
 * @return hqlc_error HQLC_OK on success, or an error code on failure.
 */
hqlc_error hqlc_decoder_init(hqlc_decoder *dec, uint8_t channels, uint32_t sample_rate);

/*
 * @brief Reset decoder state after a gap in reception.
 *
 * Clears the MDCT overlap buffer so the next frame starts clean.
 */
void hqlc_decoder_reset(hqlc_decoder *dec);

/**
 * @brief Decode a single compressed frame to interleaved PCM.
 *
 * @param dec Pointer to the hqlc_decoder struct.
 * @param payload Pointer to the compressed frame bytes.
 * @param payload_len Size of the payload in bytes.
 * @param pcm_out Destination for decoded PCM (hqlc_frame_pcm_bytes() bytes).
 * @param fmt Desired output PCM format.
 * @param scratch Temporary workspace buffer (hqlc_decoder_scratch_size()
 * bytes).
 * @return hqlc_error HQLC_OK on success, or an error code on failure.
 */
hqlc_error hqlc_decode_frame(hqlc_decoder *dec,
                             const uint8_t *payload,
                             size_t payload_len,
                             uint8_t *pcm_out,
                             hqlc_pcm_format fmt,
                             void *scratch);

#ifdef __cplusplus
}
#endif

#endif /* HQLC_H */
