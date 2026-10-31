/**
 * Header for the hqlc container, used when writing hqlc frames into files. This is
 * optional, and hqlc frames can be directly wrapped in whatever else. See
 * HQLC_CONTAINER.md for a description of frame format
 */
#ifndef HQLC_CONTAINER_H
#define HQLC_CONTAINER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hqlc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Container constants
#define HQLC_CONTAINER_VERSION      10
#define HQLC_CONTAINER_PAGE_FRAMES  64
#define HQLC_CONTAINER_HEADER_BYTES 8
#define HQLC_CONTAINER_MAX_CHANNELS 16

// total_samples of a stream without a trailer
#define HQLC_CONTAINER_UNKNOWN_SAMPLES UINT64_MAX

// page header + entry length varint + frame + page CRC
#define HQLC_CONTAINER_MAX_WRITE_BYTES (7 + 2 + HQLC_MAX_FRAME_BYTES + 2)

// page header + end marker varint + pad + page CRC + trailer
#define HQLC_CONTAINER_MAX_END_BYTES (7 + 2 + 2 + 2 + 20)

// Status code enumeration
typedef enum {
  HQLC_CONTAINER_OK = 0,          /**< Success */
  HQLC_CONTAINER_END,             /**< No more frames */
  HQLC_CONTAINER_ERR_INVALID_ARG, /**< NULL pointer, out-of-range value or small buffer */
  HQLC_CONTAINER_ERR_IO,          /**< read callback failed */
  HQLC_CONTAINER_ERR_FORMAT,      /**< not an .hqlc stream, or malformed */
  HQLC_CONTAINER_ERR_VERSION,     /**< legacy or unknown container version */
  HQLC_CONTAINER_ERR_NOT_SEEKABLE, /**< stream has no trailer or unknown size */
} hqlc_container_error;

// I/O callbacks the reader pulls bytes through, with size 0 it works as a pipe
typedef struct {
  int (*read_at)(void *ctx, uint64_t off, void *buf, size_t len, size_t *got);
  /**< Fills up to len bytes at offset off, nonzero on error */
  void *ctx;     /**< Passed to read_at */
  uint64_t size; /**< Total bytes, 0 if unknown */
} hqlc_container_io;

// Stream properties, valid after hqlc_container_reader_open()
typedef struct {
  uint8_t channels;          /**< 1..16 */
  uint8_t bitstream_version; /**< Check against HQLC_BITSTREAM_VERSION */
  uint16_t pre_skip;         /**< Decoded samples dropped at the start */
  uint64_t total_samples;    /**< Per channel after trimming, or unknown */
  uint64_t tag_offset;       /**< APEv2 tag start, 0 if none */
  uint32_t tag_size;         /**< APEv2 tag size including header and footer */
  bool seekable;             /**< Size known and trailer present */
} hqlc_container_info;

// One frame returned by hqlc_container_read_frame()
typedef struct {
  uint64_t index; /**< Frame index in the stream */
  size_t len;     /**< Frame bytes */
  uint16_t skip;  /**< Decoded samples to drop from the start of this frame */
  uint16_t keep;  /**< Decoded samples to output after skip */
  bool reset;     /**< Reset the decoder before decoding this frame */
  bool damaged;   /**< Data before this frame was corrupt or missing */
} hqlc_container_frame;

// Fwd defs
typedef struct hqlc_container_writer hqlc_container_writer;
typedef struct hqlc_container_reader hqlc_container_reader;

/**
 * @brief Return the required size of a writer instance.
 *
 * @return Size of `hqlc_container_writer` in bytes.
 */
size_t hqlc_container_writer_size(void);

/**
 * @brief Return the required size of a reader instance.
 *
 * @return Size of `hqlc_container_reader` in bytes.
 */
size_t hqlc_container_reader_size(void);

/**
 * @brief Initialize a writer and emit the file header.
 *
 * @param writer Writer storage allocated by the caller with hqlc_container_writer_size()
 * bytes.
 * @param channels Number of channels, 1..16.
 * @param pre_skip Decoded samples to drop at the start, usually 512
 * @param out Receives HQLC_CONTAINER_HEADER_BYTES bytes to write.
 * @return HQLC_CONTAINER_OK on success, or an error code on failure.
 */
hqlc_container_error hqlc_container_writer_init(hqlc_container_writer *writer,
                                                uint8_t channels,
                                                uint16_t pre_skip,
                                                uint8_t *out);

/**
 * @brief Add one codec frame.
 *
 * @param writer Initialized writer instance.
 * @param frame Compressed frame from hqlc_encode_frame().
 * @param len Frame size, 1..HQLC_MAX_FRAME_BYTES.
 * @param out Receives the bytes to write, HQLC_CONTAINER_MAX_WRITE_BYTES capacity.
 * @param out_len Receives the number of bytes in `out`.
 * @return HQLC_CONTAINER_OK on success, or an error code on failure.
 */
hqlc_container_error hqlc_container_write_frame(hqlc_container_writer *writer,
                                                const uint8_t *frame,
                                                size_t len,
                                                uint8_t *out,
                                                size_t *out_len);

/**
 * @brief End the stream with the end marker, the last page CRC and the trailer.
 *
 * @param writer Initialized writer instance.
 * @param input_samples Samples per channel that went into the encoder.
 * @param out Receives the bytes to write, HQLC_CONTAINER_MAX_END_BYTES capacity.
 * @param out_len Receives the number of bytes in `out`.
 * @return HQLC_CONTAINER_OK, or HQLC_CONTAINER_ERR_INVALID_ARG when the frames
 *         written do not cover pre_skip + input_samples within one frame.
 */
hqlc_container_error hqlc_container_write_end(hqlc_container_writer *writer,
                                              uint64_t input_samples,
                                              uint8_t *out,
                                              size_t *out_len);

/**
 * @brief Opens a stream, parses the header and tail when valid
 *
 * @param reader Reader storage allocated by the caller with hqlc_container_reader_size()
 * bytes.
 * @param io I/O callbacks, copied into `reader`.
 * @return HQLC_CONTAINER_OK on success, or an error code on failure.
 */
hqlc_container_error hqlc_container_reader_open(hqlc_container_reader *reader,
                                                const hqlc_container_io *io);

/**
 * @brief Return the stream info
 *
 * @param reader Needs to be opened first
 * @return Stream properties, owned by `reader`.
 */
const hqlc_container_info *
hqlc_container_reader_info(const hqlc_container_reader *reader);

/**
 * @brief Read the next frame
 *
 * Skips to a valid page in case of a desync, setting the damage flag if so
 *
 * @param reader Opened reader instance.
 * @param buf Receives the frame, at least HQLC_MAX_FRAME_BYTES capacity.
 * @param cap Capacity of `buf`.
 * @param frame Receives the frame description.
 * @return HQLC_CONTAINER_OK, HQLC_CONTAINER_END, or an error code on failure.
 */
hqlc_container_error hqlc_container_read_frame(hqlc_container_reader *reader,
                                               uint8_t *buf,
                                               size_t cap,
                                               hqlc_container_frame *frame);

/**
 * @brief Seek so that output resumes exactly at `sample`.
 *
 * The decoder only needs to be reset if the 'reset' flag is set afterwards.
 *
 * @param reader Opened reader instance.
 * @param sample Output sample per channel, clamped to total_samples.
 * @return HQLC_CONTAINER_OK, HQLC_CONTAINER_ERR_NOT_SEEKABLE, or an error code.
 */
hqlc_container_error hqlc_container_seek(hqlc_container_reader *reader, uint64_t sample);

#ifdef __cplusplus
}
#endif

#endif /* HQLC_CONTAINER_H */
