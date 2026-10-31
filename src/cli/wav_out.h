/**
 * WAV output through dr_wav, for the decoder's packed little-endian PCM
 */
#ifndef HQLC_CLI_WAV_OUT_H
#define HQLC_CLI_WAV_OUT_H

#include <stdint.h>
#include <stdio.h>

#include "dr_wav.h"

// Writer state, to be initialized with wav_out_open
typedef struct {
  FILE *file;                /**< Open output, NULL once closed */
  drwav wav;                 /**< dr_wav writer */
  unsigned channels;         /**< 1 or 2 */
  unsigned bytes_per_sample; /**< 2 or 3 */
  uint64_t frames_written;   /**< Frames written so far */
} wav_out;

/**
 * @brief Open a WAV file for writing.
 *
 * Files get their sizes fixed on close. Pipes need the frame count up front,
 * an unknown count is written as the largest a WAV can hold.
 *
 * @param writer Writer state.
 * @param path File path, or "-" for stdout.
 * @param channels Number of channels.
 * @param bits Bits per sample, 16 or 24.
 * @param frames Expected frame count, only used for pipes.
 * @return 0 on success, -1 on failure.
 */
int wav_out_open(
    wav_out *writer, const char *path, unsigned channels, unsigned bits, uint64_t frames);

/**
 * @brief Write interleaved packed PCM frames.
 *
 * @param writer Open writer.
 * @param pcm Packed little-endian samples, channels * bytes_per_sample per frame.
 * @param frames Frames to write.
 * @return 0 on success, -1 on failure.
 */
int wav_out_write(wav_out *writer, const uint8_t *pcm, size_t frames);

/**
 * @brief Finish the file and close it.
 *
 * @param writer Open writer.
 * @return 0 on success, -1 when the final flush fails.
 */
int wav_out_close(wav_out *writer);

#endif /* HQLC_CLI_WAV_OUT_H */
