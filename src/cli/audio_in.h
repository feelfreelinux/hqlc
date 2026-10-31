/**
 * Audio input for the CLI, handles wav, flac and resampling
 */
#ifndef HQLC_CLI_AUDIO_IN_H
#define HQLC_CLI_AUDIO_IN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dr_flac.h"
#include "dr_wav.h"
#include "hqlc.h"
#include "file_reader.h"
#include "speex_resampler.h"

// total_frames of a source without a known length
#define AUDIO_UNKNOWN_FRAMES UINT64_MAX

// Source frames decoded and resampled per step
#define AUDIO_CHUNK_FRAMES 1024

// Source properties
typedef struct {
  unsigned channels;     /**< 1 or 2 */
  unsigned source_rate;  /**< Sample rate of the file in Hz */
  unsigned bits;         /**< Significant bits per sample, 32 for float sources */
  uint64_t total_frames; /**< Frames at the codec rate, or AUDIO_UNKNOWN_FRAMES */
} audio_info;

// Reader state, to be initialized with audio_in_open
typedef struct {
  audio_info info;                /**< Source properties */
  file_reader source;             /**< Where the bytes come from */
  drwav wav;                      /**< dr_wav decoder */
  bool wav_open;                  /**< wav holds an initialized decoder */
  drflac *flac;                   /**< dr_flac decoder, NULL for WAV */
  SpeexResamplerState *resampler; /**< NULL at the codec rate */
  float pending[AUDIO_CHUNK_FRAMES * HQLC_MAX_CHANNELS]; /**< Source frames to resample */
  size_t pending_pos;       /**< Frames of pending consumed */
  size_t pending_len;       /**< Frames in pending */
  uint64_t frames_read;     /**< Source frames decoded so far */
  uint64_t frames_produced; /**< Resampled frames delivered so far */
  bool source_ended;        /**< The decoder ran out of frames */
} audio_in;

/**
 * @brief Open an audio file, printing the reason on failure.
 *
 * @param input Reader state.
 * @param path File path, or "-" for stdin.
 * @return 0 on success, -1 on failure.
 */
int audio_in_open(audio_in *input, const char *path);

/**
 * @brief Read interleaved frames at the codec rate.
 *
 * @param input Open reader.
 * @param dst Receives frames * channels samples.
 * @param frames Frames wanted.
 * @return Frames read, fewer only at end of input.
 */
size_t audio_in_read(audio_in *input, int32_t *dst, size_t frames);

/**
 * @brief Close the reader.
 *
 * @param input Open reader.
 */
void audio_in_close(audio_in *input);

#endif /* HQLC_CLI_AUDIO_IN_H */
