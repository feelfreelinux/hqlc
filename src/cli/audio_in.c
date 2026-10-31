#include "audio_in.h"

#include <stdio.h>
#include <string.h>

// The dr_libs implementations
#define DR_WAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#include "dr_wav.h"
#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#define DR_FLAC_NO_OGG
#include "dr_flac.h"

// Byte source callbacks, shared by both decoders
static size_t source_read(void *user, void *buf, size_t size) {
  return file_reader_read(&((audio_in *)user)->source, buf, size);
}

// Seek relative to the start, the current position or the end
static bool source_seek(audio_in *input, int64_t offset, bool from_end, bool relative) {
  file_reader *source = &input->source;
  if (from_end && source->size < 0) {
    return false;
  }
  int64_t target = offset;
  if (from_end) {
    target = source->size + offset;
  } else if (relative) {
    target = source->pos + offset;
  }
  return file_reader_seek(source, target) == 0;
}

static drwav_bool32 wav_seek(void *user, int offset, drwav_seek_origin origin) {
  if (source_seek(
          (audio_in *)user, offset, origin == DRWAV_SEEK_END, origin == DRWAV_SEEK_CUR)) {
    return DRWAV_TRUE;
  }
  return DRWAV_FALSE;
}

static drwav_bool32 wav_tell(void *user, drwav_int64 *pos) {
  *pos = ((audio_in *)user)->source.pos;
  return DRWAV_TRUE;
}

// WAV through dr_wav, sequential mode on pipes
static int open_wav(audio_in *input, const char *path) {
  audio_info *info = &input->info;
  drwav_uint32 flags = 0;
  if (!input->source.seekable) {
    flags = DRWAV_SEQUENTIAL; // pipes cannot seek back
  }
  if (!drwav_init_ex(
          &input->wav, source_read, wav_seek, wav_tell, NULL, input, NULL, flags, NULL)) {
    fprintf(stderr, "error: '%s' is not a WAV or FLAC file\n", path);
    return -1;
  }
  input->wav_open = true;
  info->channels = input->wav.channels;
  info->source_rate = input->wav.sampleRate;
  info->bits = input->wav.bitsPerSample;
  if (input->wav.translatedFormatTag == DR_WAVE_FORMAT_IEEE_FLOAT) {
    info->bits = 32;
  }
  info->total_frames = input->wav.totalPCMFrameCount;
  return 0;
}

static drflac_bool32 flac_seek(void *user, int offset, drflac_seek_origin origin) {
  if (source_seek((audio_in *)user,
                  offset,
                  origin == DRFLAC_SEEK_END,
                  origin == DRFLAC_SEEK_CUR)) {
    return DRFLAC_TRUE;
  }
  return DRFLAC_FALSE;
}

static drflac_bool32 flac_tell(void *user, drflac_int64 *pos) {
  *pos = ((audio_in *)user)->source.pos;
  return DRFLAC_TRUE;
}

// FLAC through dr_flac
static int open_flac(audio_in *input, const char *path) {
  audio_info *info = &input->info;
  input->flac = drflac_open(source_read, flac_seek, flac_tell, input, NULL);
  if (!input->flac) {
    fprintf(stderr, "error: cannot decode FLAC '%s'\n", path);
    return -1;
  }
  info->channels = input->flac->channels;
  info->source_rate = input->flac->sampleRate;
  info->bits = input->flac->bitsPerSample;
  info->total_frames = input->flac->totalPCMFrameCount;
  if (info->total_frames == 0) {
    info->total_frames = AUDIO_UNKNOWN_FRAMES;
  }
  return 0;
}

// Convert to the codec rate if needed
static int open_resampler(audio_in *input) {
  audio_info *info = &input->info;
  int err = 0;
  if (info->source_rate == HQLC_SAMPLE_RATE) {
    return 0;
  }
  input->resampler = speex_resampler_init(info->channels,
                                          info->source_rate,
                                          HQLC_SAMPLE_RATE,
                                          SPEEX_RESAMPLER_QUALITY_MAX,
                                          &err);
  if (!input->resampler) {
    fprintf(stderr, "error: cannot resample from %u Hz\n", info->source_rate);
    return -1;
  }
  speex_resampler_skip_zeros(input->resampler);
  if (info->total_frames != AUDIO_UNKNOWN_FRAMES) {
    info->total_frames = (info->total_frames * HQLC_SAMPLE_RATE + info->source_rate - 1) /
                         info->source_rate;
  }
  // Resampled output has full precision
  if (info->bits < 24) {
    info->bits = 24;
  }
  return 0;
}

// Decoded frames from whichever decoder is open
static size_t read_decoded(audio_in *input, int32_t *dst, size_t frames) {
  if (input->flac) {
    return (size_t)drflac_read_pcm_frames_s32(input->flac, frames, dst);
  }
  return (size_t)drwav_read_pcm_frames_s32(&input->wav, frames, dst);
}

// Refill pending with source frames, or with zeros to flush the filter
static void refill_pending(audio_in *input) {
  int32_t chunk[AUDIO_CHUNK_FRAMES * HQLC_MAX_CHANNELS];
  size_t got = 0;
  if (!input->source_ended) {
    got = read_decoded(input, chunk, AUDIO_CHUNK_FRAMES);
  }
  size_t got_samples = got * input->info.channels;
  input->frames_read += got;
  input->source_ended = input->source_ended || got < AUDIO_CHUNK_FRAMES;
  for (size_t i = 0; i < got_samples; i++) {
    input->pending[i] = (float)chunk[i] / 2147483648.0f;
  }
  for (size_t i = got_samples; i < (size_t)AUDIO_CHUNK_FRAMES * input->info.channels;
       i++) {
    input->pending[i] = 0.0f;
  }
  input->pending_pos = 0;
  input->pending_len = got;
  if (got == 0) {
    input->pending_len = AUDIO_CHUNK_FRAMES;
  }
}

// The filter delay is skipped at the start and flushed with zeros at the end,
// so the output covers exactly the input duration
static size_t read_resampled(audio_in *input, int32_t *dst, size_t frames) {
  const unsigned channels = input->info.channels;
  float out[AUDIO_CHUNK_FRAMES * HQLC_MAX_CHANNELS];
  size_t produced = 0;
  while (produced < frames) {
    if (input->pending_pos == input->pending_len) {
      refill_pending(input);
    }
    size_t want = frames - produced;
    if (want > AUDIO_CHUNK_FRAMES) {
      want = AUDIO_CHUNK_FRAMES;
    }
    if (input->source_ended) {
      uint64_t total =
          (input->frames_read * HQLC_SAMPLE_RATE + input->info.source_rate - 1) /
          input->info.source_rate;
      if (input->frames_produced >= total) {
        break;
      }
      if (want > total - input->frames_produced) {
        want = (size_t)(total - input->frames_produced);
      }
    }
    spx_uint32_t in_len = (spx_uint32_t)(input->pending_len - input->pending_pos);
    spx_uint32_t out_len = (spx_uint32_t)want;
    speex_resampler_process_interleaved_float(input->resampler,
                                              input->pending +
                                                  input->pending_pos * channels,
                                              &in_len,
                                              out,
                                              &out_len);
    input->pending_pos += in_len;
    for (size_t i = 0; i < (size_t)out_len * channels; i++) {
      float value = out[i] * 2147483648.0f;
      int32_t sample = (int32_t)value;
      if (value >= 2147483648.0f) {
        sample = INT32_MAX;
      } else if (value <= -2147483648.0f) {
        sample = INT32_MIN;
      }
      dst[produced * channels + i] = sample;
    }
    produced += out_len;
    input->frames_produced += out_len;
  }
  return produced;
}

// Only mono and stereo at a known rate
static int check_format(const audio_in *input) {
  const audio_info *info = &input->info;
  if (info->channels >= 1 && info->channels <= HQLC_MAX_CHANNELS && info->source_rate) {
    return 0;
  }
  fprintf(stderr,
          "error: %u channels at %u Hz, only mono and stereo are supported\n",
          info->channels,
          info->source_rate);
  return -1;
}

int audio_in_open(audio_in *input, const char *path) {
  memset(input, 0, sizeof(*input));
  if (file_reader_open(&input->source, path) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", path);
    return -1;
  }
  const file_reader *source = &input->source;
  bool flac = source->magic_len == 4 && memcmp(source->magic, "fLaC", 4) == 0;
  int status;
  if (flac) {
    status = open_flac(input, path);
  } else {
    status = open_wav(input, path);
  }
  if (status == 0 && check_format(input) == 0 && open_resampler(input) == 0) {
    return 0;
  }
  audio_in_close(input);
  return -1;
}

size_t audio_in_read(audio_in *input, int32_t *dst, size_t frames) {
  if (input->resampler) {
    return read_resampled(input, dst, frames);
  }
  return read_decoded(input, dst, frames);
}

void audio_in_close(audio_in *input) {
  if (input->wav_open) {
    drwav_uninit(&input->wav);
  }
  drflac_close(input->flac);
  if (input->resampler) {
    speex_resampler_destroy(input->resampler);
  }
  file_reader_close(&input->source);
  memset(input, 0, sizeof(*input));
}
