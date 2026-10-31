#include "wav_out.h"

#include <stdbool.h>
#include <string.h>

#include "hqlc.h"
#include "platform.h"

// dr_wav write callback
static size_t file_write(void *user, const void *data, size_t size) {
  return fwrite(data, 1, size, ((wav_out *)user)->file);
}

// dr_wav only seeks back to the header when closing
static drwav_bool32 file_seek(void *user, int offset, drwav_seek_origin origin) {
  return origin == DRWAV_SEEK_SET && platform_seek(((wav_out *)user)->file, offset) == 0;
}

int wav_out_open(wav_out *writer,
                 const char *path,
                 unsigned channels,
                 unsigned bits,
                 uint64_t frames) {
  memset(writer, 0, sizeof(*writer));
  writer->channels = channels;
  writer->bytes_per_sample = bits / 8;
  if (!(writer->file = platform_open(path, true))) {
    return -1;
  }
  drwav_data_format fmt = {
      drwav_container_riff, DR_WAVE_FORMAT_PCM, channels, HQLC_SAMPLE_RATE, bits};
  if (platform_is_file(writer->file, NULL)) {
    if (!drwav_init_write(&writer->wav, &fmt, file_write, file_seek, writer, NULL)) {
      return -1;
    }
    return 0;
  }
  uint64_t max = (0xFFFFFFFFu - 36) / (channels * writer->bytes_per_sample);
  if (frames > max) {
    frames = max;
  }
  if (!drwav_init_write_sequential_pcm_frames(
          &writer->wav, &fmt, frames, file_write, writer, NULL)) {
    return -1;
  }
  return 0;
}

int wav_out_write(wav_out *writer, const uint8_t *pcm, size_t frames) {
  writer->frames_written += frames;
  if (drwav_write_pcm_frames(&writer->wav, frames, pcm) != frames) {
    return -1;
  }
  return 0;
}

int wav_out_close(wav_out *writer) {
  // A pipe whose count fell short still gets all its audio, so the result is ignored
  drwav_uninit(&writer->wav);
  bool failed = fflush(writer->file) != 0;
  platform_close(writer->file);
  writer->file = NULL;
  if (failed) {
    return -1;
  }
  return 0;
}
