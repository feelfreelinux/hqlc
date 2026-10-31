/**
 * Command line encoder / decoder tool, compiled separately from the main lib
 *
 * Defining HQLC_TRAIN_TABLES during compilation enables the -T flag that is used for
 * training the entropy tables / rANS contexts during development
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ape_tags.h"
#include "audio_in.h"
#include "file_reader.h"
#include "hqlc.h"
#include "hqlc_container.h"
#include "platform.h"
#include "wav_out.h"

#ifdef HQLC_TRAIN_TABLES
#include "entropy.h"
#endif

// Codec latency, dropped from the start of the decoded output
#define PRE_SKIP HQLC_FRAME_SAMPLES

// What the tool does with its input
typedef enum { OP_ROUNDTRIP, OP_ENCODE, OP_DECODE, OP_INFO, OP_HELP } cli_op;

// Parsed command line
typedef struct {
  cli_op op;
  hqlc_mode mode; // rate controlled or fixed gain
  uint32_t bitrate;
  float gain;
  unsigned out_bits;      // WAV output depth, 16 or 24
  float start_seconds;    // decode start
  const char *train_path; // rANS histogram output, with HQLC_TRAIN_TABLES
  ape_tag_list tags;
  const char *in_path;
  const char *out_path;
} options;

// fwrite with a status return
static int write_all(FILE *file, const void *data, size_t size) {
  if (fwrite(data, 1, size, file) != size) {
    return -1;
  }
  return 0;
}

// Duration and bitrate summary on stderr
static void print_stats(const char *in_path,
                        const char *out_path,
                        uint64_t samples,
                        uint64_t bytes,
                        unsigned channels) {
  float secs = (float)samples / HQLC_SAMPLE_RATE;
  fprintf(stderr, "%s -> %s\n  %.2fs, %uch\n", in_path, out_path, secs, channels);
  if (secs > 0 && bytes > 0) {
    float bps = (float)bytes * 8 / secs;
    fprintf(stderr,
            "  avg bitrate: %.0f bps (%.1f:1)\n",
            bps,
            (float)(HQLC_SAMPLE_RATE * channels * 16) / bps);
  }
}

#ifdef HQLC_TRAIN_TABLES
// rANS table training collects symbol histograms while encoding
static void train_begin(const options *opts) {
  if (opts->train_path) {
    rans_train_reset();
  }
}

static void train_end(const options *opts) {
  if (!opts->train_path) {
    return;
  }
  FILE *file = fopen(opts->train_path, "w");
  if (!file) {
    fprintf(stderr, "warning: cannot write training histogram '%s'\n", opts->train_path);
    return;
  }
  for (int table = 0; table < RANS_COEF_NTABLES; table++) {
    for (int sym = 0; sym < RANS_MAX_SYM; sym++) {
      fprintf(file,
              "%llu%c",
              (unsigned long long)rans_train_hist[table][sym],
              sym == RANS_MAX_SYM - 1 ? '\n' : ' ');
    }
  }
  fclose(file);
}
#else
#define train_begin(opts) ((void)(opts))
#define train_end(opts)   ((void)(opts))
#endif

// Encoder struct
typedef struct {
  audio_in *input;
  hqlc_encoder *state;
  void *scratch;
  hqlc_pcm_format fmt;
  uint64_t samples_read; // input samples per channel
  uint64_t frames_encoded;
  uint64_t bytes_encoded;
  bool input_ended;
} encoder;

// Codec encoder from the input format and the options
static int encoder_open(encoder *enc, audio_in *input, const options *opts) {
  hqlc_encoder_config cfg = {0};
  memset(enc, 0, sizeof(*enc));
  enc->input = input;
  enc->fmt = input->info.bits > 16 ? HQLC_PCM24 : HQLC_PCM16;
  cfg.channels = (uint8_t)input->info.channels;
  cfg.sample_rate = HQLC_SAMPLE_RATE;
  cfg.mode = opts->mode;
  if (opts->mode == HQLC_MODE_RC) {
    cfg.bitrate = opts->bitrate;
  } else {
    cfg.gain = opts->gain;
  }
  enc->state = (hqlc_encoder *)malloc(hqlc_encoder_size());
  enc->scratch = malloc(hqlc_encoder_scratch_size());
  if (!enc->state || !enc->scratch || hqlc_encoder_init(enc->state, &cfg) != HQLC_OK) {
    fprintf(stderr, "error: encoder init failed\n");
    return -1;
  }
  return 0;
}

// Pack one codec frame of input, zero-padded past the end, returns the frames read
static size_t read_input_frame(encoder *enc, uint8_t *pcm) {
  int32_t samples[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS];
  unsigned channels = enc->input->info.channels;
  size_t got = audio_in_read(enc->input, samples, HQLC_FRAME_SAMPLES);
  memset(samples + got * channels,
         0,
         (HQLC_FRAME_SAMPLES - got) * channels * sizeof(int32_t));
  unsigned shift = 8; // left-justified 32-bit samples down to 16 or 24 bits
  if (enc->fmt == HQLC_PCM16) {
    shift = 16;
  }
  for (size_t i = 0; i < (size_t)HQLC_FRAME_SAMPLES * channels; i++) {
    uint32_t value = (uint32_t)samples[i] >> shift;
    *pcm++ = (uint8_t)value;
    *pcm++ = (uint8_t)(value >> 8);
    if (enc->fmt == HQLC_PCM24) {
      *pcm++ = (uint8_t)(value >> 16);
    }
  }
  return got;
}

/**
 * @brief Encode the next frame, padding with silence until PRE_SKIP is covered.
 *
 * @param enc Open encoder.
 * @param frame Receives the compressed frame, HQLC_MAX_FRAME_BYTES capacity.
 * @param len Receives the frame size in bytes.
 * @return 1 while frames come, 0 once the input plus PRE_SKIP is covered, -1 on error.
 */
static int encoder_next_frame(encoder *enc, uint8_t *frame, size_t *len) {
  uint8_t pcm[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS * 3] = {0};
  if (enc->input_ended &&
      enc->frames_encoded * HQLC_FRAME_SAMPLES >= enc->samples_read + PRE_SKIP) {
    return 0;
  }
  if (!enc->input_ended) {
    size_t got = read_input_frame(enc, pcm);
    enc->samples_read += got;
    enc->input_ended = got < HQLC_FRAME_SAMPLES;
  }
  if (hqlc_encode_frame(
          enc->state, pcm, enc->fmt, frame, HQLC_MAX_FRAME_BYTES, len, enc->scratch) !=
      HQLC_OK) {
    fprintf(stderr,
            "error: encode failed at frame %llu\n",
            (unsigned long long)enc->frames_encoded);
    return -1;
  }
  enc->frames_encoded++;
  enc->bytes_encoded += *len;
  return 1;
}

/**
 * @brief Decoded samples of the last frame that map to real input
 *
 * Same trimming the container reader applies with pre_skip and pad
 *
 * @param enc Encoder that has produced at least one frame.
 * @param skip Receives the decoded samples to drop from the start of the frame.
 * @param keep Receives the decoded samples to output after skip.
 */
static void encoder_frame_trim(const encoder *enc, size_t *skip, size_t *keep) {
  uint64_t first = (enc->frames_encoded - 1) * HQLC_FRAME_SAMPLES;
  uint64_t start = first;
  if (start < PRE_SKIP) {
    start = PRE_SKIP;
  }
  uint64_t end = enc->samples_read + PRE_SKIP;
  if (end > first + HQLC_FRAME_SAMPLES) {
    end = first + HQLC_FRAME_SAMPLES;
  }
  if (start > end) {
    start = end;
  }
  *skip = (size_t)(start - first);
  *keep = (size_t)(end - start);
}

static void encoder_close(encoder *enc) {
  free(enc->state);
  free(enc->scratch);
}

// Decoder, codec frames to WAV
typedef struct {
  hqlc_decoder *state;
  void *scratch;
  hqlc_pcm_format fmt;
  wav_out wav;
} decoder;

static int decoder_open(
    decoder *dec, const char *path, unsigned channels, unsigned bits, uint64_t frames) {
  memset(dec, 0, sizeof(*dec));
  dec->fmt = bits == 24 ? HQLC_PCM24 : HQLC_PCM16;
  dec->state = (hqlc_decoder *)malloc(hqlc_decoder_size());
  dec->scratch = malloc(hqlc_decoder_scratch_size());
  if (!dec->state || !dec->scratch ||
      hqlc_decoder_init(dec->state, (uint8_t)channels, HQLC_SAMPLE_RATE) != HQLC_OK) {
    fprintf(stderr, "error: decoder init failed\n");
    return -1;
  }
  if (wav_out_open(&dec->wav, path, channels, bits, frames) != 0) {
    fprintf(stderr, "error: cannot write '%s'\n", path);
    return -1;
  }
  return 0;
}

/**
 * @brief Decode one frame and write the samples to the WAV, excluding the skip
 *
 * A rejected frame is written as silence so the timeline stays intact
 *
 * @param dec Open decoder.
 * @param frame Compressed frame bytes.
 * @param len Frame size in bytes.
 * @param reset Reset the decoder state before decoding, after a seek or a gap.
 * @param skip Decoded samples to drop from the start of the frame.
 * @param keep Decoded samples to write after skip.
 * @return 0 on success, 1 if the decoder rejected the frame, -1 on a write error.
 */
static int decoder_write_frame(decoder *dec,
                               const uint8_t *frame,
                               size_t len,
                               bool reset,
                               size_t skip,
                               size_t keep) {
  uint8_t pcm[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS * 3];
  bool bad = false;
  if (reset) {
    hqlc_decoder_reset(dec->state);
  }
  if (hqlc_decode_frame(dec->state, frame, len, pcm, dec->fmt, dec->scratch) != HQLC_OK) {
    hqlc_decoder_reset(dec->state);
    memset(pcm, 0, sizeof(pcm));
    bad = true;
  }
  size_t stride = (size_t)dec->wav.channels * dec->wav.bytes_per_sample;
  if (keep && wav_out_write(&dec->wav, pcm + skip * stride, keep) != 0) {
    fprintf(stderr, "error: write failed\n");
    return -1;
  }
  if (bad) {
    return 1;
  }
  return 0;
}

static int decoder_close(decoder *dec) {
  bool failed = dec->wav.file && wav_out_close(&dec->wav) != 0;
  free(dec->state);
  free(dec->scratch);
  if (failed) {
    return -1;
  }
  return 0;
}

static int handle_encode(const options *opts) {
  audio_in input;
  encoder enc = {0};
  // Allocate the container handler
  hqlc_container_writer *writer =
      (hqlc_container_writer *)malloc(hqlc_container_writer_size());
  uint8_t out[HQLC_CONTAINER_MAX_WRITE_BYTES];
  uint8_t frame[HQLC_MAX_FRAME_BYTES];
  size_t len, size;
  if (audio_in_open(&input, opts->in_path) != 0) {
    return 1;
  }
  train_begin(opts);
  FILE *file = platform_open(opts->out_path, true);
  bool ok =
      file && writer && encoder_open(&enc, &input, opts) == 0 &&
      hqlc_container_writer_init(writer, (uint8_t)input.info.channels, PRE_SKIP, out) ==
          HQLC_CONTAINER_OK &&
      write_all(file, out, HQLC_CONTAINER_HEADER_BYTES) == 0;
  int status = 0;
  while (ok && (status = encoder_next_frame(&enc, frame, &len)) > 0) {
    ok =
        hqlc_container_write_frame(writer, frame, len, out, &size) == HQLC_CONTAINER_OK &&
        write_all(file, out, size) == 0;
  }
  ok = ok && status == 0 &&
       hqlc_container_write_end(writer, enc.samples_read, out, &size) ==
           HQLC_CONTAINER_OK &&
       write_all(file, out, size) == 0;
  // APEv2 goes last
  if (ok && opts->tags.count) {
    size_t tag_len;
    uint8_t *tag = ape_tags_build(&opts->tags, &tag_len);
    ok = tag && write_all(file, tag, tag_len) == 0;
    free(tag);
  }
  ok = ok && fflush(file) == 0;
  if (ok) {
    train_end(opts);
    print_stats(opts->in_path,
                opts->out_path,
                enc.samples_read,
                enc.bytes_encoded,
                input.info.channels);
  } else {
    fprintf(stderr, "error: cannot encode '%s' to '%s'\n", opts->in_path, opts->out_path);
  }
  platform_close(file);
  free(writer);
  encoder_close(&enc);
  audio_in_close(&input);
  return ok ? 0 : 1;
}

// Decodes each frame as it is encoded, trimmed like the container reader would
static int handle_roundtrip(const options *opts) {
  audio_in input;
  encoder enc = {0};
  decoder dec = {0};
  uint8_t frame[HQLC_MAX_FRAME_BYTES];
  size_t len, skip, keep;
  if (audio_in_open(&input, opts->in_path) != 0) {
    return 1;
  }
  bool ok = encoder_open(&enc, &input, opts) == 0;
  ok = ok && decoder_open(&dec,
                          opts->out_path,
                          input.info.channels,
                          opts->out_bits,
                          input.info.total_frames) == 0;
  int status = 0;
  while (ok && (status = encoder_next_frame(&enc, frame, &len)) > 0) {
    encoder_frame_trim(&enc, &skip, &keep);
    ok = decoder_write_frame(&dec, frame, len, false, skip, keep) == 0;
  }
  ok = ok && status == 0;
  ok = decoder_close(&dec) == 0 && ok;
  if (ok) {
    print_stats(opts->in_path,
                opts->out_path,
                enc.samples_read,
                enc.bytes_encoded,
                input.info.channels);
  }
  encoder_close(&enc);
  audio_in_close(&input);
  return ok ? 0 : 1;
}

// I/O callback for the container reader
static int file_read_at(void *ctx, uint64_t offset, void *buf, size_t len, size_t *got) {
  file_reader *file = (file_reader *)ctx;
  if (file_reader_seek(file, (int64_t)offset) != 0) {
    return -1;
  }
  *got = file_reader_read(file, buf, len);
  return 0;
}

// Open a .hqlc stream for reading
static int
open_stream(hqlc_container_reader *reader, file_reader *file, const char *path) {
  static const char *reasons[] = {
      [HQLC_CONTAINER_ERR_IO] = "read error",
      [HQLC_CONTAINER_ERR_FORMAT] = "not an .hqlc file",
      [HQLC_CONTAINER_ERR_VERSION] =
          "unsupported .hqlc version (older files need the matching hqlc)",
  };
  if (file_reader_open(file, path) != 0) {
    fprintf(stderr, "error: cannot read '%s'\n", path);
    return -1;
  }
  uint64_t size = 0;
  if (file->seekable) {
    size = (uint64_t)file->size;
  }
  hqlc_container_io io = {file_read_at, file, size};
  hqlc_container_error status = hqlc_container_reader_open(reader, &io);
  if (status != HQLC_CONTAINER_OK) {
    const char *reason = "cannot open";
    if (status <= HQLC_CONTAINER_ERR_VERSION && reasons[status]) {
      reason = reasons[status];
    }
    fprintf(stderr, "error: '%s': %s\n", path, reason);
    file_reader_close(file);
    return -1;
  }
  if (hqlc_container_reader_info(reader)->bitstream_version != HQLC_BITSTREAM_VERSION) {
    fprintf(stderr,
            "error: '%s' needs a newer hqlc (bitstream %u)\n",
            path,
            hqlc_container_reader_info(reader)->bitstream_version);
    file_reader_close(file);
    return -1;
  }
  return 0;
}

// Decodes an hqlc file to a given format
static int handle_decode(const options *opts) {
  file_reader file;
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  if (open_stream(reader, &file, opts->in_path) != 0) {
    free(reader);
    return 1;
  }
  const hqlc_container_info *info = hqlc_container_reader_info(reader);
  uint64_t start = (uint64_t)(opts->start_seconds * HQLC_SAMPLE_RATE);
  uint64_t frames = info->total_samples;
  bool ok = true;
  if (start) {
    ok = hqlc_container_seek(reader, start) == HQLC_CONTAINER_OK;
    if (!ok) {
      fprintf(stderr, "error: '%s' is not seekable\n", opts->in_path);
    }
    if (frames != HQLC_CONTAINER_UNKNOWN_SAMPLES) {
      if (start < frames) {
        frames -= start;
      } else {
        frames = 0;
      }
    }
  }

  decoder dec = {0};
  ok = ok &&
       decoder_open(&dec, opts->out_path, info->channels, opts->out_bits, frames) == 0;
  uint8_t frame[HQLC_MAX_FRAME_BYTES];
  hqlc_container_frame frame_info;
  hqlc_container_error status = HQLC_CONTAINER_OK;
  uint64_t damaged = 0;
  uint64_t bytes = 0;
  uint64_t next_index = UINT64_MAX;
  while (ok && (status = hqlc_container_read_frame(
                    reader, frame, sizeof(frame), &frame_info)) == HQLC_CONTAINER_OK) {
    // Lost frames become silence so we dont break the timeline
    for (; next_index < frame_info.index && ok; next_index++) {
      static const uint8_t silence[HQLC_FRAME_SAMPLES * HQLC_MAX_CHANNELS * 3];
      ok = wav_out_write(&dec.wav, silence, HQLC_FRAME_SAMPLES) == 0;
    }
    next_index = frame_info.index + 1;
    int result = decoder_write_frame(
        &dec, frame, frame_info.len, frame_info.reset, frame_info.skip, frame_info.keep);
    if (frame_info.damaged || result > 0) {
      damaged++;
    }
    bytes += frame_info.len;
    ok = result >= 0;
  }
  if (ok && status != HQLC_CONTAINER_END) {
    fprintf(stderr, "error: read error in '%s'\n", opts->in_path);
    ok = false;
  }
  ok = decoder_close(&dec) == 0 && ok;
  if (damaged) {
    fprintf(stderr,
            "warning: '%s' is damaged, skipped %llu bad spots\n",
            opts->in_path,
            (unsigned long long)damaged);
  }
  if (ok) {
    print_stats(
        opts->in_path, opts->out_path, dec.wav.frames_written, bytes, info->channels);
  }
  file_reader_close(&file);
  free(reader);
  return ok ? 0 : 1;
}

// Print info about the file, including ape tags
static int handle_info(const options *opts) {
  file_reader file;
  hqlc_container_reader *reader =
      (hqlc_container_reader *)malloc(hqlc_container_reader_size());
  if (open_stream(reader, &file, opts->in_path) != 0) {
    free(reader);
    return 1;
  }
  const hqlc_container_info *info = hqlc_container_reader_info(reader);
  printf("%s\n  %s, %d Hz\n",
         opts->in_path,
         info->channels == 2 ? "stereo" : "mono",
         HQLC_SAMPLE_RATE);
  if (info->seekable) {
    float secs = (float)info->total_samples / HQLC_SAMPLE_RATE;
    uint64_t audio_bytes = (uint64_t)file.size - info->tag_size;
    float kbps = 0.0f;
    if (secs > 0) {
      kbps = (float)audio_bytes * 8 / secs / 1000;
    }
    printf("  %.3f s (%llu samples), %.1f kbps\n",
           secs,
           (unsigned long long)info->total_samples,
           kbps);
  } else {
    printf("  length unknown (stream or truncated file)\n");
  }

  ape_tag_list tags = {0};
  uint8_t *tag = NULL;
  if (info->tag_size) {
    tag = (uint8_t *)malloc(info->tag_size);
  }
  if (tag && file_reader_seek(&file, (int64_t)info->tag_offset) == 0 &&
      file_reader_read(&file, tag, info->tag_size) == info->tag_size &&
      ape_tags_parse(tag, info->tag_size, &tags) == 0) {
    for (size_t i = 0; i < tags.count; i++) {
      printf("  %s=%s\n", tags.items[i].key, tags.items[i].value);
    }
  }
  free(tag);
  ape_tags_free(&tags);
  file_reader_close(&file);
  free(reader);
  return 0;
}

static void print_usage(const char *program) {
  fprintf(stderr,
          "Usage: %s [options] input [output]\n"
          "\n"
          "Modes:\n"
          "  (default)     Roundtrip: encode+decode to WAV for quality evaluation\n"
          "  -e            Encode WAV/FLAC at any rate to .hqlc\n"
          "  -d            Decode .hqlc to WAV\n"
          "  -i            Print .hqlc info and tags\n"
          "\n"
          "Options:\n"
          "  -b <bps>      Rate-controlled mode (default: 128000)\n"
          "  -g <gain>     Fixed-gain mode\n"
          "  -t KEY=VALUE  Add a tag when encoding (repeatable)\n"
          "  -s <sec>      Start decoding at this position\n"
          "  --24          Write 24-bit WAV\n"
#ifdef HQLC_TRAIN_TABLES
          "  -T <file>     Dump rANS table-training histogram while encoding\n"
#endif
          "\n"
          "Use \"-\" for stdin/stdout.\n",
          program);
}

/**
 * @brief Parse the command line into opts.
 *
 * @param argc Argument count
 * @param argv UTF-8 arguments
 * @param opts Receives the options, tags included, free with ape_tags_free().
 * @return 0 on success, -1 for a usage error, -2 for an error already reported.
 */
static int parse_args(int argc, char **argv, options *opts) {
  memset(opts, 0, sizeof(*opts));
  opts->mode = HQLC_MODE_RC;
  opts->bitrate = 128000;
  opts->out_bits = 16;
  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    bool has_value = i + 1 < argc;
    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      opts->op = OP_HELP;
      return 0;
    } else if (strcmp(arg, "-e") == 0) {
      opts->op = OP_ENCODE;
    } else if (strcmp(arg, "-d") == 0) {
      opts->op = OP_DECODE;
    } else if (strcmp(arg, "-i") == 0) {
      opts->op = OP_INFO;
    } else if (strcmp(arg, "-b") == 0 && has_value) {
      opts->mode = HQLC_MODE_RC;
      opts->bitrate = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "-g") == 0 && has_value) {
      opts->mode = HQLC_MODE_FIXED;
      opts->gain = strtof(argv[++i], NULL);
    } else if (strcmp(arg, "-s") == 0 && has_value) {
      opts->start_seconds = strtof(argv[++i], NULL);
    } else if (strcmp(arg, "--24") == 0) {
      opts->out_bits = 24;
    } else if (strcmp(arg, "-t") == 0 && has_value) {
      const char *pair = argv[++i];
      const char *equals = strchr(pair, '=');
      if (!equals || ape_tags_set(&opts->tags,
                                  pair,
                                  (size_t)(equals - pair),
                                  equals + 1,
                                  strlen(equals + 1)) != 0) {
        fprintf(stderr, "error: bad tag '%s', expected KEY=VALUE\n", pair);
        return -2;
      }
#ifdef HQLC_TRAIN_TABLES
    } else if (strcmp(arg, "-T") == 0 && has_value) {
      opts->op = OP_ENCODE;
      opts->train_path = argv[++i];
#endif
    } else if (arg[0] == '-' && arg[1] != '\0') {
      return -1;
    } else if (!opts->in_path) {
      opts->in_path = arg;
    } else if (!opts->out_path) {
      opts->out_path = arg;
    } else {
      return -1;
    }
  }
  if (!opts->in_path || (!opts->out_path && opts->op != OP_INFO)) {
    return -1;
  }
  return 0;
}

int main(int argc, char **argv) {
  argv = platform_args(argc, argv);
  options opts;
  int status = parse_args(argc, argv, &opts);
  int ret = 1;
  if (status == 0) {
    switch (opts.op) {
    case OP_ENCODE:
      ret = handle_encode(&opts);
      break;
    case OP_DECODE:
      ret = handle_decode(&opts);
      break;
    case OP_INFO:
      ret = handle_info(&opts);
      break;
    case OP_ROUNDTRIP:
      ret = handle_roundtrip(&opts);
      break;
    case OP_HELP:
      print_usage(argv[0]);
      ret = 0;
      break;
    }
  } else if (status == -1) {
    print_usage(argv[0]);
  }
  ape_tags_free(&opts.tags);
  return ret;
}
