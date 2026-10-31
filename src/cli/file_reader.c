#include "file_reader.h"

#include <string.h>

#include "platform.h"

int file_reader_open(file_reader *reader, const char *path) {
  memset(reader, 0, sizeof(*reader));
  reader->size = -1;
  if (!(reader->file = platform_open(path, false))) {
    return -1;
  }
  reader->seekable = platform_is_file(reader->file, &reader->size);
  reader->magic_len = fread(reader->magic, 1, sizeof(reader->magic), reader->file);
  return 0;
}

size_t file_reader_read(file_reader *reader, void *buf, size_t size) {
  size_t replay = reader->magic_len - reader->magic_pos;
  if (replay > size) {
    replay = size;
  }
  memcpy(buf, reader->magic + reader->magic_pos, replay);
  reader->magic_pos += replay;
  size_t got = replay + fread((uint8_t *)buf + replay, 1, size - replay, reader->file);
  reader->pos += (int64_t)got;
  return got;
}

int file_reader_seek(file_reader *reader, int64_t offset) {
  if (offset == reader->pos) {
    return 0;
  }
  // Nothing read past the magic yet, so even a pipe can go back there
  if (reader->pos <= (int64_t)reader->magic_len && offset >= 0 &&
      offset <= (int64_t)reader->magic_len) {
    reader->magic_pos = (size_t)offset;
    reader->pos = offset;
    return 0;
  }
  if (reader->seekable) {
    reader->magic_len = 0;
    reader->magic_pos = 0;
    if (platform_seek(reader->file, offset)) {
      return -1;
    }
    reader->pos = offset;
    return 0;
  }
  uint8_t skip[4096];
  while (offset > reader->pos) {
    int64_t left = offset - reader->pos;
    size_t chunk = sizeof(skip);
    if (left < (int64_t)sizeof(skip)) {
      chunk = (size_t)left;
    }
    if (file_reader_read(reader, skip, chunk) != chunk) {
      return -1;
    }
  }
  if (offset != reader->pos) {
    return -1;
  }
  return 0;
}

void file_reader_close(file_reader *reader) {
  platform_close(reader->file);
  reader->file = NULL;
}
