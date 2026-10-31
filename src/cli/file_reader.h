/**
 * Read-only file with a tracked position, pipe-aware seeking and the first
 * bytes kept for format peeking
 */
#ifndef HQLC_CLI_FILE_READER_H
#define HQLC_CLI_FILE_READER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// Reader state, to be initialized with file_reader_open
typedef struct {
  FILE *file;       /**< Open stream */
  bool seekable;    /**< Regular file, seeks and has a known size */
  int64_t pos;      /**< Next byte to read */
  int64_t size;     /**< Total bytes, -1 if unknown */
  uint8_t magic[4]; /**< First bytes of the stream */
  size_t magic_len; /**< Valid bytes in magic */
  size_t magic_pos; /**< Magic bytes replayed so far */
} file_reader;

/**
 * @brief Open a file and read its magic bytes
 *
 * @param reader Reader state.
 * @param path File path, or "-" for stdin.
 * @return 0 on success, -1 on failure.
 */
int file_reader_open(file_reader *reader, const char *path);

/**
 * @brief Read bytes, replaying the magic first
 *
 * @param reader Open reader.
 * @param buf Destination.
 * @param size Bytes wanted.
 * @return Bytes read, fewer only at end of stream.
 */
size_t file_reader_read(file_reader *reader, void *buf, size_t size);

/**
 * @brief Move to an absolute offset
 *
 * Pipes only move forward by discarding, except within the magic bytes
 *
 * @param reader Open reader.
 * @param offset Byte offset from the start.
 * @return 0 on success, -1 on failure.
 */
int file_reader_seek(file_reader *reader, int64_t offset);

/**
 * @brief Close the reader
 *
 * @param reader Open reader.
 */
void file_reader_close(file_reader *reader);

#endif /* HQLC_CLI_FILE_READER_H */
