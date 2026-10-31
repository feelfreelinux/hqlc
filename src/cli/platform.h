/**
 * Platform glue for the CLI
 */
#ifndef HQLC_CLI_PLATFORM_H
#define HQLC_CLI_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Return the program arguments as UTF-8
 *
 * On Windows rebuilt from the wide command line, elsewhere argv as is
 *
 * @param argc Argument count.
 * @param argv Arguments from main().
 * @return UTF-8 arguments.
 */
char **platform_args(int argc, char **argv);

/**
 * @brief Open a UTF-8 path in binary mode.
 *
 * @param path File path, or "-" for stdin and stdout.
 * @param for_writing True to open for writing.
 * @return The stream, or NULL on failure.
 */
FILE *platform_open(const char *path, bool for_writing);

/**
 * @brief Close a stream from platform_open(), stdin and stdout stay open.
 *
 * @param file Stream, NULL is ignored.
 */
void platform_close(FILE *file);

/**
 * @brief Tell regular files, which seek reliably, from pipes and consoles.
 *
 * @param file Open stream.
 * @param size Receives the file size when not NULL and the stream is a file.
 * @return True for a regular file.
 */
bool platform_is_file(FILE *file, int64_t *size);

/**
 * @brief Seek to an absolute offset, 64-bit safe.
 *
 * @param file Open stream.
 * @param offset Byte offset from the start.
 * @return 0 on success, nonzero on failure.
 */
int platform_seek(FILE *file, int64_t offset);

#endif /* HQLC_CLI_PLATFORM_H */
