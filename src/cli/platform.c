#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "platform.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <shellapi.h>

char **platform_args(int argc, char **argv) {
  SetConsoleOutputCP(CP_UTF8);
  int count;
  wchar_t **wide = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!wide || count != argc) {
    return argv;
  }
  // Lives for the whole program
  char **args = (char **)calloc((size_t)count + 1, sizeof(char *));
  for (int i = 0; args && i < count; i++) {
    int len = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, NULL, 0, NULL, NULL);
    if (!(args[i] = (char *)malloc((size_t)len))) {
      return argv;
    }
    WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, args[i], len, NULL, NULL);
  }
  LocalFree(wide);
  if (args) {
    return args;
  }
  return argv;
}

// fopen for a UTF-8 path, through the wide API
static FILE *open_file(const char *path, bool for_writing) {
  int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
  wchar_t *wide = NULL;
  if (len > 0) {
    wide = (wchar_t *)malloc((size_t)len * sizeof(wchar_t));
  }
  FILE *file = NULL;
  if (wide && MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, len) > 0) {
    file = _wfopen(wide, for_writing ? L"wb" : L"rb");
  }
  free(wide);
  return file;
}

bool platform_is_file(FILE *file, int64_t *size) {
  HANDLE handle = (HANDLE)_get_osfhandle(_fileno(file));
  LARGE_INTEGER large;
  if (GetFileType(handle) != FILE_TYPE_DISK || !GetFileSizeEx(handle, &large)) {
    return false;
  }
  if (size) {
    *size = large.QuadPart;
  }
  return true;
}

int platform_seek(FILE *file, int64_t offset) {
  return _fseeki64(file, offset, SEEK_SET);
}

#else
#include <sys/stat.h>

char **platform_args(int argc, char **argv) {
  (void)argc;
  return argv;
}

static FILE *open_file(const char *path, bool for_writing) {
  return fopen(path, for_writing ? "wb" : "rb");
}

bool platform_is_file(FILE *file, int64_t *size) {
  struct stat st;
  if (fstat(fileno(file), &st) != 0 || !S_ISREG(st.st_mode)) {
    return false;
  }
  if (size) {
    *size = (int64_t)st.st_size;
  }
  return true;
}

int platform_seek(FILE *file, int64_t offset) {
  return fseeko(file, (off_t)offset, SEEK_SET);
}
#endif

FILE *platform_open(const char *path, bool for_writing) {
  FILE *file;
  if (strcmp(path, "-") == 0) {
    file = for_writing ? stdout : stdin;
#ifdef _WIN32
    _setmode(_fileno(file), _O_BINARY);
#endif
  } else {
    file = open_file(path, for_writing);
  }
  if (file) {
    setvbuf(file, NULL, _IOFBF, 1 << 16);
  }
  return file;
}

void platform_close(FILE *file) {
  if (file && file != stdin && file != stdout) {
    fclose(file);
  }
}
