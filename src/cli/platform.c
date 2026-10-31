#define _FILE_OFFSET_BITS 64
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "platform.h"

#include <stdlib.h>
#include <string.h>

// Append a malloc'd copy of path to a growing array
static int add_path(char ***paths, size_t *count, const char *path) {
  char **grown = (char **)realloc(*paths, (*count + 1) * sizeof(char *));
  if (!grown) {
    return -1;
  }
  *paths = grown;
  size_t len = strlen(path) + 1;
  char *copy = (char *)malloc(len);
  if (!copy) {
    return -1;
  }
  memcpy(copy, path, len);
  grown[*count] = copy;
  (*count)++;
  return 0;
}

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <shellapi.h>

// malloc'd UTF-16 copy of a UTF-8 string, NULL on failure
static wchar_t *utf8_to_wide(const char *text) {
  int len = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
  wchar_t *wide = NULL;
  if (len > 0) {
    wide = (wchar_t *)malloc((size_t)len * sizeof(wchar_t));
  }
  if (wide && MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, len) <= 0) {
    free(wide);
    wide = NULL;
  }
  return wide;
}

// malloc'd UTF-8 copy of a UTF-16 string, NULL on failure
static char *wide_to_utf8(const wchar_t *wide) {
  int len = WideCharToMultiByte(CP_UTF8, 0, wide, -1, NULL, 0, NULL, NULL);
  char *text = NULL;
  if (len > 0) {
    text = (char *)malloc((size_t)len);
  }
  if (text && WideCharToMultiByte(CP_UTF8, 0, wide, -1, text, len, NULL, NULL) <= 0) {
    free(text);
    text = NULL;
  }
  return text;
}

char **platform_args(int argc, char **argv) {
  SetConsoleOutputCP(CP_UTF8);
  int count;
  wchar_t **wide = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!wide || count != argc) {
    return argv;
  }
  // Lives for the whole program
  char **args = (char **)calloc((size_t)count + 1, sizeof(char *));
  bool ok = args != NULL;
  for (int i = 0; ok && i < count; i++) {
    args[i] = wide_to_utf8(wide[i]);
    ok = args[i] != NULL;
  }
  LocalFree(wide);
  if (ok) {
    return args;
  }
  return argv;
}

// fopen for a UTF-8 path, through the wide API
static FILE *open_file(const char *path, bool for_writing) {
  wchar_t *wide = utf8_to_wide(path);
  FILE *file = NULL;
  if (wide) {
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

int platform_glob(const char *pattern, char ***paths, size_t *count) {
  *paths = NULL;
  *count = 0;
  if (!strpbrk(pattern, "*?")) {
    return add_path(paths, count, pattern);
  }
  wchar_t *wide = utf8_to_wide(pattern);
  if (!wide) {
    return -1;
  }
  WIN32_FIND_DATAW found;
  HANDLE find = FindFirstFileW(wide, &found);
  free(wide);
  if (find == INVALID_HANDLE_VALUE) {
    if (GetLastError() == ERROR_FILE_NOT_FOUND) {
      return 0;
    }
    return -1;
  }
  // Matches are bare names, the pattern's directory part goes in front
  size_t dir_len = 0;
  for (const char *p = pattern; *p; p++) {
    if (*p == '/' || *p == '\\') {
      dir_len = (size_t)(p - pattern) + 1;
    }
  }
  int status = 0;
  do {
    if (found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      continue;
    }
    char *name = wide_to_utf8(found.cFileName);
    char *path = NULL;
    if (name) {
      path = (char *)malloc(dir_len + strlen(name) + 1);
    }
    if (path) {
      memcpy(path, pattern, dir_len);
      strcpy(path + dir_len, name);
      status = add_path(paths, count, path);
    } else {
      status = -1;
    }
    free(name);
    free(path);
  } while (status == 0 && FindNextFileW(find, &found));
  FindClose(find);
  return status;
}

#else
#include <glob.h>
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

int platform_glob(const char *pattern, char ***paths, size_t *count) {
  *paths = NULL;
  *count = 0;
  if (!strpbrk(pattern, "*?[")) {
    return add_path(paths, count, pattern);
  }
  glob_t found;
  int status = glob(pattern, 0, NULL, &found);
  if (status == GLOB_NOMATCH) {
    return 0;
  }
  if (status != 0) {
    return -1;
  }
  status = 0;
  for (size_t i = 0; status == 0 && i < found.gl_pathc; i++) {
    status = add_path(paths, count, found.gl_pathv[i]);
  }
  globfree(&found);
  return status;
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
