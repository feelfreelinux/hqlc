#include "ape_tags.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define APE_VERSION    2000
#define APE_HAS_HEADER 0x80000000u
#define APE_IS_HEADER  0x20000000u

// Item type bits, 0 = UTF-8 text
#define APE_TYPE_MASK 0x6u

// Header and footer size
#define APE_FRAME_BYTES 32

// Case-insensitive match of a key against a NUL-terminated one
static bool key_equals(const char *key, size_t key_len, const char *other) {
  for (size_t i = 0; i < key_len; i++) {
    if (tolower((unsigned char)key[i]) != tolower((unsigned char)other[i])) {
      return false;
    }
  }
  return other[key_len] == '\0';
}

// APEv2 keys validation (ASCII minus a few reserved names)
static bool key_is_valid(const char *key, size_t key_len) {
  static const char *reserved[] = {"ID3", "TAG", "OggS", "MP+"};
  if (key_len < 2 || key_len > 255) {
    return false;
  }
  for (size_t i = 0; i < key_len; i++) {
    if (key[i] < 0x20 || key[i] > 0x7E) {
      return false;
    }
  }
  for (size_t i = 0; i < sizeof(reserved) / sizeof(reserved[0]); i++) {
    if (key_equals(key, key_len, reserved[i])) {
      return false;
    }
  }
  return true;
}

// basically strndup, reimplement to avoid Windows shim
static char *copy_string(const char *text, size_t len) {
  char *copy = (char *)malloc(len + 1);
  if (copy) {
    memcpy(copy, text, len);
    copy[len] = '\0';
  }
  return copy;
}

int ape_tags_set(ape_tag_list *tags,
                 const char *key,
                 size_t key_len,
                 const char *value,
                 size_t value_len) {
  if (!key_is_valid(key, key_len)) {
    return -1;
  }
  char *value_copy = copy_string(value, value_len);
  if (!value_copy) {
    return -1;
  }
  for (size_t i = 0; i < tags->count; i++) {
    if (key_equals(key, key_len, tags->items[i].key)) {
      free(tags->items[i].value);
      tags->items[i].value = value_copy;
      return 0;
    }
  }
  ape_tag *grown = (ape_tag *)realloc(tags->items, (tags->count + 1) * sizeof(ape_tag));
  char *key_copy = copy_string(key, key_len);
  if (!grown || !key_copy) {
    if (grown) {
      tags->items = grown;
    }
    free(key_copy);
    free(value_copy);
    return -1;
  }
  tags->items = grown;
  tags->items[tags->count].key = key_copy;
  tags->items[tags->count].value = value_copy;
  tags->count++;
  return 0;
}

void ape_tags_free(ape_tag_list *tags) {
  for (size_t i = 0; i < tags->count; i++) {
    free(tags->items[i].key);
    free(tags->items[i].value);
  }
  free(tags->items);
  tags->items = NULL;
  tags->count = 0;
}

static void write_le32(uint8_t *dst, uint32_t value) {
  for (int i = 0; i < 4; i++) {
    dst[i] = (uint8_t)(value >> (8 * i));
  }
}

static uint32_t read_le32(const uint8_t *src) {
  return src[0] | (uint32_t)src[1] << 8 | (uint32_t)src[2] << 16 | (uint32_t)src[3] << 24;
}

// APEv2 header or footer, told apart by the flags
static void write_frame(uint8_t *dst, uint32_t tag_size, uint32_t count, uint32_t flags) {
  memset(dst, 0, APE_FRAME_BYTES);
  memcpy(dst, "APETAGEX", 8);
  write_le32(dst + 8, APE_VERSION);
  write_le32(dst + 12, tag_size);
  write_le32(dst + 16, count);
  write_le32(dst + 20, flags);
}

uint8_t *ape_tags_build(const ape_tag_list *tags, size_t *len) {
  size_t items_bytes = 0;
  for (size_t i = 0; i < tags->count; i++) {
    items_bytes += 8 + strlen(tags->items[i].key) + 1 + strlen(tags->items[i].value);
  }
  if (items_bytes > UINT32_MAX - 2 * APE_FRAME_BYTES) {
    return NULL;
  }
  // The size field covers the items and the footer, by APEv2 convention
  uint32_t tag_size = (uint32_t)items_bytes + APE_FRAME_BYTES;
  uint8_t *out = (uint8_t *)malloc(items_bytes + 2 * APE_FRAME_BYTES);
  if (!out) {
    return NULL;
  }
  uint8_t *pos = out;
  write_frame(pos, tag_size, (uint32_t)tags->count, APE_HAS_HEADER | APE_IS_HEADER);
  pos += APE_FRAME_BYTES;
  for (size_t i = 0; i < tags->count; i++) {
    size_t key_size = strlen(tags->items[i].key) + 1;
    size_t value_len = strlen(tags->items[i].value);
    write_le32(pos, (uint32_t)value_len);
    write_le32(pos + 4, 0);
    memcpy(pos + 8, tags->items[i].key, key_size);
    memcpy(pos + 8 + key_size, tags->items[i].value, value_len);
    pos += 8 + key_size + value_len;
  }
  write_frame(pos, tag_size, (uint32_t)tags->count, APE_HAS_HEADER);
  *len = items_bytes + 2 * APE_FRAME_BYTES;
  return out;
}

int ape_tags_parse(const uint8_t *data, size_t len, ape_tag_list *tags) {
  if (len < APE_FRAME_BYTES || memcmp(data + len - APE_FRAME_BYTES, "APETAGEX", 8) != 0) {
    return -1;
  }
  uint32_t count = read_le32(data + len - 16);
  const uint8_t *end = data + len - APE_FRAME_BYTES;
  const uint8_t *pos = data;
  if (memcmp(pos, "APETAGEX", 8) == 0 && len >= 2 * APE_FRAME_BYTES) {
    pos += APE_FRAME_BYTES;
  }
  for (uint32_t i = 0; i < count; i++) {
    if (end - pos < 9) {
      return -1;
    }
    uint32_t value_len = read_le32(pos);
    uint32_t flags = read_le32(pos + 4);
    const uint8_t *key = pos + 8;
    const uint8_t *key_end = (const uint8_t *)memchr(key, 0, (size_t)(end - key));
    if (!key_end || (size_t)(end - key_end - 1) < value_len) {
      return -1;
    }
    if (!(flags & APE_TYPE_MASK)) {
      ape_tags_set(tags,
                   (const char *)key,
                   (size_t)(key_end - key),
                   (const char *)key_end + 1,
                   value_len);
    }
    pos = key_end + 1 + value_len;
  }
  return 0;
}
