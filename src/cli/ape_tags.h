/**
 * APEv2 text tags, stored after the container trailer
 */
#ifndef HQLC_CLI_APE_TAGS_H
#define HQLC_CLI_APE_TAGS_H

#include <stddef.h>
#include <stdint.h>

// One text tag
typedef struct {
  char *key;   /**< Valid APEv2 key */
  char *value; /**< UTF-8, NUL-terminated */
} ape_tag;

// Tag list, zero-init is empty
typedef struct {
  ape_tag *items; /**< Tags in insertion order */
  size_t count;   /**< Number of tags */
} ape_tag_list;

/**
 * @brief Add a tag, or replace the value of an existing one.
 *
 * Keys compare case-insensitively.
 *
 * @param tags List to update.
 * @param key Key bytes, not NUL-terminated.
 * @param key_len Key length.
 * @param value Value bytes, not NUL-terminated.
 * @param value_len Value length.
 * @return 0 on success, -1 for an invalid key or out of memory.
 */
int ape_tags_set(ape_tag_list *tags,
                 const char *key,
                 size_t key_len,
                 const char *value,
                 size_t value_len);

/**
 * @brief Free a tag list, leaving it empty.
 *
 * @param tags List to free.
 */
void ape_tags_free(ape_tag_list *tags);

/**
 * @brief Build an APEv2 tag with header and footer.
 *
 * @param tags Tags to write.
 * @param len Receives the tag size in bytes.
 * @return A malloc'd buffer the caller frees, or NULL on failure.
 */
uint8_t *ape_tags_build(const ape_tag_list *tags, size_t *len);

/**
 * @brief Parse the text items of an APEv2 tag.
 *
 * @param data Tag bytes, the header is optional.
 * @param len Tag size in bytes.
 * @param tags Receives the items.
 * @return 0 on success, -1 if malformed.
 */
int ape_tags_parse(const uint8_t *data, size_t len, ape_tag_list *tags);

#endif /* HQLC_CLI_APE_TAGS_H */
