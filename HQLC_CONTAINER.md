# HQLC container spec

HQLC includes a basic container implementation in `hqlc_container.c`, supporting seeking, ape tags, and basic crc checks. The container is optional, and raw HQLC frames can be embedded wherever else.

The `.hqlc` file consists of a file header, pages of codec frames (64 samples each), a trailer page at the end and optional APEv2 tags. A frame is a single codec frame, 512 samples at 48kHz.

## File header (8 B)

| Size | Field |
|---|---|
| 4 | `"HQLC"` magic string |
| 1 | container version, `10` (older rejected) |
| 1 | flags, written 0, ignored |
| 2 | `pre_skip`, samples to skip at start for gapless playback |

## Page

| Size | Field |
|---|---|
| 2 | syncword `0x68 0x71` (`"hq"`) |
| 1 | `cfg`: bits 0–3 channels count (-1), bits 4–7 bitstream version |
| 4 | `frame_index` of the first frame in the page, multiple of 64 |
| ? | raw hqlc frame entries, with varint length prepended  |
| 2 | CRC-16 of the page from sync up to here |

Each page has 64 frames, except for the last one. Last page carries the end marker.

The first entry has its full size as a varint prefix (leb128). All entries that come after it encode a delta from the previous length, zigzag coded.
Length of zero is a special case, and is used as the end marker at the end of the file, followed by a pad value (0 to 511)

CRC-16 is the FLAC CRC variant, 0x8005 poly, see the `hqlc_container.c` for details.

## Trailer (20 B)

Included after the last page, needed for seeking

| Size | Field |
|---|---|
| 8 | `total_samples` per channel (after the trim of pad and pre-skip) |
| 8 | byte offset of the page holding the end marker, counted from the start of the file |
| 4 | `"HQLT"` magic string |

A file without a trailler still decodes, but seeking is not possible.

## Tag

Apev2 is supported at the end of file, after the trailer.

## Total sample count

Each frame has 512 samples per channel. The output has to drop `pre_skip` at start, and `pad` at the end.

## Seeking

Seeking preferably is implemented as interpolation and basic bissection (after parsing the trailler to get the total sample count and the pre-skip value form header).

See `hqlc_container.c` for details.
