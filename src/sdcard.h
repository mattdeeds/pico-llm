#ifndef SDCARD_H
#define SDCARD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize SD card via SDIO (PIO-based 4-bit mode)
// Pin assignments are in sdio_rp2350_config.h
bool sdcard_init(void);

// Read contiguous blocks from SD card into buffer
// block_addr: 512-byte block number, count: number of blocks
bool sdcard_read_blocks(uint32_t block_addr, uint8_t *buf, uint32_t count);

// Write contiguous blocks from buffer to SD card
// block_addr: 512-byte block number, count: number of blocks
// buf must be 4-byte aligned
bool sdcard_write_blocks(uint32_t block_addr, const uint8_t *buf, uint32_t count);

// --- Double-buffer weight streaming ---
// Manages two weight buffers. Core 1 prefetches into the inactive buffer
// while Core 0 computes on the active buffer.

typedef struct WeightBuf {
    int8_t *buf_a;
    int8_t *buf_b;
    int8_t *active;       // buffer Core 0 is computing on
    int8_t *prefetch;     // buffer Core 1 is filling
    uint32_t buf_size;    // size of each buffer in bytes
} WeightBuf;

void weightbuf_init(WeightBuf *wb, int8_t *buf_a, int8_t *buf_b, uint32_t buf_size);

// Request Core 1 to prefetch 'size' bytes starting at SD byte offset
void weightbuf_start_prefetch(WeightBuf *wb, uint32_t sd_byte_offset, uint32_t size);

// Block until prefetch is done, swap buffers, return pointer to the filled data
int8_t *weightbuf_get(WeightBuf *wb);

// Core 1 prefetch worker entry point
void prefetch_worker(void);

#ifdef __cplusplus
}
#endif

#endif
