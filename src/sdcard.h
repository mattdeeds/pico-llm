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

// --- Double-buffer weight streaming with async DMA ---
// Core 0 fills the prefetch buffer via async DMA.
// Core 1 computes matmul on the active buffer.
// Overlap: DMA and compute run concurrently on different buffers.

typedef struct WeightBuf {
    int8_t *buf_a;
    int8_t *buf_b;
    int8_t *active;       // buffer being consumed
    int8_t *prefetch;     // buffer being filled by async DMA
    uint32_t buf_size;    // size of each buffer in bytes
    bool dma_pending;     // true if async DMA is in flight
} WeightBuf;

void weightbuf_init(WeightBuf *wb, int8_t *buf_a, int8_t *buf_b, uint32_t buf_size);

// Start async DMA read of 'size' bytes from SD into the prefetch buffer.
// Returns immediately — DMA IRQ handler on Core 0 chains blocks.
void weightbuf_start_prefetch(WeightBuf *wb, uint32_t sd_byte_offset, uint32_t size);

// Block until async DMA is done, then swap buffers and return the filled data.
int8_t *weightbuf_get(WeightBuf *wb);

// Core 1 compute worker entry point (runs matmul_q8_tile on demand)
void compute_worker(void);

#ifdef __cplusplus
}
#endif

#endif
