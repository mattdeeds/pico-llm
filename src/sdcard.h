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

// SDIO data clock divider chosen at init. The bus clock is sysclk / divider;
// it is not recalculated if sysclk changes after init.
int sdcard_data_clk_divider(void);

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

#ifdef PICO_LLM_PROFILE
// Time Core 1 has spent in the matmul kernel since the last reset.
// Written by Core 1; read and reset by Core 0 only while Core 1 is idle.
extern uint64_t g_core1_busy_us;

// SD error/retry counters, reset per token. Non-zero values mean the bus is
// unhealthy -- the prime suspect when the bus runs past the high-speed spec.
typedef struct {
    uint32_t blk_retries;  // sdcard_read_blocks() CRC retries (KV cache path)
    uint32_t pf_cmd_fail;  // weight prefetch: CMD18 rejected
    uint32_t pf_rx_fail;   // weight prefetch: rx_start rejected
    uint32_t pf_dma_err;   // weight prefetch: DMA completed with an error
} sd_counters_t;
extern sd_counters_t g_sd;

// Weight-stream time split, reset per token. cmd18 + stop is the fixed cost
// paid once per buffer; data is the part that scales with the bus clock.
extern uint64_t g_pf_cmd18_us;
extern uint64_t g_pf_data_us;
extern uint64_t g_pf_stop_us;
extern uint32_t g_pf_buffers;
#endif

#ifdef __cplusplus
}
#endif

#endif
