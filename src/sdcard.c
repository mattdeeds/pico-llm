#include "sdcard.h"
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include <string.h>
#include <stdio.h>

// ============================================================================
// SDIO adapter layer
//
// This wraps the SDIO_RP2350 library. The library header is expected at
// lib/SDIO_RP2350/sdio_rp2350.h. Adapt the function names below if the
// actual API differs.
// ============================================================================

#ifdef USE_SDIO_RP2350

#include "sdio_rp2350.h"

bool sdcard_init(void) {
    // Initialize SDIO with PIO-based 4-bit mode
    // The SDIO_RP2350 library handles:
    //   - PIO program loading
    //   - DMA channel allocation
    //   - Card init sequence (CMD0, CMD8, ACMD41, CMD2, CMD3, CMD7)
    //   - Switching to 4-bit mode and high-speed (50 MHz)
    sdio_status_t status = sdio_init(SDIO_CLK_PIN, SDIO_CMD_PIN, SDIO_DAT0_PIN);
    if (status != SDIO_OK) {
        printf("SDIO init failed: %d\n", status);
        return false;
    }
    printf("SDIO init OK (4-bit mode, 50 MHz)\n");
    return true;
}

bool sdcard_read_blocks(uint32_t block_addr, uint8_t *buf, uint32_t count) {
    sdio_status_t status = sdio_read_blocks(block_addr, buf, count);
    return status == SDIO_OK;
}

#else

// ============================================================================
// Stub implementation (no hardware)
// ============================================================================

bool sdcard_init(void) {
    printf("SD card: using stub (no SDIO hardware)\n");
    return false;
}

bool sdcard_read_blocks(uint32_t block_addr, uint8_t *buf, uint32_t count) {
    (void)block_addr;
    (void)buf;
    (void)count;
    return false;
}

#endif // USE_SDIO_RP2350

// ============================================================================
// Double-buffer weight streaming
//
// Core 0 computes on the "active" buffer.
// Core 1 fills the "prefetch" buffer from SD in the background.
// When Core 0 calls weightbuf_get(), it blocks until prefetch is done,
// then swaps the buffers.
//
// Communication uses the RP2350 multicore FIFO:
//   Core 0 → Core 1: FIFO push signals "start prefetch"
//   Core 1 → Core 0: FIFO push signals "prefetch done"
// ============================================================================

// Shared state between cores (only written by one core at a time)
static WeightBuf *shared_wb;

void weightbuf_init(WeightBuf *wb, int8_t *buf_a, int8_t *buf_b, uint32_t buf_size) {
    wb->buf_a = buf_a;
    wb->buf_b = buf_b;
    wb->active = buf_a;
    wb->prefetch = buf_b;
    wb->buf_size = buf_size;
    wb->prefetch_block = 0;
    wb->prefetch_bytes = 0;
    wb->prefetch_ready = false;
    shared_wb = wb;
}

void weightbuf_start_prefetch(WeightBuf *wb, uint32_t sd_byte_offset, uint32_t size) {
    // Convert byte offset to 512-byte block address
    wb->prefetch_block = sd_byte_offset / 512;
    wb->prefetch_bytes = size;
    wb->prefetch_ready = false;

    // Signal Core 1 to begin prefetch
    // Pack block addr and block count into two FIFO words
    uint32_t block_count = (size + 511) / 512;
    multicore_fifo_push_blocking(wb->prefetch_block);
    multicore_fifo_push_blocking(block_count);
}

int8_t *weightbuf_get(WeightBuf *wb) {
    // Wait for Core 1 to signal completion
    uint32_t done = multicore_fifo_pop_blocking();
    (void)done;

    wb->prefetch_ready = true;

    // Swap active and prefetch buffers
    int8_t *filled = wb->prefetch;
    wb->prefetch = wb->active;
    wb->active = filled;

    return filled;
}

void prefetch_worker(void) {
    WeightBuf *wb = shared_wb;

    while (1) {
        // Wait for Core 0 to send a prefetch request
        uint32_t block_addr = multicore_fifo_pop_blocking();
        uint32_t block_count = multicore_fifo_pop_blocking();

        // Read from SD card into the prefetch buffer
        bool ok = sdcard_read_blocks(block_addr, (uint8_t *)wb->prefetch, block_count);
        if (!ok) {
            printf("prefetch_worker: read failed at block %lu\n",
                   (unsigned long)block_addr);
        }

        // Signal Core 0 that data is ready
        multicore_fifo_push_blocking(1);
    }
}
