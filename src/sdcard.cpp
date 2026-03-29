#include "sdcard.h"
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include <string.h>
#include <stdio.h>

// ============================================================================
// SDIO adapter layer
//
// Wraps the SDIO_RP2350 low-level PIO driver into a simple block-read API.
// The library provides raw SDIO bus operations (command, rx, tx); this file
// implements the SD card initialization sequence and multi-block reads on top.
// ============================================================================

#include "sdio_rp2350.h"

// SD command indices
#define CMD0   0   // GO_IDLE_STATE
#define CMD2   2   // ALL_SEND_CID
#define CMD3   3   // SEND_RELATIVE_ADDR
#define CMD6   6   // SWITCH_FUNC
#define CMD7   7   // SELECT/DESELECT_CARD
#define CMD8   8   // SEND_IF_COND
#define CMD12  12  // STOP_TRANSMISSION
#define CMD16  16  // SET_BLOCKLEN
#define CMD17  17  // READ_SINGLE_BLOCK
#define CMD18  18  // READ_MULTIPLE_BLOCK
#define CMD55  55  // APP_CMD
#define ACMD6  6   // SET_BUS_WIDTH
#define ACMD41 41  // SD_SEND_OP_COND

// Card state
static uint32_t card_rca;    // Relative card address
static uint32_t card_ocr;    // Operating condition register
static bool card_sdhc;       // true = SDHC (sector addressing), false = byte addressing

static sdio_status_t sd_cmd(uint8_t cmd, uint32_t arg, uint32_t *resp) {
    return rp2350_sdio_command_u32(cmd, arg, resp, 0);
}

static sdio_status_t sd_cmd_no_crc(uint8_t cmd, uint32_t arg, uint32_t *resp) {
    return rp2350_sdio_command_u32(cmd, arg, resp, SDIO_FLAG_NO_CRC | SDIO_FLAG_NO_CMD_TAG);
}

bool sdcard_init(void) {
    uint32_t reply;
    sdio_status_t status;

    // Start at 400 kHz for card identification
    rp2350_sdio_init(rp2350_sdio_get_timing(SDIO_INITIALIZE));
    busy_wait_us_32(1000);

    // CMD0: GO_IDLE_STATE (no response)
    // Retry a few times to establish contact
    for (int i = 0; i < 5; i++) {
        busy_wait_us_32(1000);
        rp2350_sdio_command(CMD0, 0, NULL, 0, SDIO_FLAG_NO_LOGMSG);
        busy_wait_us_32(1000);
        status = rp2350_sdio_command_u32(CMD8, 0x1AA, &reply, SDIO_FLAG_NO_LOGMSG);
        if (status == SDIO_OK && reply == 0x1AA)
            break;
    }

    if (status != SDIO_OK || reply != 0x1AA) {
        printf("SDIO: no response to CMD8 (status=%d reply=0x%lx)\n",
               status, (unsigned long)reply);
        return false;
    }

    // ACMD41: wait for card initialization (up to 1 second)
    absolute_time_t deadline = make_timeout_time_ms(1000);
    uint32_t ocr_arg = (1 << 30) | (1 << 28) | (1 << 20); // HCS + max perf + 3.3V
    do {
        status = sd_cmd(CMD55, 0, &reply);
        if (status != SDIO_OK) { printf("SDIO: CMD55 fail\n"); return false; }
        status = sd_cmd_no_crc(ACMD41, ocr_arg, &card_ocr);
        if (status != SDIO_OK) { printf("SDIO: ACMD41 fail\n"); return false; }
        if (absolute_time_diff_us(get_absolute_time(), deadline) <= 0) {
            printf("SDIO: ACMD41 timeout\n");
            return false;
        }
    } while (!(card_ocr & (1 << 31)));

    card_sdhc = (card_ocr & (1 << 30)) != 0;

    // CMD2: ALL_SEND_CID (136-bit response, we don't need the content)
    uint8_t cid[16];
    status = rp2350_sdio_command(CMD2, 0, cid, 16, SDIO_FLAG_NO_CRC | SDIO_FLAG_NO_CMD_TAG);
    if (status != SDIO_OK) { printf("SDIO: CMD2 fail\n"); return false; }

    // CMD3: SEND_RELATIVE_ADDR
    status = sd_cmd(CMD3, 0, &card_rca);
    if (status != SDIO_OK) { printf("SDIO: CMD3 fail\n"); return false; }

    // CMD7: SELECT_CARD
    status = sd_cmd(CMD7, card_rca, &reply);
    if (status != SDIO_OK) { printf("SDIO: CMD7 fail\n"); return false; }

    // ACMD6: SET_BUS_WIDTH to 4-bit
    status = sd_cmd(CMD55, card_rca, &reply);
    if (status != SDIO_OK) { printf("SDIO: CMD55 fail\n"); return false; }
    status = sd_cmd(ACMD6, 2, &reply);
    if (status != SDIO_OK) { printf("SDIO: ACMD6 fail\n"); return false; }

    // Switch to high-speed 50 MHz
    // CMD6: SWITCH_FUNC — select SDR25 (function group 1 = 1)
    rp2350_sdio_timing_t hs_timing = rp2350_sdio_get_timing(SDIO_HIGHSPEED);
    if (hs_timing.use_high_speed) {
        uint8_t switch_status[64] __attribute__((aligned(4)));
        // Set block length for the 64-byte switch status
        sd_cmd(CMD16, 64, &reply);
        status = rp2350_sdio_command_u32(CMD6, 0x80FFFF01, &reply, SDIO_FLAG_STOP_CLK);
        if (status == SDIO_OK) {
            rp2350_sdio_rx_start(switch_status, 1, 64);
            sdio_status_t rx;
            do { rx = rp2350_sdio_rx_poll(NULL); } while (rx == SDIO_BUSY);
            rp2350_sdio_stop();
        }
    }

    // Apply high-speed clock
    rp2350_sdio_init(hs_timing);

    // Set block length to 512 for data transfers
    sd_cmd(CMD16, 512, &reply);

    printf("SDIO init OK (%s, %s)\n",
           card_sdhc ? "SDHC" : "SD",
           hs_timing.use_high_speed ? "high-speed" : "standard");
    return true;
}

bool sdcard_read_blocks(uint32_t block_addr, uint8_t *buf, uint32_t count) {
    uint32_t reply;
    sdio_status_t status;

    // SDHC uses sector addressing; older cards use byte addressing
    uint32_t address = card_sdhc ? block_addr : (block_addr * 512);

    if (count == 1) {
        // Single-block read
        status = rp2350_sdio_command_u32(CMD17, address, &reply, SDIO_FLAG_STOP_CLK);
        if (status != SDIO_OK) return false;

        status = rp2350_sdio_rx_start(buf, 1, 512);
        if (status != SDIO_OK) return false;

        do { status = rp2350_sdio_rx_poll(NULL); } while (status == SDIO_BUSY);
        rp2350_sdio_stop();
        return status == SDIO_OK;
    }

    // Multi-block read
    status = rp2350_sdio_command_u32(CMD18, address, &reply, SDIO_FLAG_STOP_CLK);
    if (status != SDIO_OK) return false;

    status = rp2350_sdio_rx_start(buf, count, 512);
    if (status != SDIO_OK) {
        rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
        rp2350_sdio_stop();
        return false;
    }

    do { status = rp2350_sdio_rx_poll(NULL); } while (status == SDIO_BUSY);

    // CMD12: STOP_TRANSMISSION
    rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
    rp2350_sdio_stop();

    return status == SDIO_OK;
}

// ============================================================================
// Double-buffer weight streaming
//
// Core 0 computes on the "active" buffer.
// Core 1 fills the "prefetch" buffer from SD in the background.
// When Core 0 calls weightbuf_get(), it blocks until prefetch is done,
// then swaps the buffers.
//
// Communication uses the RP2350 multicore FIFO:
//   Core 0 -> Core 1: FIFO push signals "start prefetch"
//   Core 1 -> Core 0: FIFO push signals "prefetch done"
// ============================================================================

// Shared state between cores (only written by one core at a time)
static WeightBuf *shared_wb;

void weightbuf_init(WeightBuf *wb, int8_t *buf_a, int8_t *buf_b, uint32_t buf_size) {
    wb->buf_a = buf_a;
    wb->buf_b = buf_b;
    wb->active = buf_a;
    wb->prefetch = buf_b;
    wb->buf_size = buf_size;
    shared_wb = wb;
}

void weightbuf_start_prefetch(WeightBuf *, uint32_t sd_byte_offset, uint32_t size) {
    uint32_t block = sd_byte_offset / 512;
    // Account for sub-block offset when computing how many blocks to read
    uint32_t block_count = (sd_byte_offset % 512 + size + 511) / 512;

    multicore_fifo_push_blocking(block);
    multicore_fifo_push_blocking(block_count);
}

int8_t *weightbuf_get(WeightBuf *wb) {
    // Wait for Core 1 to signal completion
    multicore_fifo_pop_blocking();

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
