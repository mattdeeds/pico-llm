#include "sdcard.h"
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"
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
#define CMD13  13  // SEND_STATUS
#define CMD16  16  // SET_BLOCKLEN
#define CMD17  17  // READ_SINGLE_BLOCK
#define CMD18  18  // READ_MULTIPLE_BLOCK
#define CMD24  24  // WRITE_BLOCK
#define CMD25  25  // WRITE_MULTIPLE_BLOCK
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

    // Power-cycle the SD card via LDO (GPIO8 = LDO_EN, active high)
    // In case JP2 bridges VDD to LDO output instead of direct 3.3V
    gpio_init(8);
    gpio_set_dir(8, GPIO_OUT);
    gpio_put(8, 0);          // Force LDO off
    busy_wait_us_32(250000); // Let caps discharge, card fully power down
    gpio_put(8, 1);          // LDO on
    busy_wait_us_32(500000); // Let card power up fully

    // Start at 400 kHz for card identification
    rp2350_sdio_init(rp2350_sdio_get_timing(SDIO_INITIALIZE));

    // Enable internal pull-ups on CMD and D0-D3 (pad register, independent of PIO)
    gpio_pull_up(SDIO_CMD);
    gpio_pull_up(SDIO_D0);
    gpio_pull_up(SDIO_D1);
    gpio_pull_up(SDIO_D2);
    gpio_pull_up(SDIO_D3);

    // Give card 74+ clock cycles to stabilize after power-up
    busy_wait_us_32(1000);

    // CMD0: GO_IDLE_STATE (no response)
    // Retry a few times to establish contact
    bool sd_v2 = false;
    for (int i = 0; i < 10; i++) {
        busy_wait_us_32(5000);
        rp2350_sdio_command(CMD0, 0, NULL, 0, 0);
        busy_wait_us_32(5000);
        status = rp2350_sdio_command_u32(CMD8, 0x1AA, &reply, 0);
        if (status == SDIO_OK && reply == 0x1AA) {
            sd_v2 = true;
            break;
        }
    }

    if (sd_v2) {
        printf("CMD8 OK (SD v2.0+)\n");
    } else {
        printf("CMD8 no response — assuming SD v1.x (status=%d)\n", status);
        // Re-send CMD0 to ensure card is in idle state
        rp2350_sdio_command(CMD0, 0, NULL, 0, 0);
        busy_wait_us_32(5000);
    }

    // ACMD41: wait for card initialization (up to 2 seconds)
    // SD v2.0+: set HCS (host supports SDHC); SD v1.x: no HCS
    absolute_time_t deadline = make_timeout_time_ms(2000);
    // Use the SDIO library's recommended OCR mode:
    // HCS (bit 30) + XPC max perf (bit 28) + 3.3V (bit 20)
    uint32_t ocr_arg = (sd_v2 ? ((1 << 30) | (1 << 28)) : 0) | (1 << 20);
    printf("ACMD41 arg: 0x%08lx\n", (unsigned long)ocr_arg);
    int acmd41_tries = 0;
    do {
        status = sd_cmd(CMD55, 0, &reply);
        if (status != SDIO_OK) { printf("SDIO: CMD55 fail\n"); return false; }
        status = sd_cmd_no_crc(ACMD41, ocr_arg, &card_ocr);
        if (status != SDIO_OK) { printf("SDIO: ACMD41 fail (status=%d)\n", status); return false; }
        acmd41_tries++;
        if (acmd41_tries <= 3 || (card_ocr & (1 << 31))) {
            printf("  #%d: CMD55_R1=0x%08lx OCR=0x%08lx\n",
                   acmd41_tries, (unsigned long)reply, (unsigned long)card_ocr);
        }
        busy_wait_us_32(10000); // 10ms between polls
        if (absolute_time_diff_us(get_absolute_time(), deadline) <= 0) {
            printf("ACMD41: timeout after %d tries (OCR=0x%08lx)\n",
                   acmd41_tries, (unsigned long)card_ocr);
            return false;
        }
    } while (!(card_ocr & (1 << 31)));

    // SD v2 with HCS → assume SDHC; SD v1.x → SDSC (byte addressing)
    card_sdhc = sd_v2;

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

    // CMD6: SWITCH_FUNC to SDR25 (50 MHz high-speed)
    // Argument 0x80FFFF01 = set mode + group 1 function 1 (High-Speed).
    // The card returns a 64-byte status block; byte 16 low nibble reports
    // the function selected for group 1: 0x1 = HS accepted, 0xF = rejected.
    rp2350_sdio_mode_t speed_mode = SDIO_STANDARD;
    uint8_t cmd6_status[64] __attribute__((aligned(4)));
    status = rp2350_sdio_command_u32(CMD6, 0x80FFFF01, &reply, SDIO_FLAG_STOP_CLK);
    if (status == SDIO_OK) {
        status = rp2350_sdio_rx_start(cmd6_status, 1, 64);
        if (status == SDIO_OK) {
            do {
                rp2350_sdio_poll_dma();
                status = rp2350_sdio_rx_poll(NULL);
            } while (status == SDIO_BUSY);
        }
        rp2350_sdio_stop();
        if (status == SDIO_OK) {
            busy_wait_us_32(1000);
            uint8_t g1_selected = cmd6_status[16] & 0x0F;
            if (g1_selected == 0x1) {
                speed_mode = SDIO_HIGHSPEED;
            } else {
                printf("SDIO: card refused HS switch (g1=0x%x), staying at 25 MHz\n",
                       g1_selected);
            }
        } else {
            printf("SDIO: CMD6 status read failed, staying at 25 MHz\n");
        }
    } else {
        printf("SDIO: CMD6 command failed, staying at 25 MHz\n");
    }

    rp2350_sdio_timing_t hs_timing = rp2350_sdio_get_timing(speed_mode);
    rp2350_sdio_init(hs_timing);

    // Set block length to 512 for data transfers
    sd_cmd(CMD16, 512, &reply);

    // HS smoke test: apr-13 showed multi-block reads fail at 50 MHz while
    // single-block reads pass. Do a 2-block CMD18 read of block 0; if it
    // trips CRC, fall back cleanly to 25 MHz instead of bricking inference.
    if (hs_timing.use_high_speed) {
        uint8_t smoke[2 * 512] __attribute__((aligned(4)));
        bool smoke_ok = sdcard_read_blocks(0, smoke, 2);
        if (!smoke_ok) {
            printf("SDIO: HS multi-block smoke read failed, falling back to 25 MHz\n");
            hs_timing = rp2350_sdio_get_timing(SDIO_STANDARD);
            rp2350_sdio_init(hs_timing);
            sd_cmd(CMD16, 512, &reply);
        }
    }

    uint32_t actual_khz = clock_get_hz(clk_sys) / hs_timing.data_clk_divider / 1000;
    printf("SDIO init OK (%s, %s, actual = %lu kHz)\n",
           card_sdhc ? "SDHC" : "SD",
           hs_timing.use_high_speed ? "high-speed 50MHz" : "standard 25MHz",
           (unsigned long)actual_khz);
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

        do {
            rp2350_sdio_poll_dma();
            status = rp2350_sdio_rx_poll(NULL);
        } while (status == SDIO_BUSY);
        rp2350_sdio_stop();
        return status == SDIO_OK;
    }

    // Multi-block read with retries
    for (int retry = 0; retry < 3; retry++) {
        if (retry > 0) {
            rp2350_sdio_stop();
            busy_wait_us_32(1000);
        }

        status = rp2350_sdio_command_u32(CMD18, address, &reply, SDIO_FLAG_STOP_CLK);
        if (status != SDIO_OK) continue;

        status = rp2350_sdio_rx_start(buf, count, 512);
        if (status != SDIO_OK) {
            rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
            continue;
        }

        do {
            rp2350_sdio_poll_dma();
            status = rp2350_sdio_rx_poll(NULL);
        } while (status == SDIO_BUSY);

        // CMD12: STOP_TRANSMISSION
        rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
        rp2350_sdio_stop();

        if (status == SDIO_OK) return true;
    }

    return false;
}

// Wait for the card to leave programming state after a write.
// Polls CMD13 (SEND_STATUS) until READY_FOR_DATA is set.
static void wait_card_ready(void) {
    uint32_t status_reg;
    for (int i = 0; i < 1000; i++) {
        sdio_status_t s = sd_cmd(CMD13, card_rca, &status_reg);
        if (s == SDIO_OK && (status_reg & (1 << 8))) // READY_FOR_DATA
            return;
        busy_wait_us_32(100);
    }
}

bool sdcard_write_blocks(uint32_t block_addr, const uint8_t *buf, uint32_t count) {
    uint32_t reply;
    sdio_status_t status;

    uint32_t address = card_sdhc ? block_addr : (block_addr * 512);

    if (count == 1) {
        status = rp2350_sdio_command_u32(CMD24, address, &reply, SDIO_FLAG_STOP_CLK);
        if (status != SDIO_OK) return false;

        status = rp2350_sdio_tx_start(buf, 1, 512);
        if (status != SDIO_OK) return false;

        do {
            rp2350_sdio_poll_dma();
            status = rp2350_sdio_tx_poll(NULL);
        } while (status == SDIO_BUSY);
        rp2350_sdio_stop();
        if (status == SDIO_OK) wait_card_ready();
        return status == SDIO_OK;
    }

    status = rp2350_sdio_command_u32(CMD25, address, &reply, SDIO_FLAG_STOP_CLK);
    if (status != SDIO_OK) return false;

    status = rp2350_sdio_tx_start(buf, count, 512);
    if (status != SDIO_OK) {
        rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
        rp2350_sdio_stop();
        return false;
    }

    do {
        rp2350_sdio_poll_dma();
        status = rp2350_sdio_tx_poll(NULL);
    } while (status == SDIO_BUSY);

    rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
    rp2350_sdio_stop();
    if (status == SDIO_OK) wait_card_ready();
    return status == SDIO_OK;
}

// ============================================================================
// Double-buffer weight streaming with async DMA
//
// Core 0 starts SD card DMA transfers (async via IRQ handler).
// Core 1 runs matmul compute on the active buffer.
// Overlap: DMA fills the prefetch buffer while Core 1 computes on the active one.
// ============================================================================

void weightbuf_init(WeightBuf *wb, int8_t *buf_a, int8_t *buf_b, uint32_t buf_size) {
    wb->buf_a = buf_a;
    wb->buf_b = buf_b;
    wb->active = buf_a;
    wb->prefetch = buf_b;
    wb->buf_size = buf_size;
    wb->dma_pending = false;
}

void weightbuf_start_prefetch(WeightBuf *wb, uint32_t sd_byte_offset, uint32_t size) {
    uint32_t block = sd_byte_offset / 512;
    uint32_t block_count = (sd_byte_offset % 512 + size + 511) / 512;
    // Clamp to buffer capacity to prevent overflow when offset is not block-aligned
    uint32_t max_blocks = wb->buf_size / 512;
    if (block_count > max_blocks) block_count = max_blocks;

    // Start async DMA transfer — returns immediately.
    // DMA IRQ handler on Core 0 chains blocks automatically.
    uint32_t address = card_sdhc ? block : (block * 512);
    uint32_t reply;

    sdio_status_t status = rp2350_sdio_command_u32(CMD18, address, &reply,
                                                    SDIO_FLAG_STOP_CLK);
    if (status != SDIO_OK) {
        printf("prefetch CMD18 fail at block %lu (status=%d)\n",
               (unsigned long)block, status);
        wb->dma_pending = false;
        return;
    }

    status = rp2350_sdio_rx_start((uint8_t *)wb->prefetch, block_count, 512);
    if (status != SDIO_OK) {
        printf("prefetch rx_start fail (status=%d)\n", status);
        rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
        rp2350_sdio_stop();
        wb->dma_pending = false;
        return;
    }

    wb->dma_pending = true;
}

int8_t *weightbuf_get(WeightBuf *wb) {
    if (wb->dma_pending) {
        // Poll until async DMA completes
        sdio_status_t status;
        do {
            rp2350_sdio_poll_dma();
            status = rp2350_sdio_rx_poll(NULL);
        } while (status == SDIO_BUSY);

        uint32_t reply;
        rp2350_sdio_command_u32(CMD12, 0, &reply, 0);
        rp2350_sdio_stop();
        wb->dma_pending = false;

        if (status != SDIO_OK) {
            printf("weightbuf_get: DMA error %d\n", status);
        }
    }

    // Swap active and prefetch buffers
    int8_t *filled = wb->prefetch;
    wb->prefetch = wb->active;
    wb->active = filled;
    return filled;
}

// ============================================================================
// Core 1 compute worker
//
// Runs matmul_q8_tile on demand. Core 0 sends (weights_ptr, n_rows) via FIFO.
// Core 1 accumulates into the shared accumulator, then signals done.
// ============================================================================

#include "quantize.h"

// Shared state — set by Core 0 before dispatching chunks
typedef struct {
    const int8_t *x_q;   // quantized input vector (int8 activations)
    float *acc;           // float accumulator (Q1_0_g128 has per-block dequant)
    int cols;             // matmul column count
    int rows_done;        // rows accumulated so far
} ComputeState;

ComputeState g_compute;

void compute_worker(void) {
    while (1) {
        uint32_t weights_ptr = multicore_fifo_pop_blocking();
        uint32_t n_rows = multicore_fifo_pop_blocking();

        matmul_q1_0_g128_tile(g_compute.acc + g_compute.rows_done,
                              (const uint8_t *)weights_ptr,
                              g_compute.x_q, (int)n_rows, g_compute.cols);
        g_compute.rows_done += (int)n_rows;

        multicore_fifo_push_blocking(1);
    }
}
