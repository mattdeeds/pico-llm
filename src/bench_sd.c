// Raw SD read benchmark for pico-llm.
//
// Isolates the SD read path from inference: no matmul, no KV cache, no
// sampling. Answers the question the per-token profile could not -- why the
// weight stream sustains only ~9.5 MB/s on a bus rated for 31.25 MB/s.
//
// Three things get measured:
//   1. Streaming throughput via the exact double-buffered prefetch path
//      production uses (weightbuf_start_prefetch / weightbuf_get), with the
//      compute removed. If this still reads ~9.5 MB/s, the ceiling is the
//      card or the bus, not the inference pipeline.
//   2. Throughput vs. request size, from 8 KB to 64 KB per CMD18. Flat means
//      per-request overhead is irrelevant (the per-token profile already said
//      0.83%); rising means bigger reads would help.
//   3. Time spent in the per-block software CRC16 (SDIO_PROFILE_CRC), which
//      runs on Core 0 in the DMA IRQ. This separates "slow card" from "CPU
//      can't checksum fast enough" -- the two candidates left standing.
//
// Build:
//   cmake -B build-bench -DPICO_BOARD=pico2 -DPICO_LLM_BENCH=ON
//   cmake --build build-bench --target bench_sd

#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "sdcard.h"

// Largest chunk we test. SDIO_MAX_BLOCKS_PER_REQ is 128 blocks = 64 KB, which
// is the hard ceiling for one CMD18 request in this driver. No model
// activations here, so two 64 KB buffers fit comfortably.
#define BENCH_MAX_CHUNK 65536

static int8_t buf_a[BENCH_MAX_CHUNK] __attribute__((aligned(4)));
static int8_t buf_b[BENCH_MAX_CHUNK] __attribute__((aligned(4)));

// Read this far into the card, past the model header, to mimic the weight
// stream's access pattern. 64 MB is enough to average out card variation
// without taking all day.
#define BENCH_START_BYTE (2u * 1024u * 1024u)
#define BENCH_BYTES      (64u * 1024u * 1024u)

// Per-block CRC16 attribution. These counters are incremented inside the
// vendored SDIO driver, which is a git submodule, so the instrumentation is NOT
// part of this repo. Defined weakly here so the benchmark links and runs
// against a stock submodule -- the CRC column then simply reads zero.
//
// To measure CRC cost, patch lib/SDIO_RP2350/src/sdio_rp2350.cpp to time the
// sdio_verify_rx_checksums() call in rp2350_sdio_dma_irq() behind
// -DSDIO_PROFILE_CRC, incrementing these two symbols. Measured on this board it
// came to ~5 us per 512-byte block: 9% of SD time on a slow card, 22% on a fast
// one, i.e. real but never the bottleneck.
__attribute__((weak)) uint64_t g_sdio_crc_us;
__attribute__((weak)) uint32_t g_sdio_crc_blocks;

static void reset_crc_counters(void) {
    g_sdio_crc_us = 0;
    g_sdio_crc_blocks = 0;
}

// Streaming read using the production double-buffer path, compute removed.
static void bench_stream(uint32_t chunk_bytes) {
    WeightBuf wb;
    weightbuf_init(&wb, buf_a, buf_b, chunk_bytes);
    reset_crc_counters();

    uint32_t off = BENCH_START_BYTE;
    uint32_t end = BENCH_START_BYTE + BENCH_BYTES;
    uint32_t n_chunks = 0;
    volatile int8_t sink = 0;

    uint64_t t0 = time_us_64();

    // Prime the pipeline, then keep one read in flight at all times -- exactly
    // what ws_ensure() does during inference.
    weightbuf_start_prefetch(&wb, off, chunk_bytes);
    while (off < end) {
        int8_t *filled = weightbuf_get(&wb);
        off += chunk_bytes;
        n_chunks++;
        if (off < end) {
            weightbuf_start_prefetch(&wb, off, chunk_bytes);
        }
        // Touch the data so the read cannot be optimized away, but do far less
        // work than a matmul so compute never becomes the limiter.
        sink ^= filled[0] ^ filled[chunk_bytes - 1];
    }

    uint64_t elapsed_us = time_us_64() - t0;
    (void)sink;

    uint32_t mb_x100 = (uint32_t)((uint64_t)BENCH_BYTES * 100u / elapsed_us);
    uint32_t crc_pct = elapsed_us ? (uint32_t)(g_sdio_crc_us * 100 / elapsed_us) : 0;
    uint32_t per_blk_ns = g_sdio_crc_blocks
                        ? (uint32_t)(g_sdio_crc_us * 1000 / g_sdio_crc_blocks) : 0;

    printf("  chunk=%5lu B  chunks=%4lu  %llu ms  %lu.%02lu MB/s  "
           "crc=%llu ms (%lu%%, %lu blocks, %lu ns/blk)\n",
           (unsigned long)chunk_bytes, (unsigned long)n_chunks,
           (unsigned long long)(elapsed_us / 1000),
           (unsigned long)(mb_x100 / 100), (unsigned long)(mb_x100 % 100),
           (unsigned long long)(g_sdio_crc_us / 1000),
           (unsigned long)crc_pct, (unsigned long)g_sdio_crc_blocks,
           (unsigned long)per_blk_ns);
}

// Blocking read, no double buffering -- the serial-cost baseline.
static void bench_blocking(uint32_t blocks_per_req) {
    reset_crc_counters();
    uint32_t bytes = BENCH_BYTES / 8;  // shorter; this path is slower by design
    uint32_t block = BENCH_START_BYTE / 512;
    uint32_t n_reqs = bytes / (blocks_per_req * 512);

    uint64_t t0 = time_us_64();
    for (uint32_t i = 0; i < n_reqs; i++) {
        if (!sdcard_read_blocks(block, (uint8_t *)buf_a, blocks_per_req)) {
            printf("  READ FAILED at block %lu\n", (unsigned long)block);
            return;
        }
        block += blocks_per_req;
    }
    uint64_t elapsed_us = time_us_64() - t0;

    uint32_t mb_x100 = (uint32_t)((uint64_t)bytes * 100u / elapsed_us);
    printf("  blocking %3lu blk/req (%lu KB)  %llu ms  %lu.%02lu MB/s\n",
           (unsigned long)blocks_per_req, (unsigned long)(blocks_per_req / 2),
           (unsigned long long)(elapsed_us / 1000),
           (unsigned long)(mb_x100 / 100), (unsigned long)(mb_x100 % 100));
}

int main(void) {
    // Match production: init at 200 MHz for an exact PIO divider, then overclock.
    set_sys_clock_khz(200000, true);
    stdio_init_all();
    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }
    sleep_ms(200);

    printf("\n=== pico-llm raw SD read benchmark ===\n");

    if (!sdcard_init()) {
        printf("FATAL: SD init failed\n");
        while (1) tight_loop_contents();
    }

    // Same as production: overclock after init, divider is not recalculated.
    set_sys_clock_khz(250000, true);
    sleep_ms(200);

    int div = sdcard_data_clk_divider();
    uint32_t bus_khz = clock_get_hz(clk_sys) / div / 1000;
    printf("sysclk=%lu kHz  divider=%d  bus=%lu kHz  theoretical=%lu.%02lu MB/s\n",
           (unsigned long)(clock_get_hz(clk_sys) / 1000), div,
           (unsigned long)bus_khz,
           (unsigned long)(bus_khz / 2000), (unsigned long)((bus_khz / 20) % 100));
    printf("reading %lu MB per test from byte %lu\n\n",
           (unsigned long)(BENCH_BYTES / 1024 / 1024),
           (unsigned long)BENCH_START_BYTE);

    printf("Streaming (production double-buffer path, no compute):\n");
    bench_stream(8192);
    bench_stream(16384);
    bench_stream(32768);   // production configuration
    bench_stream(65536);   // driver ceiling: SDIO_MAX_BLOCKS_PER_REQ

    printf("\nBlocking (single request at a time):\n");
    bench_blocking(16);
    bench_blocking(64);
    bench_blocking(128);   // SDIO_MAX_BLOCKS_PER_REQ

    printf("\n=== done ===\n");
    while (1) tight_loop_contents();
    return 0;
}
