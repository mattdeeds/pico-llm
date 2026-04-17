#include <stdio.h>
#include <string.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "quantize.h"

// DWT cycle counter via raw MMIO (avoids CMSIS header dependency)
#define DWT_CTRL_REG   (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT_REG (*(volatile uint32_t *)0xE0001004)
#define DEMCR_REG      (*(volatile uint32_t *)0xE000EDFC)
#define DEMCR_TRCENA   (1u << 24)
#define DWT_CYCCNTENA  (1u << 0)

static void dwt_init(void) {
    DEMCR_REG |= DEMCR_TRCENA;
    DWT_CYCCNT_REG = 0;
    DWT_CTRL_REG |= DWT_CYCCNTENA;
}

static inline uint32_t dwt_read(void) {
    return DWT_CYCCNT_REG;
}

#define BENCH_ROWS 4
#define BENCH_COLS 2048
#define BLOCKS_PER_ROW (BENCH_COLS / 128)
#define ROW_BYTES (BLOCKS_PER_ROW * 18)
#define WEIGHT_BYTES (BENCH_ROWS * ROW_BYTES)

static uint8_t weights[WEIGHT_BYTES] __attribute__((aligned(4)));
static int8_t  x_q[BENCH_COLS] __attribute__((aligned(4)));
static float   acc_ref[BENCH_ROWS];
static float   acc_test[BENCH_ROWS];

static void fill_test_data(void) {
    uint32_t s = 0xDEADBEEF;
    for (int i = 0; i < WEIGHT_BYTES; i++) {
        s = s * 1103515245 + 12345;
        weights[i] = (uint8_t)(s >> 16);
    }
    for (int i = 0; i < BENCH_COLS; i++) {
        s = s * 1103515245 + 12345;
        x_q[i] = (int8_t)((s >> 16) % 255 - 127);
    }
}

typedef void (*matmul_fn)(float *, const uint8_t *, const int8_t *, int, int);

static void bench(const char *name, matmul_fn fn, int iters) {
    fn(acc_test, weights, x_q, BENCH_ROWS, BENCH_COLS);

    uint32_t start = dwt_read();
    for (int i = 0; i < iters; i++) {
        fn(acc_test, weights, x_q, BENCH_ROWS, BENCH_COLS);
    }
    uint32_t end = dwt_read();

    uint32_t total_cycles = end - start;
    uint32_t total_weights = (uint32_t)BENCH_ROWS * BENCH_COLS * iters;
    float cycles_per_weight = (float)total_cycles / (float)total_weights;

    printf("  %-12s  %10lu cycles  %5.2f cyc/wt  acc[0]=%.2f\n",
           name, (unsigned long)total_cycles, (double)cycles_per_weight,
           (double)acc_test[0]);
}

static void check(const char *name, matmul_fn fn) {
    fn(acc_test, weights, x_q, BENCH_ROWS, BENCH_COLS);
    float max_err = 0.0f;
    for (int i = 0; i < BENCH_ROWS; i++) {
        float err = fabsf(acc_test[i] - acc_ref[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err < 0.01f) {
        printf("  %-12s  OK (max_err=%.4f)\n", name, (double)max_err);
    } else {
        printf("  %-12s  FAIL max_err=%.4f\n", name, (double)max_err);
        for (int i = 0; i < BENCH_ROWS; i++) {
            printf("    row %d: ref=%.4f got=%.4f\n", i,
                   (double)acc_ref[i], (double)acc_test[i]);
        }
    }
}

int main(void) {
    stdio_init_all();
    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }
    sleep_ms(500);

    printf("\n=== Q1_0_g128 matmul benchmark ===\n");
    printf("tile_rows=%d  cols=%d  weights/call=%d\n\n",
           BENCH_ROWS, BENCH_COLS, BENCH_ROWS * BENCH_COLS);

    dwt_init();

    uint32_t t0 = dwt_read();
    uint32_t t1 = dwt_read();
    printf("DWT overhead: %lu cycles\n\n", (unsigned long)(t1 - t0));

    fill_test_data();

    matmul_q1_0_g128_tile(acc_ref, weights, x_q, BENCH_ROWS, BENCH_COLS);

    printf("Correctness:\n");
    check("baseline", matmul_q1_0_g128_tile);
    check("lut",      matmul_q1_0_g128_tile_lut);
    check("dsp",      matmul_q1_0_g128_tile_dsp);
    check("dsp2",     matmul_q1_0_g128_tile_dsp2);

    int iters = 100;
    printf("\nTiming (%d iters each):\n", iters);
    bench("baseline", matmul_q1_0_g128_tile,      iters);
    bench("lut",      matmul_q1_0_g128_tile_lut,   iters);
    bench("dsp",      matmul_q1_0_g128_tile_dsp,   iters);
    bench("dsp2",     matmul_q1_0_g128_tile_dsp2,  iters);

    printf("\nDone.\n");
    while (1) {
        tight_loop_contents();
    }
    return 0;
}
