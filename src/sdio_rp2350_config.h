// pico-llm SDIO configuration for SDIO_RP2350 library
// Pin assignment: CLK=10, CMD=11, D0-D3=12-15

#pragma once

// Disable SdFat integration (we use our own adapter in sdcard.c)
#define SDIO_USE_SDFAT 0

// Enable error logging only (debug is too verbose for normal operation)
#include <stdio.h>
static inline void sdio_log(const char *txt, uint32_t arg1, uint32_t arg2) {
    printf("SDIO: %s (%lu, %lu)\n", txt, (unsigned long)arg1, (unsigned long)arg2);
}
#define SDIO_CRITMSG(txt, arg1, arg2) sdio_log(txt, arg1, arg2)
#define SDIO_ERRMSG(txt, arg1, arg2) sdio_log(txt, arg1, arg2)

// PIO block and state machine
#define SDIO_PIO pio1
#define SDIO_SM  0

// GPIO function must match PIO block
#define SDIO_GPIO_FUNC GPIO_FUNC_PIO1
#define SDIO_GPIO_SLEW GPIO_SLEW_RATE_FAST
#define SDIO_GPIO_DRIVE GPIO_DRIVE_STRENGTH_8MA

// DMA channels
#define SDIO_DMACH_A 4
#define SDIO_DMACH_B 5
#define SDIO_DMAIRQ_IDX 1
#define SDIO_DMAIRQ DMA_IRQ_1

// GPIO pins (D0-D3 must be consecutive)
#define SDIO_CLK 10
#define SDIO_CMD 11
#define SDIO_D0  12
#define SDIO_D1  13
#define SDIO_D2  14
#define SDIO_D3  15
