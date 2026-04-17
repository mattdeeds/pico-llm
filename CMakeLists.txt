cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico-llm C CXX ASM)

pico_sdk_init()

# SDIO_RP2350 library (PIO-based 4-bit SDIO)
add_subdirectory(lib/SDIO_RP2350)

add_executable(pico_llm
    src/main.c
    src/transformer.c
    src/quantize.c
    src/sdcard.cpp
    src/tokenizer.c
)

target_include_directories(pico_llm PRIVATE src)

target_compile_definitions(pico_llm PRIVATE
    D_MODEL=1024
    N_LAYERS=28
    N_HEADS=16
    N_KV_HEADS=8
    VOCAB_SIZE=151936
    HIDDEN_DIM=3072
    MAX_SEQ_LEN=512
    WEIGHT_BUF_SIZE=32768
    HEAD_DIM=128
    ROPE_THETA=1000000.0f
    RMSNORM_EPS=1e-6f
)

# src/ must be on the SDIO library's include path so it finds sdio_rp2350_config.h
# PRIVATE so the library itself sees the config during compilation (not just consumers)
target_include_directories(sdio_rp2350 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)

# Stack for forward pass VLAs (attn_norm[2048]=8KB, hb_q[6144]=6KB)
set(PICO_STACK_SIZE 32768)

target_compile_options(pico_llm PRIVATE -O2 -Wall -Wextra)
set_source_files_properties(src/quantize.c
    PROPERTIES COMPILE_OPTIONS "-O3;-funroll-loops")

target_link_libraries(pico_llm
    pico_stdlib
    pico_stdio_usb
    hardware_dma
    hardware_gpio
    hardware_pio
    pico_multicore
    sdio_rp2350
)

pico_enable_stdio_uart(pico_llm 0)
pico_enable_stdio_usb(pico_llm 1)

pico_add_extra_outputs(pico_llm)

add_subdirectory(test)
