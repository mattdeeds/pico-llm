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
    D_MODEL=256
    N_LAYERS=6
    N_HEADS=4
    N_KV_HEADS=4
    VOCAB_SIZE=8192
    HIDDEN_DIM=704
    MAX_SEQ_LEN=512
    WEIGHT_BUF_SIZE=32768
)

# src/ must be on the SDIO library's include path so it finds sdio_rp2350_config.h
target_include_directories(sdio_rp2350 INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src)

target_compile_options(pico_llm PRIVATE -O2 -Wall -Wextra)

target_link_libraries(pico_llm
    pico_stdlib
    hardware_dma
    hardware_gpio
    hardware_pio
    pico_multicore
    sdio_rp2350
)

pico_enable_stdio_uart(pico_llm 1)
pico_enable_stdio_usb(pico_llm 0)

pico_add_extra_outputs(pico_llm)
