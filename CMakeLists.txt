cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico-llm C CXX ASM)

pico_sdk_init()

# SDIO_RP2350 library (PIO-based 4-bit SDIO)
# Clone into lib/SDIO_RP2350:
#   git submodule add https://github.com/rabbitholecomputing/SDIO_RP2350 lib/SDIO_RP2350
option(USE_SDIO "Enable SDIO_RP2350 SD card driver" OFF)

if(USE_SDIO)
    add_subdirectory(lib/SDIO_RP2350)
endif()

add_executable(pico_llm
    src/main.c
    src/transformer.c
    src/quantize.c
    src/sdcard.c
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

# SDIO pin configuration and library linkage
if(USE_SDIO)
    target_compile_definitions(pico_llm PRIVATE
        USE_SDIO_RP2350
        SDIO_CLK_PIN=10
        SDIO_CMD_PIN=11
        SDIO_DAT0_PIN=12
    )
    target_include_directories(pico_llm PRIVATE lib/SDIO_RP2350)
endif()

target_compile_options(pico_llm PRIVATE -O2 -Wall -Wextra)

target_link_libraries(pico_llm
    pico_stdlib
    hardware_dma
    hardware_gpio
    pico_multicore
)

if(USE_SDIO)
    target_link_libraries(pico_llm sdio_rp2350)
endif()

pico_enable_stdio_uart(pico_llm 1)
pico_enable_stdio_usb(pico_llm 0)

pico_add_extra_outputs(pico_llm)
