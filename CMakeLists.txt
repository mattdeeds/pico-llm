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

# Model selection. Default is Qwen3-0.6B Q4_0; PICO_LLM_MODEL=bonsai builds the
# Bonsai-1.7B Q1_0_g128 variant from the same sources (see quantize.h). Kept so
# the two models can be compared on identical firmware -- same clocks, same
# profiling, same SD path -- which the original 38 s/tok run could not be.
#   cmake -B build-bonsai -DPICO_BOARD=pico2 -DPICO_LLM_PROFILE=ON -DPICO_LLM_MODEL=bonsai
set(PICO_LLM_MODEL "qwen3" CACHE STRING "Model: qwen3 or bonsai")

# Shared across both models: same layer count, head geometry and rope base.
target_compile_definitions(pico_llm PRIVATE
    N_LAYERS=28
    N_HEADS=16
    N_KV_HEADS=8
    MAX_SEQ_LEN=512
    WEIGHT_BUF_SIZE=32768
    HEAD_DIM=128
    ROPE_THETA=1000000.0f
    RMSNORM_EPS=1e-6f
)

if(PICO_LLM_MODEL STREQUAL "bonsai")
    target_compile_definitions(pico_llm PRIVATE
        PICO_LLM_Q1=1
        D_MODEL=2048
        HIDDEN_DIM=6144
        VOCAB_SIZE=151669
    )
    message(STATUS "pico-llm model: Bonsai-1.7B Q1_0_g128 (dim=2048 hidden=6144 vocab=151669)")
else()
    target_compile_definitions(pico_llm PRIVATE
        D_MODEL=1024
        HIDDEN_DIM=3072
        VOCAB_SIZE=151936
    )
    message(STATUS "pico-llm model: Qwen3-0.6B Q4_0 (dim=1024 hidden=3072 vocab=151936)")
endif()

option(PICO_LLM_PROFILE "Print a per-token timing breakdown over serial" OFF)
if(PICO_LLM_PROFILE)
    target_compile_definitions(pico_llm PRIVATE PICO_LLM_PROFILE=1)
endif()

# SDIO data clock divider override. The divider is picked at 200 MHz sysclk and
# is not recalculated when main() switches to 250 MHz, so the bus ends up at
# 250 / divider:  4 -> 62.5 MHz (out of spec),  5 -> 50 MHz (in spec).
# Leave unset for the default (4). Set to 5 to A/B against a compliant bus:
#   cmake -B build-62 -DPICO_LLM_PROFILE=ON
#   cmake -B build-50 -DPICO_LLM_PROFILE=ON -DSDIO_DATA_CLK_DIVIDER=5
set(SDIO_DATA_CLK_DIVIDER "" CACHE STRING "Override SDIO data clock divider (e.g. 5)")
if(NOT SDIO_DATA_CLK_DIVIDER STREQUAL "")
    target_compile_definitions(pico_llm PRIVATE
        SDIO_DATA_CLK_DIVIDER=${SDIO_DATA_CLK_DIVIDER})
endif()

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

# Raw SD read benchmark: isolates the SD path from inference to find the
# ~9.5 MB/s ceiling. Separate target so the production firmware is untouched.
#   cmake -B build-bench -DPICO_BOARD=pico2 -DPICO_LLM_BENCH=ON
#   cmake --build build-bench --target bench_sd
option(PICO_LLM_BENCH "Build the raw SD read benchmark" OFF)
if(PICO_LLM_BENCH)
    add_executable(bench_sd src/bench_sd.c src/sdcard.cpp src/quantize.c)
    target_include_directories(bench_sd PRIVATE src)
    target_compile_definitions(bench_sd PRIVATE
        D_MODEL=1024 HIDDEN_DIM=3072 VOCAB_SIZE=151936 WEIGHT_BUF_SIZE=32768
        N_HEADS=16 N_KV_HEADS=8 HEAD_DIM=128 N_LAYERS=28 MAX_SEQ_LEN=512
        ROPE_THETA=1000000.0f RMSNORM_EPS=1e-6f)
    target_compile_options(bench_sd PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(bench_sd
        pico_stdlib pico_stdio_usb hardware_dma hardware_gpio hardware_pio
        pico_multicore sdio_rp2350)
    pico_enable_stdio_uart(bench_sd 0)
    pico_enable_stdio_usb(bench_sd 1)
    pico_add_extra_outputs(bench_sd)
    # CRC attribution lives in the vendored driver, behind this guard.
    target_compile_definitions(sdio_rp2350 PRIVATE SDIO_PROFILE_CRC=1)
endif()

add_subdirectory(test)
