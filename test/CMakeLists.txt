add_executable(test_matmul
    test_matmul.c
    ${CMAKE_SOURCE_DIR}/src/quantize.c
)

target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_compile_definitions(test_matmul PRIVATE
    D_MODEL=256
    WEIGHT_BUF_SIZE=32768
)

target_compile_options(test_matmul PRIVATE -O2 -Wall -Wextra)

target_link_libraries(test_matmul pico_stdlib)

pico_enable_stdio_uart(test_matmul 1)
pico_enable_stdio_usb(test_matmul 0)

pico_add_extra_outputs(test_matmul)
