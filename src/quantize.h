#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantize a float vector to int8, returns the scale factor used
float quantize_vec(int8_t *out, const float *in, int size);

// Q1_0_g128 tiled matmul: weights are Q1_0_g128 blocks (18 bytes per 128 values).
// For each row: (cols/128) blocks of [float16 scale + 16 sign bytes].
// In each sign byte, bit k selects weight at position (j*8 + k) within the block:
//   bit = 1 → +scale, bit = 0 → -scale.
// Inner loop: sign × int8 → int32, multiplied per-block by the float16 scale.
// Output is float (per-block dequantization is built-in).
void matmul_q1_0_g128_tile(float *acc, const uint8_t *weights,
                           const int8_t *x_q, int tile_rows, int cols);

// Q4_0 tiled matmul: weights are Q4_0 blocks (18 bytes per 32 values).
// For each row: (cols/32) blocks of [float16 scale + 16 packed nibble bytes].
// Nibble packing: qs[j] low nibble = weight[j], high nibble = weight[j+16].
// Dequant: weight_value = (nibble - 8) * scale.
// Inner loop: int4×int8 → int32, multiplied per-block by the float16 scale.
void matmul_q4_0_tile(float *acc, const uint8_t *weights,
                      const int8_t *x_q, int tile_rows, int cols);

// Variant kernels for benchmarking (same signature as matmul_q1_0_g128_tile)
void matmul_q1_0_g128_tile_lut(float *acc, const uint8_t *weights,
                                const int8_t *x_q, int tile_rows, int cols);
void matmul_q1_0_g128_tile_dsp(float *acc, const uint8_t *weights,
                                const int8_t *x_q, int tile_rows, int cols);
void matmul_q1_0_g128_tile_dsp2(float *acc, const uint8_t *weights,
                                 const int8_t *x_q, int tile_rows, int cols);

// Convert IEEE 754 float16 (half-precision) to float32
float fp16_to_fp32(uint16_t h);

// ---------------------------------------------------------------------------
// Compile-time quantization format selection.
//
// Both formats store 18 bytes per block; they differ only in how many weights
// a block covers (32 for Q4_0, 128 for Q1_0_g128). Everything downstream --
// row sizes, layout offsets, the embedding reader, the streaming matmul --
// depends on the format only through these two macros, so one source tree
// builds either model.
//
// Build with -DPICO_LLM_Q1 for Bonsai-1.7B (Q1_0_g128); default is Q4_0.
// ---------------------------------------------------------------------------
#ifdef PICO_LLM_Q1
#define QUANT_WEIGHTS_PER_BLOCK 128
#define QUANT_NAME              "Q1_0_g128"
#define matmul_quant_tile       matmul_q1_0_g128_tile
#else
#define QUANT_WEIGHTS_PER_BLOCK 32
#define QUANT_NAME              "Q4_0"
#define matmul_quant_tile       matmul_q4_0_tile
#endif

// Row size in bytes: (cols / weights_per_block) blocks x 18 bytes per block.
#define QUANT_ROW_BYTES(cols) (((cols) / QUANT_WEIGHTS_PER_BLOCK) * 18)

#ifdef __cplusplus
}
#endif

#endif
