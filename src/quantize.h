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

// Convert IEEE 754 float16 (half-precision) to float32
float fp16_to_fp32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif
