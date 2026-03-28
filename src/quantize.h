#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stdint.h>

// Quantized matmul: out[rows] = (weights[rows][cols] @ x[cols]) * scale
// Weights are int8, input is float (quantized internally), output is float
void matmul_q8(float *out, const int8_t *weights, float w_scale,
               const float *x, int rows, int cols);

// Tiled matmul: process a tile of rows at a time from a weight buffer
// out[tile_rows] += weights_tile[tile_rows][cols] @ x_q[cols]
// Pure integer inner loop for SMLAL optimization
void matmul_q8_tile(int32_t *acc, const int8_t *weights, const int8_t *x_q,
                    int tile_rows, int cols);

// Quantize a float vector to int8, returns the scale factor used
float quantize_vec(int8_t *out, const float *in, int size);

// Dequantize int32 accumulators to float with combined scale
void dequant_acc(float *out, const int32_t *acc, float w_scale, float x_scale,
                 int size);

#endif
