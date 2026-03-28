#include "quantize.h"
#include <math.h>

float quantize_vec(int8_t *out, const float *in, int size) {
    // Find absmax for symmetric quantization
    float absmax = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(in[i]);
        if (a > absmax) absmax = a;
    }

    float scale = absmax / 127.0f;
    if (scale == 0.0f) scale = 1.0f; // avoid div by zero for zero vectors

    float inv_scale = 127.0f / absmax;
    for (int i = 0; i < size; i++) {
        int v = (int)(in[i] * inv_scale + 0.5f);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }

    return scale;
}

void matmul_q8_tile(int32_t *acc, const int8_t *weights, const int8_t *x_q,
                    int tile_rows, int cols) {
    // Core int8 matmul: acc[i] = sum_j(weights[i*cols+j] * x_q[j])
    // On Cortex-M33 this inner loop maps well to SMLAL instructions.
    for (int i = 0; i < tile_rows; i++) {
        int32_t sum = 0;
        const int8_t *w_row = weights + i * cols;
        for (int j = 0; j < cols; j++) {
            sum += (int32_t)w_row[j] * (int32_t)x_q[j];
        }
        acc[i] = sum;
    }
}

void dequant_acc(float *out, const int32_t *acc, float w_scale, float x_scale,
                 int size) {
    float combined = w_scale * x_scale;
    for (int i = 0; i < size; i++) {
        out[i] = (float)acc[i] * combined;
    }
}

void matmul_q8(float *out, const int8_t *weights, float w_scale,
               const float *x, int rows, int cols) {
    // Quantize input vector
    int8_t x_q[cols]; // VLA, cols <= d_model (256) = 256 bytes on stack
    float x_scale = quantize_vec(x_q, x, cols);

    // Tiled matmul with integer accumulation
    int32_t acc[rows]; // VLA, rows typically <= tile size
    matmul_q8_tile(acc, weights, x_q, rows, cols);

    // Dequantize back to float
    dequant_acc(out, acc, w_scale, x_scale, rows);
}
