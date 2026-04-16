#include "quantize.h"
#include <math.h>
#include <string.h>

float quantize_vec(int8_t *out, const float *in, int size) {
    // Find absmax for symmetric quantization
    float absmax = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(in[i]);
        if (a > absmax) absmax = a;
    }

    float scale = absmax / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    float inv_scale = 127.0f / absmax;
    for (int i = 0; i < size; i++) {
        int v = (int)(in[i] * inv_scale + 0.5f);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        out[i] = (int8_t)v;
    }

    return scale;
}

// IEEE 754 float16 → float32 conversion
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        // Zero or subnormal
        if (mant == 0) {
            float result;
            memcpy(&result, &sign, sizeof(float));
            return result;
        }
        // Subnormal: normalize
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t f = sign | 0x7F800000 | (mant << 13);
        float result;
        memcpy(&result, &f, sizeof(float));
        return result;
    }

    // Normal: rebias exponent from fp16 bias (15) to fp32 bias (127)
    uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Q1_0_g128 tiled matmul: each weight row is (cols/128) blocks of 18 bytes.
// Block layout: [float16 scale (2 bytes)] [16 sign bytes (128 values)]
//   bit k of qs[j] selects the weight at position j*8 + k within the block:
//     bit = 1 → +scale,  bit = 0 → -scale
//
// Inner loop: sign × int8 activation → int32 per-block accumulator
// (using 2*pos_sum - total to avoid per-bit branches on the hot path),
// then multiply by fp16 block scale. Output is float per row.
void matmul_q1_0_g128_tile(float *acc, const uint8_t *weights,
                           const int8_t *x_q, int tile_rows, int cols) {
    int blocks_per_row = cols / 128;
    int row_bytes = blocks_per_row * 18;

    for (int i = 0; i < tile_rows; i++) {
        float row_sum = 0.0f;
        const uint8_t *row = weights + i * row_bytes;

        for (int b = 0; b < blocks_per_row; b++) {
            float d = fp16_to_fp32((uint16_t)row[0] | ((uint16_t)row[1] << 8));
            const uint8_t *qs = row + 2;
            const int8_t *xb = x_q + b * 128;

            // block_sum = sum over j,k of sign_{j,k} * xb[j*8 + k]
            //           = 2*sum(xb[i] where bit_i=1) - sum(xb[i])
            int32_t total = 0;
            int32_t pos = 0;
            for (int j = 0; j < 16; j++) {
                uint8_t m = qs[j];
                const int8_t *x8 = xb + j * 8;
                int32_t s0 = x8[0], s1 = x8[1], s2 = x8[2], s3 = x8[3];
                int32_t s4 = x8[4], s5 = x8[5], s6 = x8[6], s7 = x8[7];
                total += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
                if (m & 0x01) pos += s0;
                if (m & 0x02) pos += s1;
                if (m & 0x04) pos += s2;
                if (m & 0x08) pos += s3;
                if (m & 0x10) pos += s4;
                if (m & 0x20) pos += s5;
                if (m & 0x40) pos += s6;
                if (m & 0x80) pos += s7;
            }
            int32_t block_sum = 2 * pos - total;

            row_sum += (float)block_sum * d;
            row += 18;
        }

        acc[i] = row_sum;
    }
}
