#include "quantize.h"
#include <math.h>
#include <string.h>

#ifdef __ARM_FEATURE_DSP
#include <arm_acle.h>

// SXTAB16 with built-in ROR #8. GCC doesn't fold __sxtab16(__ror(x, 8))
// into a single instruction, so we use inline asm.
static inline __attribute__((always_inline))
uint32_t sxtab16_ror8(uint32_t acc, uint32_t val) {
    uint32_t r;
    __asm__ ("sxtab16 %0, %1, %2, ror #8" : "=r"(r) : "r"(acc), "r"(val));
    return r;
}
#endif

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

// ============================================================================
// Nibble → byte-mask LUTs for Q1_0_g128 kernel
// ============================================================================

// nib2mask: nibble → 4 bytes of 0xFF (bit set) or 0x00 (bit clear).
static const uint32_t nib2mask[16] = {
    0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF,
    0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF,
    0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF,
    0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF,
};

#ifdef __ARM_FEATURE_DSP
// nib2ge: nibble → 4 bytes for __sadd8 GE-flag setting.
// Byte = 0x01 (non-negative → GE=1) where sign bit is set,
// byte = 0x80 (negative → GE=0) where sign bit is clear.
// Little-endian: nibble bit 0 → byte 0 (LSB).
static const uint32_t nib2ge[16] = {
    0x80808080, 0x80808001, 0x80800180, 0x80800101,
    0x80018080, 0x80018001, 0x80010180, 0x80010101,
    0x01808080, 0x01808001, 0x01800180, 0x01800101,
    0x01018080, 0x01018001, 0x01010180, 0x01010101,
};
#endif

// Q1_0_g128 tiled matmul: each weight row is (cols/128) blocks of 18 bytes.
// Block layout: [float16 scale (2 bytes)] [16 sign bytes (128 values)]
//   bit k of qs[j] selects the weight at position j*8 + k within the block:
//     bit = 1 → +scale,  bit = 0 → -scale
//
// Uses ARM DSP intrinsics (SADD8/SEL/SXTB16/SXTAB16/SADD16) for packed
// int16 accumulation. Split into two phases per block: a tight total-sum
// loop (no sign-byte dependency), then a 2×-unrolled pos-sum loop with
// GE-flag-based byte selection. Requires -O3 -funroll-loops for best results.
void matmul_q1_0_g128_tile(float *acc, const uint8_t *weights,
                           const int8_t *x_q, int tile_rows, int cols) {
#ifdef __ARM_FEATURE_DSP
    int blocks_per_row = cols / 128;
    int row_bytes = blocks_per_row * 18;

    for (int i = 0; i < tile_rows; i++) {
        float row_sum = 0.0f;
        const uint8_t *row = weights + i * row_bytes;

        for (int b = 0; b < blocks_per_row; b++) {
            float d = fp16_to_fp32((uint16_t)row[0] | ((uint16_t)row[1] << 8));
            const uint8_t *qs = row + 2;
            const int8_t *xb = x_q + b * 128;

            // Phase 1: sum all 128 activations (no sign-byte dependency)
            uint32_t total16 = 0;
            const uint32_t *xw_all = (const uint32_t *)xb;
            for (int j = 0; j < 32; j++) {
                uint32_t w = xw_all[j];
                uint32_t p = __sxtb16(w);
                total16 = sxtab16_ror8(total16, w);
                total16 = __sadd16(total16, p);
            }
            int32_t total = (int16_t)(total16) + (int16_t)(total16 >> 16);

            // Phase 2: sum activated bytes (2 sign bytes per iteration)
            uint32_t pos16 = 0;
            for (int j = 0; j < 16; j += 2) {
                uint8_t m0 = qs[j];
                uint8_t m1 = qs[j + 1];
                const uint32_t *xw = (const uint32_t *)(xb + j * 8);
                uint32_t w0 = xw[0], w1 = xw[1], w2 = xw[2], w3 = xw[3];

                __sadd8(nib2ge[m0 & 0x0F], 0);
                uint32_t sel0 = __sel(w0, 0);
                __sadd8(nib2ge[m0 >> 4], 0);
                uint32_t sel1 = __sel(w1, 0);
                __sadd8(nib2ge[m1 & 0x0F], 0);
                uint32_t sel2 = __sel(w2, 0);
                __sadd8(nib2ge[m1 >> 4], 0);
                uint32_t sel3 = __sel(w3, 0);

                uint32_t sp0 = __sxtb16(sel0);
                pos16 = sxtab16_ror8(pos16, sel0);
                pos16 = __sadd16(pos16, sp0);
                uint32_t sp1 = __sxtb16(sel1);
                pos16 = sxtab16_ror8(pos16, sel1);
                pos16 = __sadd16(pos16, sp1);
                uint32_t sp2 = __sxtb16(sel2);
                pos16 = sxtab16_ror8(pos16, sel2);
                pos16 = __sadd16(pos16, sp2);
                uint32_t sp3 = __sxtb16(sel3);
                pos16 = sxtab16_ror8(pos16, sel3);
                pos16 = __sadd16(pos16, sp3);
            }
            int32_t pos = (int16_t)(pos16) + (int16_t)(pos16 >> 16);
            int32_t block_sum = 2 * pos - total;

            row_sum += (float)block_sum * d;
            row += 18;
        }
        acc[i] = row_sum;
    }
#else
    // Scalar fallback for non-DSP targets
    int blocks_per_row = cols / 128;
    int row_bytes = blocks_per_row * 18;
    for (int i = 0; i < tile_rows; i++) {
        float row_sum = 0.0f;
        const uint8_t *row = weights + i * row_bytes;
        for (int b = 0; b < blocks_per_row; b++) {
            float d = fp16_to_fp32((uint16_t)row[0] | ((uint16_t)row[1] << 8));
            const uint8_t *qs = row + 2;
            const int8_t *xb = x_q + b * 128;
            int32_t total = 0, pos = 0;
            for (int j = 0; j < 16; j++) {
                uint8_t m = qs[j];
                const int8_t *x8 = xb + j * 8;
                int32_t s0=x8[0], s1=x8[1], s2=x8[2], s3=x8[3];
                int32_t s4=x8[4], s5=x8[5], s6=x8[6], s7=x8[7];
                total += s0+s1+s2+s3+s4+s5+s6+s7;
                if (m&0x01) pos+=s0; if (m&0x02) pos+=s1;
                if (m&0x04) pos+=s2; if (m&0x08) pos+=s3;
                if (m&0x10) pos+=s4; if (m&0x20) pos+=s5;
                if (m&0x40) pos+=s6; if (m&0x80) pos+=s7;
            }
            row_sum += (float)(2*pos - total) * d;
            row += 18;
        }
        acc[i] = row_sum;
    }
#endif
}

// (LUTs defined above, before matmul_q1_0_g128_tile)

// ============================================================================
// Variant A: Branchless LUT + word loads (pure C, no intrinsics)
// ============================================================================

void matmul_q1_0_g128_tile_lut(float *acc, const uint8_t *weights,
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

            int32_t total = 0;
            int32_t pos = 0;
            for (int j = 0; j < 16; j++) {
                uint8_t m = qs[j];
                const uint32_t *xw = (const uint32_t *)(xb + j * 8);
                uint32_t w0 = xw[0];
                uint32_t w1 = xw[1];

                total += (int8_t)(w0) + (int8_t)(w0 >> 8) +
                         (int8_t)(w0 >> 16) + (int8_t)(w0 >> 24) +
                         (int8_t)(w1) + (int8_t)(w1 >> 8) +
                         (int8_t)(w1 >> 16) + (int8_t)(w1 >> 24);

                uint32_t sel0 = w0 & nib2mask[m & 0x0F];
                uint32_t sel1 = w1 & nib2mask[m >> 4];

                pos += (int8_t)(sel0) + (int8_t)(sel0 >> 8) +
                       (int8_t)(sel0 >> 16) + (int8_t)(sel0 >> 24) +
                       (int8_t)(sel1) + (int8_t)(sel1 >> 8) +
                       (int8_t)(sel1 >> 16) + (int8_t)(sel1 >> 24);
            }
            int32_t block_sum = 2 * pos - total;
            row_sum += (float)block_sum * d;
            row += 18;
        }
        acc[i] = row_sum;
    }
}

// ============================================================================
// Variant B: DSP intrinsics (SADD8 + SEL + SXTB16), same loop structure
// ============================================================================

#ifdef __ARM_FEATURE_DSP

// Horizontal sum of 4 packed int8 bytes in a uint32_t → int32_t scalar.
// Uses SXTB16 to extract bytes 0,2 as int16 pair, then SXTAB16 with ROR #8
// to add bytes 1,3 into the pair, then extracts and sums the two int16 halves.
static inline int32_t hsum_packed_i8(uint32_t w) {
    uint32_t pair = __sxtb16(w);
    uint32_t sum_pair = sxtab16_ror8(pair, w);
    return (int16_t)(sum_pair) + (int16_t)(sum_pair >> 16);
}

void matmul_q1_0_g128_tile_dsp(float *acc, const uint8_t *weights,
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

            uint32_t total16 = 0;
            uint32_t pos16 = 0;

            for (int j = 0; j < 16; j++) {
                uint8_t m = qs[j];
                const uint32_t *xw = (const uint32_t *)(xb + j * 8);
                uint32_t w0 = xw[0];
                uint32_t w1 = xw[1];

                // Accumulate total into packed int16 pair
                uint32_t p0 = __sxtb16(w0);
                total16 = sxtab16_ror8(total16, w0);
                total16 = __sadd16(total16, p0);

                uint32_t p1 = __sxtb16(w1);
                total16 = sxtab16_ror8(total16, w1);
                total16 = __sadd16(total16, p1);

                // Select activated bytes via GE flags, accumulate pos
                __sadd8(nib2ge[m & 0x0F], 0);
                uint32_t sel0 = __sel(w0, 0);

                __sadd8(nib2ge[m >> 4], 0);
                uint32_t sel1 = __sel(w1, 0);

                uint32_t sp0 = __sxtb16(sel0);
                pos16 = sxtab16_ror8(pos16, sel0);
                pos16 = __sadd16(pos16, sp0);

                uint32_t sp1 = __sxtb16(sel1);
                pos16 = sxtab16_ror8(pos16, sel1);
                pos16 = __sadd16(pos16, sp1);
            }

            int32_t total = (int16_t)(total16) + (int16_t)(total16 >> 16);
            int32_t pos = (int16_t)(pos16) + (int16_t)(pos16 >> 16);
            int32_t block_sum = 2 * pos - total;

            row_sum += (float)block_sum * d;
            row += 18;
        }
        acc[i] = row_sum;
    }
}

// ============================================================================
// Variant C: Split loops + DSP + 2× unroll
// ============================================================================

void matmul_q1_0_g128_tile_dsp2(float *acc, const uint8_t *weights,
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

            // Phase 1: compute total (sum of all 128 activations)
            // No sign-byte dependency — tight loop of word loads + packed add
            uint32_t total16 = 0;
            const uint32_t *xw_all = (const uint32_t *)xb;
            for (int j = 0; j < 32; j++) {
                uint32_t w = xw_all[j];
                uint32_t p = __sxtb16(w);
                total16 = sxtab16_ror8(total16, w);
                total16 = __sadd16(total16, p);
            }
            int32_t total = (int16_t)(total16) + (int16_t)(total16 >> 16);

            // Phase 2: compute pos (2 sign bytes / 16 activations per iteration)
            uint32_t pos16 = 0;
            for (int j = 0; j < 16; j += 2) {
                uint8_t m0 = qs[j];
                uint8_t m1 = qs[j + 1];

                const uint32_t *xw = (const uint32_t *)(xb + j * 8);
                uint32_t w0 = xw[0], w1 = xw[1], w2 = xw[2], w3 = xw[3];

                __sadd8(nib2ge[m0 & 0x0F], 0);
                uint32_t sel0 = __sel(w0, 0);
                __sadd8(nib2ge[m0 >> 4], 0);
                uint32_t sel1 = __sel(w1, 0);
                __sadd8(nib2ge[m1 & 0x0F], 0);
                uint32_t sel2 = __sel(w2, 0);
                __sadd8(nib2ge[m1 >> 4], 0);
                uint32_t sel3 = __sel(w3, 0);

                // Accumulate selected bytes into packed int16
                uint32_t sp0 = __sxtb16(sel0);
                pos16 = sxtab16_ror8(pos16, sel0);
                pos16 = __sadd16(pos16, sp0);

                uint32_t sp1 = __sxtb16(sel1);
                pos16 = sxtab16_ror8(pos16, sel1);
                pos16 = __sadd16(pos16, sp1);

                uint32_t sp2 = __sxtb16(sel2);
                pos16 = sxtab16_ror8(pos16, sel2);
                pos16 = __sadd16(pos16, sp2);

                uint32_t sp3 = __sxtb16(sel3);
                pos16 = sxtab16_ror8(pos16, sel3);
                pos16 = __sadd16(pos16, sp3);
            }
            int32_t pos = (int16_t)(pos16) + (int16_t)(pos16 >> 16);
            int32_t block_sum = 2 * pos - total;

            row_sum += (float)block_sum * d;
            row += 18;
        }
        acc[i] = row_sum;
    }
}

#else
// Fallback stubs when DSP is not available (host builds, etc.)
void matmul_q1_0_g128_tile_dsp(float *acc, const uint8_t *weights,
                                const int8_t *x_q, int tile_rows, int cols) {
    matmul_q1_0_g128_tile(acc, weights, x_q, tile_rows, cols);
}
void matmul_q1_0_g128_tile_dsp2(float *acc, const uint8_t *weights,
                                 const int8_t *x_q, int tile_rows, int cols) {
    matmul_q1_0_g128_tile(acc, weights, x_q, tile_rows, cols);
}
#endif
