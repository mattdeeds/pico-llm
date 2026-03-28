#include <stdio.h>
#include <math.h>
#include <string.h>
#include "pico/stdlib.h"
#include "quantize.h"

static int tests_passed = 0;
static int tests_failed = 0;

static void assert_near(float actual, float expected, float tol, const char *name) {
    if (fabsf(actual - expected) <= tol) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("FAIL %s: expected %.6f, got %.6f (diff %.6f)\n",
               name, expected, actual, fabsf(actual - expected));
    }
}

// Test: 4x4 matmul with known values
static void test_matmul_4x4(void) {
    printf("test_matmul_4x4...\n");

    // weights (4x4 int8) = identity-ish matrix scaled
    int8_t weights[16] = {
        10,  0,  0,  0,
         0, 10,  0,  0,
         0,  0, 10,  0,
         0,  0,  0, 10,
    };
    float w_scale = 0.1f; // so effective weights are identity

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];

    matmul_q8(out, weights, w_scale, x, 4, 4);

    // With identity weights, output should approximate input
    // Quantization adds some noise
    assert_near(out[0], 1.0f, 0.1f, "4x4[0]");
    assert_near(out[1], 2.0f, 0.1f, "4x4[1]");
    assert_near(out[2], 3.0f, 0.1f, "4x4[2]");
    assert_near(out[3], 4.0f, 0.1f, "4x4[3]");
}

// Test: quantize and dequantize roundtrip
static void test_quantize_roundtrip(void) {
    printf("test_quantize_roundtrip...\n");

    float input[8] = {0.5f, -1.0f, 0.25f, 0.0f, 1.0f, -0.5f, 0.75f, -0.75f};
    int8_t quantized[8];
    float scale = quantize_vec(quantized, input, 8);

    // Dequantize and check
    for (int i = 0; i < 8; i++) {
        float recovered = (float)quantized[i] * scale;
        assert_near(recovered, input[i], 0.02f, "roundtrip");
    }
}

// Test: matmul accumulation correctness
static void test_matmul_accumulation(void) {
    printf("test_matmul_accumulation...\n");

    // 2x4 weight matrix: row 0 = [1,1,1,1], row 1 = [1,-1,1,-1]
    int8_t weights[8] = {
        1, 1, 1, 1,
        1, -1, 1, -1,
    };
    float w_scale = 1.0f;

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2];

    matmul_q8(out, weights, w_scale, x, 2, 4);

    // row 0: 1+2+3+4 = 10, row 1: 1-2+3-4 = -2
    // But x gets quantized: scale = 4/127 ≈ 0.0315
    // x_q ≈ [32, 63, 95, 127]
    // row 0 acc = 32+63+95+127 = 317, out = 317 * 1.0 * (4/127) ≈ 9.98
    // row 1 acc = 32-63+95-127 = -63, out = -63 * 1.0 * (4/127) ≈ -1.98
    assert_near(out[0], 10.0f, 0.5f, "accum[0]");
    assert_near(out[1], -2.0f, 0.5f, "accum[1]");
}

int main(void) {
    stdio_init_all();

    printf("\n=== pico-llm matmul tests ===\n\n");

    test_quantize_roundtrip();
    test_matmul_4x4();
    test_matmul_accumulation();

    printf("\n--- Results: %d passed, %d failed ---\n",
           tests_passed, tests_failed);

    while (1) {
        tight_loop_contents();
    }

    return 0;
}
