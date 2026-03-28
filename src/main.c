#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "transformer.h"
#include "sdcard.h"
#include "tokenizer.h"

// Static allocation for weight double-buffers
static int8_t weight_buf_a[WEIGHT_BUF_SIZE];
static int8_t weight_buf_b[WEIGHT_BUF_SIZE];

// Static allocation for RunState buffers
static float x_buf[D_MODEL];
static float xb_buf[D_MODEL];
static float xb2_buf[D_MODEL];
static float q_buf[D_MODEL];
static float k_buf[D_MODEL]; // sized for max kv_dim
static float v_buf[D_MODEL];
static float att_buf[N_HEADS];
static float hb_buf[HIDDEN_DIM];
static float hb2_buf[HIDDEN_DIM];
static float logits_buf[VOCAB_SIZE];

static void init_run_state(RunState *s) {
    s->x = x_buf;
    s->xb = xb_buf;
    s->xb2 = xb2_buf;
    s->q = q_buf;
    s->k = k_buf;
    s->v = v_buf;
    s->att = att_buf;
    s->hb = hb_buf;
    s->hb2 = hb2_buf;
    s->logits = logits_buf;
    s->kv_cache_len = 0;
}

int main(void) {
    stdio_init_all();

    printf("\n");
    printf("=========================\n");
    printf("  pico-llm v0.1\n");
    printf("  Bare metal LLM on RP2350\n");
    printf("=========================\n\n");

    // Initialize transformer context
    TransformerContext ctx;
    memset(&ctx, 0, sizeof(ctx));

    ctx.config.dim = D_MODEL;
    ctx.config.hidden_dim = HIDDEN_DIM;
    ctx.config.n_layers = N_LAYERS;
    ctx.config.n_heads = N_HEADS;
    ctx.config.n_kv_heads = N_KV_HEADS;
    ctx.config.vocab_size = VOCAB_SIZE;
    ctx.config.max_seq_len = MAX_SEQ_LEN;

    ctx.weight_buf_a = weight_buf_a;
    ctx.weight_buf_b = weight_buf_b;
    ctx.active_buf = weight_buf_a;

    init_run_state(&ctx.state);

    printf("Model config:\n");
    printf("  dim:         %d\n", ctx.config.dim);
    printf("  hidden_dim:  %d\n", ctx.config.hidden_dim);
    printf("  layers:      %d\n", ctx.config.n_layers);
    printf("  heads:       %d\n", ctx.config.n_heads);
    printf("  kv_heads:    %d\n", ctx.config.n_kv_heads);
    printf("  vocab:       %d\n", ctx.config.vocab_size);
    printf("  max_seq_len: %d\n", ctx.config.max_seq_len);
    printf("\n");

    // Double-buffer state
    static WeightBuf wb;

    // Initialize SD card
    printf("Initializing SD card...\n");
    if (!sdcard_init()) {
        printf("SD card init failed. Skipping weight load.\n");
    } else {
        printf("SD card OK. Starting weight prefetch on Core 1.\n");
        // TODO: Read config header from SD
        // TODO: Load vocab / tokenizer from SD
        // Launch Core 1 for weight prefetch
        weightbuf_init(&wb, weight_buf_a, weight_buf_b, WEIGHT_BUF_SIZE);
        multicore_launch_core1(prefetch_worker);
    }

    printf("\nRAM usage estimate:\n");
    printf("  Weight buffers: %d bytes (x2)\n", WEIGHT_BUF_SIZE);
    printf("  Activations:    %lu bytes\n",
           (unsigned long)(sizeof(x_buf) + sizeof(xb_buf) + sizeof(xb2_buf)));
    printf("  Attention:      %lu bytes\n",
           (unsigned long)(sizeof(q_buf) + sizeof(k_buf) + sizeof(v_buf) + sizeof(att_buf)));
    printf("  FFN:            %lu bytes\n",
           (unsigned long)(sizeof(hb_buf) + sizeof(hb2_buf)));
    printf("  Logits:         %lu bytes\n", (unsigned long)sizeof(logits_buf));
    printf("\n");

    printf("pico-llm ready. Waiting for SD card implementation.\n");

    while (1) {
        tight_loop_contents();
    }

    return 0;
}
