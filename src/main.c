#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "pico/multicore.h"
#include "hardware/dma.h"
#include "transformer.h"
#include "sdcard.h"
#include "tokenizer.h"

// Static allocation for weight double-buffers (4-byte aligned for DMA)
static int8_t weight_buf_a[WEIGHT_BUF_SIZE] __attribute__((aligned(4)));
static int8_t weight_buf_b[WEIGHT_BUF_SIZE] __attribute__((aligned(4)));

// Static allocation for RunState buffers
static float x_buf[D_MODEL];
static float xb_buf[D_MODEL];
static float xb2_buf[D_MODEL];
static float q_buf[D_MODEL];
static float k_buf[D_MODEL];
static float v_buf[D_MODEL];
static float att_buf[N_HEADS];
static float hb_buf[HIDDEN_DIM];
static float hb2_buf[HIDDEN_DIM];
static float logits_buf[VOCAB_SIZE];

// Final layer norm stays in RAM (dim * 4 bytes = 1024 bytes for dim=256)
static float final_norm_buf[D_MODEL];

// Scratch buffer for SD reads during init (reuses weight_buf_a before prefetch starts)
#define SCRATCH_BUF ((uint8_t *)weight_buf_a)
#define SCRATCH_SIZE WEIGHT_BUF_SIZE

// Token embedding read buffer: enough for one embedding + alignment slop
// dim*4 bytes, spanning at most ceil(dim*4/512)+1 blocks
#define EMB_BLOCKS (((D_MODEL * 4 + 511) / 512) + 1)
static uint8_t emb_read_buf[EMB_BLOCKS * 512] __attribute__((aligned(4)));

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

// ============================================================================
// Model loading from SD card
// ============================================================================

// Read arbitrary bytes from SD at a non-block-aligned offset.
// tmp must be 4-byte aligned and large enough: ceil((byte_off%512 + nbytes) / 512) blocks.
static bool sd_read_bytes(uint32_t byte_off, void *out, uint32_t nbytes,
                          uint8_t *tmp, uint32_t tmp_size) {
    uint32_t block = byte_off / 512;
    uint32_t block_off = byte_off % 512;
    uint32_t blocks_needed = (block_off + nbytes + 511) / 512;
    if (blocks_needed * 512 > tmp_size)
        return false;
    if (!sdcard_read_blocks(block, tmp, blocks_needed))
        return false;
    memcpy(out, tmp + block_off, nbytes);
    return true;
}

// Read the 7 x int32 config header from block 0 of the SD card.
// Returns false if the read fails or the config doesn't match compile-time values.
static bool load_config(Config *cfg) {
    Config sd;
    if (!sd_read_bytes(0, &sd, sizeof(Config), SCRATCH_BUF, SCRATCH_SIZE))
        return false;

    if (sd.dim != D_MODEL || sd.hidden_dim != HIDDEN_DIM ||
        sd.n_layers != N_LAYERS || sd.n_heads != N_HEADS ||
        sd.n_kv_heads != N_KV_HEADS || sd.vocab_size != VOCAB_SIZE ||
        sd.max_seq_len != MAX_SEQ_LEN) {
        printf("ERROR: SD model config does not match firmware\n");
        printf("  SD:  dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
               sd.dim, sd.hidden_dim, sd.n_layers, sd.n_heads,
               sd.n_kv_heads, sd.vocab_size, sd.max_seq_len);
        printf("  FW:  dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
               D_MODEL, HIDDEN_DIM, N_LAYERS, N_HEADS,
               N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN);
        return false;
    }

    *cfg = sd;
    return true;
}

// Skip over the vocab section (length-prefixed strings) and return the byte
// offset where it ends, or 0 on read failure. Uses SCRATCH_BUF for temporary reads.
static uint32_t skip_vocab(int vocab_size) {
    uint32_t pos = sizeof(Config); // vocab starts right after the 28-byte header
    int remaining = vocab_size;

    while (remaining > 0) {
        uint32_t block = pos / 512;
        uint32_t block_off = pos % 512;
        uint32_t blocks = SCRATCH_SIZE / 512;
        if (!sdcard_read_blocks(block, SCRATCH_BUF, blocks))
            return 0;

        uint32_t buf_bytes = blocks * 512;
        uint32_t local = block_off;

        while (remaining > 0 && local + 2 <= buf_bytes) {
            uint16_t len = SCRATCH_BUF[local] | (SCRATCH_BUF[local + 1] << 8);
            if (local + 2 + len > buf_bytes)
                break; // entry crosses buffer boundary, re-read from new position
            local += 2 + len;
            remaining--;
        }

        pos = block * 512 + local;
    }

    return pos;
}

// Compute all byte offsets into the model file.
static void compute_layout(ModelLayout *layout, const Config *cfg) {
    int dim = cfg->dim;
    int kv_dim = (dim / cfg->n_heads) * cfg->n_kv_heads;
    int hidden = cfg->hidden_dim;

    uint32_t layer = 0;
    layer += dim * 4;                   // attn_norm
    layer += dim * dim + 4;             // wq
    layer += kv_dim * dim + 4;          // wk
    layer += kv_dim * dim + 4;          // wv
    layer += dim * dim + 4;             // wo
    layer += dim * 4;                   // ffn_norm
    layer += hidden * dim + 4;          // w_gate
    layer += hidden * dim + 4;          // w_up
    layer += dim * hidden + 4;          // w_down
    layout->layer_bytes = layer;

    uint32_t after_layers = layout->vocab_end + (uint32_t)cfg->n_layers * layer;
    layout->final_norm_off = after_layers;
    layout->wcls_off = layout->final_norm_off + dim * 4;
    layout->emb_off = layout->wcls_off + cfg->vocab_size * dim + 4;
}

bool load_token_embedding(const TransformerContext *ctx, int token_id, float *out) {
    uint32_t byte_off = ctx->layout.emb_off
                        + (uint32_t)token_id * D_MODEL * sizeof(float);
    return sd_read_bytes(byte_off, out, D_MODEL * sizeof(float),
                         emb_read_buf, sizeof(emb_read_buf));
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    stdio_init_all();

    // Wait for USB CDC to enumerate so we don't lose boot output
    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }
    sleep_ms(200);

    printf("\n");
    printf("=========================\n");
    printf("  pico-llm v0.1\n");
    printf("  Bare metal LLM on RP2350\n");
    printf("=========================\n\n");

    // --- Initialize SD card ---
    printf("Initializing SD card...\n");
    if (!sdcard_init()) {
        printf("FATAL: SD card init failed.\n");
        goto halt;
    }
    printf("SD card OK.\n\n");

    // --- Load and validate model config ---
    TransformerContext ctx;
    memset(&ctx, 0, sizeof(ctx));

    printf("Loading model header...\n");
    if (!load_config(&ctx.config)) {
        printf("FATAL: Could not load model config from SD.\n");
        goto halt;
    }

    printf("Model config:\n");
    printf("  dim:         %d\n", ctx.config.dim);
    printf("  hidden_dim:  %d\n", ctx.config.hidden_dim);
    printf("  layers:      %d\n", ctx.config.n_layers);
    printf("  heads:       %d\n", ctx.config.n_heads);
    printf("  kv_heads:    %d\n", ctx.config.n_kv_heads);
    printf("  vocab:       %d\n", ctx.config.vocab_size);
    printf("  max_seq_len: %d\n", ctx.config.max_seq_len);
    printf("\n");

    // --- Scan past vocab section to find weight offsets ---
    printf("Scanning vocab (%d tokens)...\n", ctx.config.vocab_size);
    ctx.layout.vocab_end = skip_vocab(ctx.config.vocab_size);
    if (ctx.layout.vocab_end == 0) {
        printf("FATAL: SD read failed while scanning vocab.\n");
        goto halt;
    }
    compute_layout(&ctx.layout, &ctx.config);

    // KV cache lives on SD after all model data, block-aligned
    uint32_t model_end = ctx.layout.emb_off
                       + (uint32_t)VOCAB_SIZE * D_MODEL * sizeof(float);
    ctx.layout.kv_base_block = (model_end + 511) / 512;

    printf("Model layout (byte offsets on SD):\n");
    printf("  weights start:  %lu\n", (unsigned long)ctx.layout.vocab_end);
    printf("  layer size:     %lu bytes\n", (unsigned long)ctx.layout.layer_bytes);
    printf("  final_norm:     %lu\n", (unsigned long)ctx.layout.final_norm_off);
    printf("  classifier:     %lu\n", (unsigned long)ctx.layout.wcls_off);
    printf("  embeddings:     %lu\n", (unsigned long)ctx.layout.emb_off);
    printf("  kv_cache:       block %lu\n", (unsigned long)ctx.layout.kv_base_block);
    printf("\n");

    // --- Load final norm into RAM (stays resident) ---
    printf("Loading final_norm...\n");
    ctx.final_norm = final_norm_buf;
    if (!sd_read_bytes(ctx.layout.final_norm_off, final_norm_buf,
                       D_MODEL * sizeof(float), SCRATCH_BUF, SCRATCH_SIZE)) {
        printf("FATAL: Could not load final_norm from SD.\n");
        goto halt;
    }

    // --- Initialize run state ---
    init_run_state(&ctx.state);

    // --- Initialize weight buffers (single-core synchronous mode) ---
    // TODO: Core 1 SDIO prefetch causes CMD18 hang — PIO commands from Core 1
    // don't return. Future options: single-core async DMA, or Core 1 for compute.
    static WeightBuf wb;
    weightbuf_init(&wb, weight_buf_a, weight_buf_b, WEIGHT_BUF_SIZE);
    ctx.wb = &wb;
    printf("Weight streaming ready (single-core mode).\n\n");

    // --- Print RAM usage ---
    printf("RAM usage:\n");
    printf("  Weight buffers: %d bytes (x2)\n", WEIGHT_BUF_SIZE);
    printf("  Activations:    %lu bytes\n",
           (unsigned long)(sizeof(x_buf) + sizeof(xb_buf) + sizeof(xb2_buf)));
    printf("  Attention:      %lu bytes\n",
           (unsigned long)(sizeof(q_buf) + sizeof(k_buf) + sizeof(v_buf) + sizeof(att_buf)));
    printf("  FFN:            %lu bytes\n",
           (unsigned long)(sizeof(hb_buf) + sizeof(hb2_buf)));
    printf("  Logits:         %lu bytes\n", (unsigned long)sizeof(logits_buf));
    printf("  Final norm:     %lu bytes\n", (unsigned long)sizeof(final_norm_buf));
    printf("  Emb read buf:   %lu bytes\n", (unsigned long)sizeof(emb_read_buf));
    printf("\n");

    printf("pico-llm ready. Model loaded from SD.\n\n");

    // --- Generate tokens ---
    // Start with token 0 (BOS) and generate up to 32 tokens
    printf("Generating (BOS=0, max_tokens=32)...\n");
    int prompt[] = {0};
    int n_generated = generate(&ctx, prompt, 1, 32);
    printf("\nGenerated %d tokens.\n", n_generated);

halt:
    while (1) {
        tight_loop_contents();
    }

    return 0;
}
