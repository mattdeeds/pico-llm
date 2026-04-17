#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "transformer.h"
#include "sdcard.h"
#include "quantize.h"
#include "tokenizer.h"

// Static allocation for weight double-buffers (4-byte aligned for DMA)
static int8_t weight_buf_a[WEIGHT_BUF_SIZE] __attribute__((aligned(4)));
static int8_t weight_buf_b[WEIGHT_BUF_SIZE] __attribute__((aligned(4)));

// Static allocation for RunState buffers
static float x_buf[D_MODEL];
static float xb_buf[Q_DIM];      // also used for attention output [Q_DIM]
static float xb2_buf[D_MODEL];
static float q_buf[Q_DIM];
static float k_buf[KV_DIM];
static float v_buf[KV_DIM];
static float att_buf[N_HEADS];
static float hb_buf[HIDDEN_DIM];
static float hb2_buf[HIDDEN_DIM];

// Final layer norm stays in RAM
static float final_norm_buf[D_MODEL];

// Shared matmul accumulator for Core 1 compute (float for per-block dequant)
float matmul_acc[HIDDEN_DIM > Q_DIM ? HIDDEN_DIM : Q_DIM];

// Scratch buffer for SD reads during init (reuses weight_buf_a before prefetch starts)
#define SCRATCH_BUF ((uint8_t *)weight_buf_a)
#define SCRATCH_SIZE WEIGHT_BUF_SIZE

// Token embedding read buffer: reads one Q4_0 row from wcls (tied embeddings)
// Q4_0 row bytes = (D_MODEL / 32) * 18
#define EMB_Q4_BYTES ((D_MODEL / 32) * 18)
#define EMB_BLOCKS (((EMB_Q4_BYTES + 511) / 512) + 1)
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
    s->kv_cache_len = 0;
}

// ============================================================================
// Model loading from SD card
// ============================================================================

// Read arbitrary bytes from SD at a non-block-aligned offset.
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

// Read the V2 config header (8 x int32) from block 0 of the SD card.
static bool load_config(Config *cfg) {
    Config sd;
    if (!sd_read_bytes(0, &sd, sizeof(Config), SCRATCH_BUF, SCRATCH_SIZE))
        return false;

    if (sd.dim != D_MODEL || sd.hidden_dim != HIDDEN_DIM ||
        sd.n_layers != N_LAYERS || sd.n_heads != N_HEADS ||
        sd.n_kv_heads != N_KV_HEADS || sd.vocab_size != VOCAB_SIZE ||
        sd.max_seq_len != MAX_SEQ_LEN || sd.head_dim != HEAD_DIM) {
        printf("ERROR: SD model config does not match firmware\n");
        printf("  SD:  dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d head_dim=%d\n",
               sd.dim, sd.hidden_dim, sd.n_layers, sd.n_heads,
               sd.n_kv_heads, sd.vocab_size, sd.max_seq_len, sd.head_dim);
        printf("  FW:  dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d head_dim=%d\n",
               D_MODEL, HIDDEN_DIM, N_LAYERS, N_HEADS,
               N_KV_HEADS, VOCAB_SIZE, MAX_SEQ_LEN, HEAD_DIM);
        return false;
    }

    *cfg = sd;
    return true;
}

// Scan vocab section to find the end offset (= start of weight data).
static uint32_t scan_vocab_end(int vocab_size) {
    uint32_t pos = sizeof(Config);
    int scanned = 0;

    while (scanned < vocab_size) {
        uint32_t block = pos / 512;
        uint32_t block_off = pos % 512;
        uint32_t blocks = SCRATCH_SIZE / 512;
        if (!sdcard_read_blocks(block, SCRATCH_BUF, blocks))
            return 0;

        uint32_t buf_bytes = blocks * 512;
        uint32_t local = block_off;

        while (scanned < vocab_size && local + 2 <= buf_bytes) {
            uint16_t len = SCRATCH_BUF[local] | (SCRATCH_BUF[local + 1] << 8);
            if (local + 2 + len > buf_bytes)
                break;
            local += 2 + len;
            scanned++;
        }

        pos = block * 512 + local;
    }

    return pos;
}

// Q4_0 row size in bytes: (cols / 32) blocks × 18 bytes per block
static inline uint32_t q4_row_bytes(int cols) {
    return (uint32_t)(cols / 32) * 18;
}

// Compute all byte offsets into the Q4_0 model file.
static void compute_layout(ModelLayout *layout, const Config *cfg) {
    int dim = cfg->dim;
    int head_dim = cfg->head_dim;
    int q_dim = cfg->n_heads * head_dim;
    int kv_dim = cfg->n_kv_heads * head_dim;
    int hidden = cfg->hidden_dim;

    uint32_t layer = 0;
    layer += dim * 4;                                // attn_norm  [dim] float32
    layer += head_dim * 4;                           // q_norm     [head_dim] float32
    layer += head_dim * 4;                           // k_norm     [head_dim] float32
    layer += (uint32_t)q_dim * q4_row_bytes(dim);    // wq   Q4 [q_dim rows]
    layer += (uint32_t)kv_dim * q4_row_bytes(dim);   // wk   Q4 [kv_dim rows]
    layer += (uint32_t)kv_dim * q4_row_bytes(dim);   // wv   Q4 [kv_dim rows]
    layer += (uint32_t)dim * q4_row_bytes(q_dim);    // wo   Q4 [dim rows]
    layer += dim * 4;                                // ffn_norm  [dim] float32
    layer += (uint32_t)hidden * q4_row_bytes(dim);   // w_gate Q4
    layer += (uint32_t)hidden * q4_row_bytes(dim);   // w_up   Q4
    layer += (uint32_t)dim * q4_row_bytes(hidden);   // w_down Q4
    layout->layer_bytes = layer;

    uint32_t after_layers = layout->vocab_end + (uint32_t)cfg->n_layers * layer;
    layout->final_norm_off = after_layers;
    layout->wcls_off = layout->final_norm_off + dim * 4;
    layout->wcls_end = layout->wcls_off
                       + (uint32_t)cfg->vocab_size * q4_row_bytes(dim);
}

// Load a single token embedding from SD.
// Tied embeddings: read one Q4_0 row from wcls, dequantize per-block to float32.
bool load_token_embedding(const TransformerContext *ctx, int token_id, float *out) {
    uint32_t row_bytes = q4_row_bytes(D_MODEL);
    uint32_t byte_off = ctx->layout.wcls_off + (uint32_t)token_id * row_bytes;

    uint8_t q4_row[EMB_Q4_BYTES];
    if (!sd_read_bytes(byte_off, q4_row, row_bytes, emb_read_buf, sizeof(emb_read_buf)))
        return false;

    // Dequantize Q4_0 blocks → float32
    // Per block: 2B fp16 scale + 16 packed nibble bytes → 32 float values.
    // qs[j] low nibble = weight at position j, high nibble = weight at j+16.
    int blocks = D_MODEL / 32;
    const uint8_t *p = q4_row;
    for (int b = 0; b < blocks; b++) {
        float d = fp16_to_fp32((uint16_t)p[0] | ((uint16_t)p[1] << 8));
        const uint8_t *qs = p + 2;
        int col_base = b * 32;
        for (int j = 0; j < 16; j++) {
            uint8_t qb = qs[j];
            out[col_base + j]      = ((int)(qb & 0xF) - 8) * d;
            out[col_base + j + 16] = ((int)(qb >> 4) - 8) * d;
        }
        p += 18;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    // Pin sysclk to 200 MHz so SDIO PIO dividers are exact integers
    // (200 MHz / 4 = 50.00 MHz HS). Must be called before stdio_init_all().
    set_sys_clock_khz(200000, true);

    stdio_init_all();

    while (!stdio_usb_connected()) {
        sleep_ms(100);
    }
    sleep_ms(200);

    printf("\n");
    printf("================================\n");
    printf("  pico-llm v0.5 (Qwen3-0.6B Q4_0)\n");
    printf("  Bare metal LLM on RP2350\n");
    printf("================================\n\n");

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
    ctx.temperature = 0.7f;
    ctx.repetition_penalty = 1.1f;

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
    printf("  head_dim:    %d\n", ctx.config.head_dim);
    printf("  q_dim:       %d\n", ctx.config.n_heads * ctx.config.head_dim);
    printf("  kv_dim:      %d\n", ctx.config.n_kv_heads * ctx.config.head_dim);
    printf("\n");

    // --- Scan vocab section to find weight start offset ---
    printf("Scanning vocab (%d tokens)...\n", ctx.config.vocab_size);
    ctx.layout.vocab_end = scan_vocab_end(ctx.config.vocab_size);
    if (ctx.layout.vocab_end == 0) {
        printf("FATAL: Could not scan vocab section on SD.\n");
        goto halt;
    }
    compute_layout(&ctx.layout, &ctx.config);

    // KV cache lives on SD after wcls data
    ctx.layout.kv_base_block = (ctx.layout.wcls_end + 511) / 512;

    printf("Model layout (byte offsets on SD):\n");
    printf("  weights start:  %lu\n", (unsigned long)ctx.layout.vocab_end);
    printf("  layer size:     %lu bytes\n", (unsigned long)ctx.layout.layer_bytes);
    printf("  final_norm:     %lu\n", (unsigned long)ctx.layout.final_norm_off);
    printf("  classifier:     %lu\n", (unsigned long)ctx.layout.wcls_off);
    printf("  wcls_end:       %lu\n", (unsigned long)ctx.layout.wcls_end);
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

    // Switch to 250 MHz now that SD init is done at 200 MHz.
    // SDIO driver auto-adjusts PIO divider (250/5 = 50 MHz HS).
    set_sys_clock_khz(250000, true);
    printf("System clock: 250 MHz\n");

    // --- Initialize run state ---
    init_run_state(&ctx.state);
    ctx.state.rng_state = (uint64_t)time_us_64();

    // --- Initialize weight buffers and Core 1 compute worker ---
    static WeightBuf wb;
    weightbuf_init(&wb, weight_buf_a, weight_buf_b, WEIGHT_BUF_SIZE);
    ctx.wb = &wb;
    multicore_launch_core1(compute_worker);
    printf("Weight streaming ready (dual-core, Q4_0).\n\n");

    // --- Print RAM usage ---
    printf("RAM usage:\n");
    printf("  Weight buffers: %d bytes (x2)\n", WEIGHT_BUF_SIZE);
    printf("  Activations:    %lu bytes\n",
           (unsigned long)(sizeof(x_buf) + sizeof(xb_buf) + sizeof(xb2_buf)));
    printf("  Attention:      %lu bytes\n",
           (unsigned long)(sizeof(q_buf) + sizeof(k_buf) + sizeof(v_buf) + sizeof(att_buf)));
    printf("  FFN:            %lu bytes\n",
           (unsigned long)(sizeof(hb_buf) + sizeof(hb2_buf)));
    printf("  Final norm:     %lu bytes\n", (unsigned long)sizeof(final_norm_buf));
    printf("  Emb read buf:   %lu bytes\n", (unsigned long)sizeof(emb_read_buf));
    printf("  Matmul acc:     %lu bytes\n", (unsigned long)sizeof(matmul_acc));
    printf("\n");

    printf("pico-llm ready (Qwen3-0.6B Q4_0). No on-device tokenizer.\n");
    printf("  temperature=%.2f  repetition_penalty=%.2f\n",
           ctx.temperature, ctx.repetition_penalty);
    printf("Commands: /temp N, /rep N, /greedy\n");
    printf("Otherwise enter comma-separated token IDs to generate.\n\n");

    // --- Interactive generation loop (token ID input) ---
    while (1) {
        printf("> ");

        char input[512];
        int len = 0;
        while (len < (int)sizeof(input) - 1) {
            int c = getchar();
            if (c == '\n' || c == '\r') {
                putchar('\n');
                break;
            }
            if (c == 0x7F || c == '\b') {
                if (len > 0) {
                    len--;
                    printf("\b \b");
                }
                continue;
            }
            if (c >= 0x20 && c < 0x7F) {
                input[len++] = (char)c;
                putchar(c);
            }
        }
        input[len] = '\0';

        if (len == 0) continue;

        // Handle slash commands
        if (input[0] == '/') {
            if (strncmp(input, "/temp ", 6) == 0) {
                ctx.temperature = strtof(input + 6, NULL);
                printf("temperature = %.2f\n", ctx.temperature);
                continue;
            } else if (strncmp(input, "/rep ", 5) == 0) {
                ctx.repetition_penalty = strtof(input + 5, NULL);
                printf("repetition_penalty = %.2f\n", ctx.repetition_penalty);
                continue;
            } else if (strcmp(input, "/greedy") == 0) {
                ctx.temperature = 0.0f;
                ctx.repetition_penalty = 1.0f;
                printf("Greedy decoding (temp=0, rep=1)\n");
                continue;
            } else {
                printf("Unknown command: %s\n", input);
                continue;
            }
        }

        // Parse comma-separated token IDs
        int tokens[256];
        int n_prompt = 0;
        char *p = input;
        while (*p && n_prompt < 256) {
            tokens[n_prompt++] = (int)strtol(p, &p, 10);
            if (*p == ',') p++;
        }

        ctx.state.recent_count = 0;

        printf("[%d prompt tokens] ", n_prompt);
        generate(&ctx, NULL, tokens, n_prompt, 64);
        printf("\n");
    }

halt:
    while (1) {
        tight_loop_contents();
    }

    return 0;
}
