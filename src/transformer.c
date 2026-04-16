#include "transformer.h"
#include "sdcard.h"
#include "quantize.h"
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

// Shared state for Core 1 compute worker (defined in sdcard.cpp)
typedef struct {
    const int8_t *x_q;
    float *acc;         // float output (Q1_0_g128 has per-block dequant)
    int cols;
    int rows_done;
} ComputeState;
extern ComputeState g_compute;

// Shared matmul accumulator (defined in main.c, float for Q1_0_g128)
extern float matmul_acc[];

// Q1_0_g128 row size in bytes: (cols / 128) blocks × 18 bytes per block
#define Q1_ROW_BYTES(cols) (((cols) / 128) * 18)

// Max Q1_0_g128 row bytes for any matmul (for row_tmp VLA)
#define MAX_Q1_ROW_BYTES (Q1_ROW_BYTES(HIDDEN_DIM > Q_DIM ? HIDDEN_DIM : Q_DIM))

// ============================================================================
// Math primitives
// ============================================================================

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / size + RMSNORM_EPS);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Split-half RoPE (HuggingFace/Qwen3 convention).
// Pairs element i with element i+half within each head.
void rope_split_half(float *q, float *k, int q_dim, int kv_dim,
                     int head_dim, int pos) {
    int half = head_dim / 2;
    int n_q_heads = q_dim / head_dim;
    int n_kv_heads = kv_dim / head_dim;

    for (int h = 0; h < n_q_heads; h++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (2.0f * i) / (float)head_dim);
            float val = pos * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            int idx1 = h * head_dim + i;
            int idx2 = h * head_dim + i + half;
            float a = q[idx1], b = q[idx2];
            q[idx1] = a * cos_val - b * sin_val;
            q[idx2] = b * cos_val + a * sin_val;
        }
    }

    if (k) {
        for (int h = 0; h < n_kv_heads; h++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(ROPE_THETA, (2.0f * i) / (float)head_dim);
                float val = pos * freq;
                float cos_val = cosf(val);
                float sin_val = sinf(val);
                int idx1 = h * head_dim + i;
                int idx2 = h * head_dim + i + half;
                float a = k[idx1], b = k[idx2];
                k[idx1] = a * cos_val - b * sin_val;
                k[idx2] = b * cos_val + a * sin_val;
            }
        }
    }
}

// Per-head RMSNorm on Q and K vectors (QK-Norm).
void qk_norm(float *q, float *k, const float *q_norm_w,
             const float *k_norm_w, int n_heads, int n_kv_heads, int head_dim) {
    for (int h = 0; h < n_heads; h++) {
        rmsnorm(q + h * head_dim, q + h * head_dim, q_norm_w, head_dim);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        rmsnorm(k + h * head_dim, k + h * head_dim, k_norm_w, head_dim);
    }
}

// ============================================================================
// Online softmax attention
// ============================================================================

static void attention_online_init(float *max_score, float *sum_exp, float *out,
                                  int head_size) {
    *max_score = -1e30f;
    *sum_exp = 0.0f;
    memset(out, 0, head_size * sizeof(float));
}

static void attention_online_step(const float *q, const float *k_row,
                                  const float *v_row, float *max_score,
                                  float *sum_exp, float *out, int head_size) {
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
        score += q[i] * k_row[i];
    }
    score /= sqrtf((float)head_size);

    if (score > *max_score) {
        float correction = expf(*max_score - score);
        *sum_exp *= correction;
        for (int i = 0; i < head_size; i++) {
            out[i] *= correction;
        }
        *max_score = score;
    }

    float w = expf(score - *max_score);
    *sum_exp += w;
    for (int i = 0; i < head_size; i++) {
        out[i] += w * v_row[i];
    }
}

static void attention_online_finalize(float *out, float sum_exp, int head_size) {
    for (int i = 0; i < head_size; i++) {
        out[i] /= sum_exp;
    }
}

// ============================================================================
// Weight stream — sequential reading from double-buffered SD
// ============================================================================

void ws_init(WeightStream *ws, struct WeightBuf *wb, uint32_t sd_byte_offset,
             uint32_t total_bytes) {
    ws->wb = wb;
    ws->sd_off = sd_byte_offset;
    ws->end_sd_off = sd_byte_offset + total_bytes;
    ws->buf_off = 0;
    ws->buf_valid = 0;
    ws->buf_ptr = NULL;
    ws->prefetch_pending = false;

    // Kick off first prefetch
    uint32_t first_size = total_bytes;
    if (first_size > wb->buf_size) first_size = wb->buf_size;
    weightbuf_start_prefetch(wb, sd_byte_offset, first_size);
    ws->prefetch_pending = true;
}

void ws_ensure(WeightStream *ws, uint32_t min_bytes) {
    if (ws->buf_ptr && (ws->buf_valid - ws->buf_off) >= min_bytes)
        return;

    if (!ws->prefetch_pending)
        return; // no more data

    // Wait for the prefetched buffer
    ws->buf_ptr = weightbuf_get(ws->wb);
    ws->prefetch_pending = false;

    // Data starts at sub-block offset within the buffer
    uint32_t skip = ws->sd_off % 512;
    ws->buf_off = skip;

    // Compute valid data range
    uint32_t remaining = ws->end_sd_off - ws->sd_off;
    uint32_t fetched = ws->wb->buf_size - skip;
    if (fetched > remaining) fetched = remaining;
    ws->buf_valid = skip + fetched;

    // Prefetch next chunk
    uint32_t next_sd_off = ws->sd_off + fetched;
    if (next_sd_off < ws->end_sd_off) {
        uint32_t next_size = ws->end_sd_off - next_sd_off;
        if (next_size > ws->wb->buf_size) next_size = ws->wb->buf_size;
        weightbuf_start_prefetch(ws->wb, next_sd_off, next_size);
        ws->prefetch_pending = true;
    }
}

void ws_read_bytes(WeightStream *ws, void *out, uint32_t nbytes) {
    uint8_t *dst = (uint8_t *)out;
    while (nbytes > 0) {
        ws_ensure(ws, 1);
        uint32_t avail = ws->buf_valid - ws->buf_off;
        uint32_t chunk = nbytes < avail ? nbytes : avail;
        memcpy(dst, (uint8_t *)ws->buf_ptr + ws->buf_off, chunk);
        dst += chunk;
        ws->buf_off += chunk;
        ws->sd_off += chunk;
        nbytes -= chunk;
    }
}

void ws_drain(WeightStream *ws) {
    if (ws->prefetch_pending) {
        weightbuf_get(ws->wb);
        ws->prefetch_pending = false;
    }

    uint32_t remaining = ws->end_sd_off - ws->sd_off;
    if (remaining == 0) {
        ws->buf_ptr = NULL;
        ws->buf_off = 0;
        ws->buf_valid = 0;
        return;
    }

    uint32_t size = remaining < ws->wb->buf_size ? remaining : ws->wb->buf_size;
    weightbuf_start_prefetch(ws->wb, ws->sd_off, size);
    ws->buf_ptr = weightbuf_get(ws->wb);

    uint32_t skip = ws->sd_off % 512;
    ws->buf_off = skip;
    uint32_t fetched = ws->wb->buf_size - skip;
    if (fetched > remaining) fetched = remaining;
    ws->buf_valid = skip + fetched;
}

void ws_resume(WeightStream *ws) {
    uint32_t in_buf = ws->buf_valid - ws->buf_off;
    uint32_t next_sd_off = ws->sd_off + in_buf;
    if (next_sd_off < ws->end_sd_off) {
        uint32_t next_size = ws->end_sd_off - next_sd_off;
        if (next_size > ws->wb->buf_size) next_size = ws->wb->buf_size;
        weightbuf_start_prefetch(ws->wb, next_sd_off, next_size);
        ws->prefetch_pending = true;
    }
}

// ============================================================================
// Tiled streaming matmul
// ============================================================================

void ws_tiled_matmul(WeightStream *ws, float *out, const int8_t *x_q,
                     float x_scale, int rows, int cols) {
    uint8_t row_tmp[MAX_Q1_ROW_BYTES];
    int row_bytes = Q1_ROW_BYTES(cols);

    // Set up shared state for Core 1 compute worker
    g_compute.x_q = x_q;
    g_compute.acc = matmul_acc;
    g_compute.cols = cols;
    g_compute.rows_done = 0;

    int rows_sent = 0;
    bool core1_busy = false;

    while (rows_sent < rows) {
        if (core1_busy) {
            multicore_fifo_pop_blocking();
            core1_busy = false;
        }

        ws_ensure(ws, 1);
        uint32_t avail = ws->buf_valid - ws->buf_off;
        int tile_rows = avail / row_bytes;
        if (tile_rows > rows - rows_sent) tile_rows = rows - rows_sent;

        if (tile_rows > 0) {
            multicore_fifo_push_blocking((uint32_t)(ws->buf_ptr + ws->buf_off));
            multicore_fifo_push_blocking((uint32_t)tile_rows);
            core1_busy = true;

            ws->buf_off += tile_rows * row_bytes;
            ws->sd_off += tile_rows * row_bytes;
            rows_sent += tile_rows;
        } else {
            // Row spans buffer boundary — read into temp and process on Core 0
            ws_read_bytes(ws, row_tmp, row_bytes);
            matmul_q1_0_g128_tile(matmul_acc + g_compute.rows_done,
                                  row_tmp, x_q, 1, cols);
            g_compute.rows_done++;
            rows_sent++;
        }
    }

    if (core1_busy) {
        multicore_fifo_pop_blocking();
    }

    // Q1_0_g128 matmul output is already dequantized per-block — just apply x_scale
    for (int i = 0; i < rows; i++) {
        out[i] = matmul_acc[i] * x_scale;
    }
}

// ============================================================================
// Streaming token sampling (Gumbel-max + repetition penalty)
// ============================================================================

#define CHUNK_ROWS 256

// Simple LCG PRNG
static uint32_t rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(*state >> 32);
}

// Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
static float gumbel_noise(uint64_t *rng_state) {
    uint32_t u = rng_next(rng_state);
    float u_norm = ((float)u + 1.0f) / 4294967298.0f;
    return -logf(-logf(u_norm));
}

// Check if token_id is in the recent token history
static bool is_recent(int token_id, const int *recent, int count) {
    int len = count < RECENT_TOKENS_SIZE ? count : RECENT_TOKENS_SIZE;
    for (int i = 0; i < len; i++) {
        if (recent[i % RECENT_TOKENS_SIZE] == token_id)
            return true;
    }
    return false;
}

// Process classifier in CHUNK_ROWS chunks with temperature sampling
// (Gumbel-max trick) and repetition penalty. Returns selected token ID.
//
// When temperature=0: greedy argmax (dequantized float comparison).
// When temperature>0: argmax(logit/T + Gumbel_noise) = sample from softmax(logit/T).
int ws_sample_token(WeightStream *ws, const int8_t *x_q,
                    float x_scale, int rows, int cols,
                    float temperature, float repetition_penalty,
                    const int *recent_tokens, int recent_count,
                    uint64_t *rng_state) {
    uint8_t row_tmp[Q1_ROW_BYTES(D_MODEL)];
    int row_bytes = Q1_ROW_BYTES(cols);
    int best_token = 0;
    float best_score = -1e30f;

    g_compute.x_q = x_q;
    g_compute.cols = cols;

    int rows_done = 0;
    while (rows_done < rows) {
        int chunk = rows - rows_done;
        if (chunk > CHUNK_ROWS) chunk = CHUNK_ROWS;

        g_compute.acc = matmul_acc;
        g_compute.rows_done = 0;

        int chunk_sent = 0;
        bool core1_busy = false;

        while (chunk_sent < chunk) {
            if (core1_busy) {
                multicore_fifo_pop_blocking();
                core1_busy = false;
            }

            ws_ensure(ws, 1);
            uint32_t avail = ws->buf_valid - ws->buf_off;
            int tile = avail / row_bytes;
            if (tile > chunk - chunk_sent) tile = chunk - chunk_sent;

            if (tile > 0) {
                multicore_fifo_push_blocking((uint32_t)(ws->buf_ptr + ws->buf_off));
                multicore_fifo_push_blocking((uint32_t)tile);
                core1_busy = true;

                ws->buf_off += tile * row_bytes;
                ws->sd_off += tile * row_bytes;
                chunk_sent += tile;
            } else {
                ws_read_bytes(ws, row_tmp, row_bytes);
                matmul_q1_0_g128_tile(matmul_acc + g_compute.rows_done,
                                      row_tmp, x_q, 1, cols);
                g_compute.rows_done++;
                chunk_sent++;
            }
        }

        if (core1_busy) {
            multicore_fifo_pop_blocking();
        }

        // Score each token in this chunk
        // Q1_0_g128 matmul output is already per-block dequantized, just apply x_scale
        for (int i = 0; i < chunk; i++) {
            float logit = matmul_acc[i] * x_scale;

            // Repetition penalty
            if (repetition_penalty != 1.0f &&
                is_recent(rows_done + i, recent_tokens, recent_count)) {
                logit = (logit > 0.0f)
                    ? logit / repetition_penalty
                    : logit * repetition_penalty;
            }

            // Temperature + Gumbel-max sampling
            float score;
            if (temperature > 0.0f) {
                score = logit / temperature + gumbel_noise(rng_state);
            } else {
                score = logit;
            }

            if (score > best_score) {
                best_score = score;
                best_token = rows_done + i;
            }
        }
        rows_done += chunk;
    }

    return best_token;
}

// ============================================================================
// KV cache on SD card
// ============================================================================

#define KV_ENTRY_FLOATS(kv_dim) (2 * (kv_dim))
#define KV_ENTRY_BYTES(kv_dim) (KV_ENTRY_FLOATS(kv_dim) * (int)sizeof(float))
#define KV_ENTRY_BLOCKS(kv_dim) (KV_ENTRY_BYTES(kv_dim) / 512)

static uint32_t kv_block(const TransformerContext *ctx, int layer, int pos,
                         int entry_blocks) {
    return ctx->layout.kv_base_block
         + (uint32_t)layer * ctx->config.max_seq_len * entry_blocks
         + (uint32_t)pos * entry_blocks;
}

static void kv_cache_write(TransformerContext *ctx, int layer, int pos,
                           const float *k, const float *v) {
    int kv_dim = ctx->config.n_kv_heads * ctx->config.head_dim;
    int kv_bytes = kv_dim * (int)sizeof(float);
    int entry_blocks = KV_ENTRY_BLOCKS(kv_dim);

    uint8_t *buf = (uint8_t *)ctx->wb->prefetch;
    memcpy(buf, k, kv_bytes);
    memcpy(buf + kv_bytes, v, kv_bytes);

    sdcard_write_blocks(kv_block(ctx, layer, pos, entry_blocks),
                        buf, entry_blocks);
}

// Online attention over KV cache stored on SD.
static void attention_sd(TransformerContext *ctx, int layer, int pos,
                         const float *q, float *out) {
    int n_heads = ctx->config.n_heads;
    int n_kv_heads = ctx->config.n_kv_heads;
    int head_size = ctx->config.head_dim;
    int kv_dim = n_kv_heads * head_size;
    int kv_bytes = kv_dim * (int)sizeof(float);
    int entry_bytes = 2 * kv_bytes;
    int entry_blocks = entry_bytes / 512;
    int heads_per_kv = n_heads / n_kv_heads;

    uint8_t *buf = (uint8_t *)ctx->wb->prefetch;
    int entries_per_batch = ctx->wb->buf_size / entry_bytes;

    float max_scores[N_HEADS], sum_exps[N_HEADS];
    for (int h = 0; h < n_heads; h++) {
        attention_online_init(&max_scores[h], &sum_exps[h],
                              out + h * head_size, head_size);
    }

    for (int t_base = 0; t_base <= pos; t_base += entries_per_batch) {
        int t_end = t_base + entries_per_batch;
        if (t_end > pos + 1) t_end = pos + 1;
        int batch = t_end - t_base;

        sdcard_read_blocks(kv_block(ctx, layer, t_base, entry_blocks),
                           buf, batch * entry_blocks);

        for (int i = 0; i < batch; i++) {
            float *k_row = (float *)(buf + i * entry_bytes);
            float *v_row = (float *)(buf + i * entry_bytes + kv_bytes);

            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                float *k_head = k_row + kv_h * head_size;
                float *v_head = v_row + kv_h * head_size;

                for (int g = 0; g < heads_per_kv; g++) {
                    int h = kv_h * heads_per_kv + g;
                    attention_online_step(
                        q + h * head_size, k_head, v_head,
                        &max_scores[h], &sum_exps[h],
                        out + h * head_size, head_size);
                }
            }
        }
    }

    for (int h = 0; h < n_heads; h++) {
        attention_online_finalize(out + h * head_size, sum_exps[h], head_size);
    }
}

// ============================================================================
// Forward pass (V2 format: QK-Norm, split-half RoPE, streaming argmax)
// ============================================================================

int forward(TransformerContext *ctx, int token, int pos) {
    Config *cfg = &ctx->config;
    RunState *s = &ctx->state;
    int dim = cfg->dim;
    int head_dim = cfg->head_dim;
    int q_dim = cfg->n_heads * head_dim;
    int kv_dim = cfg->n_kv_heads * head_dim;
    int hidden_dim = cfg->hidden_dim;

    // 1. Load token embedding (tied: Q1_0_g128 row from wcls, dequantized)
    if (!load_token_embedding(ctx, token, s->x))
        return -1;

    for (int l = 0; l < cfg->n_layers; l++) {
        uint32_t layer_off = ctx->layout.vocab_end
                           + (uint32_t)l * ctx->layout.layer_bytes;
        WeightStream ws;
        ws_init(&ws, ctx->wb, layer_off, ctx->layout.layer_bytes);

        // === Phase 1: Pre-attention weights (streamed) ===

        // Attention norm
        float attn_norm[D_MODEL];
        ws_read_bytes(&ws, attn_norm, dim * sizeof(float));
        rmsnorm(s->xb, s->x, attn_norm, dim);

        // QK-Norm weights
        float q_norm_w[HEAD_DIM];
        float k_norm_w[HEAD_DIM];
        ws_read_bytes(&ws, q_norm_w, head_dim * sizeof(float));
        ws_read_bytes(&ws, k_norm_w, head_dim * sizeof(float));

        // Quantize normalized input once for Q/K/V projections
        int8_t xb_q[D_MODEL];
        float xb_scale = quantize_vec(xb_q, s->xb, dim);

        // Q, K, V projections (tiled matmul from stream)
        ws_tiled_matmul(&ws, s->q, xb_q, xb_scale, q_dim, dim);
        ws_tiled_matmul(&ws, s->k, xb_q, xb_scale, kv_dim, dim);
        ws_tiled_matmul(&ws, s->v, xb_q, xb_scale, kv_dim, dim);

        // QK-Norm: per-head RMSNorm on Q and K
        qk_norm(s->q, s->k, q_norm_w, k_norm_w,
                cfg->n_heads, cfg->n_kv_heads, head_dim);

        // RoPE (split-half convention)
        rope_split_half(s->q, s->k, q_dim, kv_dim, head_dim, pos);

        // === Phase 2: Attention (Core 0 owns SD) ===

        ws_drain(&ws);

        // Write K, V to SD KV cache
        kv_cache_write(ctx, l, pos, s->k, s->v);

        // Online attention over SD KV cache → output in s->xb [Q_DIM]
        attention_sd(ctx, l, pos, s->q, s->xb);

        // === Phase 3: Post-attention weights (streamed) ===

        ws_resume(&ws);

        // Output projection: wo @ attention_output [dim x q_dim]
        {
            int8_t att_q[Q_DIM];
            float att_scale = quantize_vec(att_q, s->xb, q_dim);
            ws_tiled_matmul(&ws, s->xb2, att_q, att_scale, dim, q_dim);
        }

        // Residual
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        // FFN norm
        float ffn_norm[D_MODEL];
        ws_read_bytes(&ws, ffn_norm, dim * sizeof(float));
        rmsnorm(s->xb, s->x, ffn_norm, dim);

        // Quantize for gate/up
        float xb_scale2 = quantize_vec(xb_q, s->xb, dim);

        // SwiGLU FFN
        ws_tiled_matmul(&ws, s->hb, xb_q, xb_scale2, hidden_dim, dim);
        ws_tiled_matmul(&ws, s->hb2, xb_q, xb_scale2, hidden_dim, dim);

        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        }

        // Down projection
        {
            int8_t hb_q[HIDDEN_DIM];
            float hb_scale = quantize_vec(hb_q, s->hb, hidden_dim);
            ws_tiled_matmul(&ws, s->xb2, hb_q, hb_scale, dim, hidden_dim);
        }

        // Residual
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        ws_drain(&ws);
    }

    // Final norm (weights already in RAM)
    rmsnorm(s->x, s->x, ctx->final_norm, dim);

    // Classifier: streaming sample over wcls (Q1_0_g128)
    uint32_t wcls_bytes = (uint32_t)cfg->vocab_size * Q1_ROW_BYTES(dim);
    WeightStream ws;
    ws_init(&ws, ctx->wb, ctx->layout.wcls_off, wcls_bytes);

    int8_t x_q[D_MODEL];
    float x_scale = quantize_vec(x_q, s->x, dim);
    int next_token = ws_sample_token(
        &ws, x_q, x_scale, cfg->vocab_size, dim,
        ctx->temperature, ctx->repetition_penalty,
        s->recent_tokens, s->recent_count, &s->rng_state);
    ws_drain(&ws);

    s->kv_cache_len = pos + 1;
    return next_token;
}

// ============================================================================
// Token generation
// ============================================================================

int generate(TransformerContext *ctx, const Tokenizer *tok,
             int *prompt_tokens, int n_prompt, int max_tokens) {
    int token = prompt_tokens[0];
    int pos = 0;
    int tokens_generated = 0;

    absolute_time_t gen_start = get_absolute_time();

    for (pos = 0; pos < n_prompt + max_tokens; pos++) {
        absolute_time_t tok_start = get_absolute_time();

        int next_token = forward(ctx, token, pos);

        int64_t tok_ms = absolute_time_diff_us(tok_start, get_absolute_time()) / 1000;

        if (next_token < 0) {
            printf("[forward failed at pos %d]\n", pos);
            break;
        }

        if (pos < n_prompt - 1) {
            token = prompt_tokens[pos + 1];
        } else {
            token = next_token;
            tokens_generated++;

            // Track token in recent history (circular buffer)
            RunState *s = &ctx->state;
            s->recent_tokens[s->recent_count % RECENT_TOKENS_SIZE] = token;
            s->recent_count++;

            // Print token ID (no on-device decoding for large-vocab models)
            if (tok && tok->vocab && token < tok->vocab_size) {
                printf("%s", tok->vocab[token]);
            } else {
                printf("[%d]", token);
            }
            printf(" (%lld ms) ", (long long)tok_ms);
        }
    }

    int64_t total_ms = absolute_time_diff_us(gen_start, get_absolute_time()) / 1000;
    printf("\n\n[%d tokens in %lld ms, %.0f ms/tok]\n",
           tokens_generated, (long long)total_ms,
           tokens_generated > 0 ? (float)total_ms / tokens_generated : 0.0f);

    return tokens_generated;
}
