#include "transformer.h"
#include "sdcard.h"
#include "quantize.h"
#include "pico/stdlib.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// Math primitives
// ============================================================================

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
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

void rope(float *q, float *k, int dim, int head_size, int pos) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
        float val = pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        if (k) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i]     = k0 * cos_val - k1 * sin_val;
            k[i + 1] = k0 * sin_val + k1 * cos_val;
        }
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
    // Discard any pending prefetch — it may not cover sd_off
    if (ws->prefetch_pending) {
        weightbuf_get(ws->wb);
        ws->prefetch_pending = false;
    }

    // Re-read from current sd_off so the buffer is correctly aligned
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
    // Compute SD offset of the byte right after what's in the current buffer
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
    // Stack accumulator — max 704*4 = 2816 bytes for hidden_dim
    int32_t acc[HIDDEN_DIM > D_MODEL ? HIDDEN_DIM : D_MODEL];
    int8_t row_tmp[HIDDEN_DIM > D_MODEL ? HIDDEN_DIM : D_MODEL];

    int rows_done = 0;
    while (rows_done < rows) {
        ws_ensure(ws, 1);
        uint32_t avail = ws->buf_valid - ws->buf_off;
        int tile_rows = avail / cols;
        if (tile_rows > rows - rows_done) tile_rows = rows - rows_done;

        if (tile_rows > 0) {
            matmul_q8_tile(acc + rows_done,
                           (const int8_t *)ws->buf_ptr + ws->buf_off,
                           x_q, tile_rows, cols);
            ws->buf_off += tile_rows * cols;
            ws->sd_off += tile_rows * cols;
            rows_done += tile_rows;
        } else {
            // Row spans buffer boundary — read into scratch and process
            ws_read_bytes(ws, row_tmp, cols);
            matmul_q8_tile(acc + rows_done, row_tmp, x_q, 1, cols);
            rows_done++;
        }
    }

    // Read the float32 scale that follows the int8 weight data
    float w_scale;
    ws_read_bytes(ws, &w_scale, sizeof(float));

    dequant_acc(out, acc, w_scale, x_scale, rows);
}

// Chunked variant for large output dimensions (classifier: vocab_size rows).
// Processes CHUNK_ROWS rows at a time to avoid a huge VLA.
#define CHUNK_ROWS 256

void ws_tiled_matmul_large(WeightStream *ws, float *out, const int8_t *x_q,
                           float x_scale, int rows, int cols) {
    int32_t acc[CHUNK_ROWS];
    int8_t row_tmp[D_MODEL];

    int rows_done = 0;
    while (rows_done < rows) {
        int chunk = rows - rows_done;
        if (chunk > CHUNK_ROWS) chunk = CHUNK_ROWS;

        int chunk_done = 0;
        while (chunk_done < chunk) {
            ws_ensure(ws, 1);
            uint32_t avail = ws->buf_valid - ws->buf_off;
            int tile = avail / cols;
            if (tile > chunk - chunk_done) tile = chunk - chunk_done;

            if (tile > 0) {
                matmul_q8_tile(acc + chunk_done,
                               (const int8_t *)ws->buf_ptr + ws->buf_off,
                               x_q, tile, cols);
                ws->buf_off += tile * cols;
                ws->sd_off += tile * cols;
                chunk_done += tile;
            } else {
                // Row spans buffer boundary — read into scratch and process
                ws_read_bytes(ws, row_tmp, cols);
                matmul_q8_tile(acc + chunk_done, row_tmp, x_q, 1, cols);
                chunk_done++;
            }
        }

        // Store as float with scale=1 (corrected after reading the real scale)
        for (int i = 0; i < chunk; i++) {
            out[rows_done + i] = (float)acc[i];
        }
        rows_done += chunk;
    }

    // Read the real scale and apply
    float w_scale;
    ws_read_bytes(ws, &w_scale, sizeof(float));

    float combined = w_scale * x_scale;
    for (int i = 0; i < rows; i++) {
        out[i] *= combined;
    }
}

// ============================================================================
// KV cache on SD card
// ============================================================================

// Bytes per KV entry: K[kv_dim] + V[kv_dim], both float32
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
    int kv_dim = (ctx->config.dim / ctx->config.n_heads) * ctx->config.n_kv_heads;
    int kv_bytes = kv_dim * (int)sizeof(float);
    int entry_blocks = KV_ENTRY_BLOCKS(kv_dim);

    // Pack K and V into the idle prefetch buffer
    uint8_t *buf = (uint8_t *)ctx->wb->prefetch;
    memcpy(buf, k, kv_bytes);
    memcpy(buf + kv_bytes, v, kv_bytes);

    sdcard_write_blocks(kv_block(ctx, layer, pos, entry_blocks),
                        buf, entry_blocks);
}

// Online attention over KV cache stored on SD.
// Reads KV entries in batches for efficiency, processes all heads per batch.
static void attention_sd(TransformerContext *ctx, int layer, int pos,
                         const float *q, float *out) {
    int dim = ctx->config.dim;
    int n_heads = ctx->config.n_heads;
    int n_kv_heads = ctx->config.n_kv_heads;
    int head_size = dim / n_heads;
    int kv_dim = head_size * n_kv_heads;
    int kv_bytes = kv_dim * (int)sizeof(float);
    int entry_bytes = 2 * kv_bytes;
    int entry_blocks = entry_bytes / 512;
    int heads_per_kv = n_heads / n_kv_heads;

    // Reuse the idle prefetch buffer for batch reads
    uint8_t *buf = (uint8_t *)ctx->wb->prefetch;
    int entries_per_batch = ctx->wb->buf_size / entry_bytes;

    // Initialize all heads
    float max_scores[N_HEADS], sum_exps[N_HEADS];
    for (int h = 0; h < n_heads; h++) {
        attention_online_init(&max_scores[h], &sum_exps[h],
                              out + h * head_size, head_size);
    }

    // Batch-read KV entries from SD
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
// Forward pass
// ============================================================================

float *forward(TransformerContext *ctx, int token, int pos) {
    Config *cfg = &ctx->config;
    RunState *s = &ctx->state;
    int dim = cfg->dim;
    int kv_dim = (dim / cfg->n_heads) * cfg->n_kv_heads;
    int head_size = dim / cfg->n_heads;
    int hidden_dim = cfg->hidden_dim;

    // 1. Load token embedding
    if (!load_token_embedding(ctx, token, s->x))
        return NULL;

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

        // Quantize normalized input once for Q/K/V projections
        int8_t xb_q[D_MODEL];
        float xb_scale = quantize_vec(xb_q, s->xb, dim);

        // Q, K, V projections (tiled matmul from stream)
        ws_tiled_matmul(&ws, s->q, xb_q, xb_scale, dim, dim);
        ws_tiled_matmul(&ws, s->k, xb_q, xb_scale, kv_dim, dim);
        ws_tiled_matmul(&ws, s->v, xb_q, xb_scale, kv_dim, dim);

        // RoPE
        rope(s->q, s->k, kv_dim, head_size, pos);

        // === Phase 2: Attention (Core 0 owns SD) ===

        ws_drain(&ws);

        // Write K, V to SD KV cache
        kv_cache_write(ctx, l, pos, s->k, s->v);

        // Online attention over SD KV cache → output in s->xb
        attention_sd(ctx, l, pos, s->q, s->xb);

        // === Phase 3: Post-attention weights (streamed) ===

        ws_resume(&ws);

        // Output projection: wo @ attention_output
        int8_t att_q[D_MODEL];
        float att_scale = quantize_vec(att_q, s->xb, dim);
        ws_tiled_matmul(&ws, s->xb2, att_q, att_scale, dim, dim);

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
        int8_t hb_q[HIDDEN_DIM];
        float hb_scale = quantize_vec(hb_q, s->hb, hidden_dim);
        ws_tiled_matmul(&ws, s->xb2, hb_q, hb_scale, dim, hidden_dim);

        // Residual
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        ws_drain(&ws);
    }

    // Final norm (weights already in RAM)
    rmsnorm(s->x, s->x, ctx->final_norm, dim);

    // Classifier: stream wcls from SD
    uint32_t wcls_bytes = (uint32_t)cfg->vocab_size * dim + sizeof(float);
    WeightStream ws;
    ws_init(&ws, ctx->wb, ctx->layout.wcls_off, wcls_bytes);

    int8_t x_q[D_MODEL];
    float x_scale = quantize_vec(x_q, s->x, dim);
    ws_tiled_matmul_large(&ws, s->logits, x_q, x_scale, cfg->vocab_size, dim);
    ws_drain(&ws);

    s->kv_cache_len = pos + 1;
    return s->logits;
}

// ============================================================================
// Token generation
// ============================================================================

static int argmax(const float *v, int n) {
    int max_i = 0;
    float max_v = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_v) {
            max_v = v[i];
            max_i = i;
        }
    }
    return max_i;
}

int generate(TransformerContext *ctx, int *prompt_tokens, int n_prompt, int max_tokens) {
    int token = prompt_tokens[0];
    int pos = 0;
    int tokens_generated = 0;

    for (pos = 0; pos < n_prompt + max_tokens; pos++) {
        absolute_time_t t0 = get_absolute_time();
        float *logits = forward(ctx, token, pos);
        int64_t elapsed_ms = absolute_time_diff_us(t0, get_absolute_time()) / 1000;

        if (!logits) {
            printf("[forward returned NULL at pos %d]\n", pos);
            break;
        }

        if (pos < n_prompt - 1) {
            token = prompt_tokens[pos + 1];
        } else {
            token = argmax(logits, ctx->config.vocab_size);
            tokens_generated++;
            if (tokens_generated <= 3) {
                // Print first few logits for diagnostics
                printf("\n  logits[0..4]: %.3f %.3f %.3f %.3f %.3f\n",
                       logits[0], logits[1], logits[2], logits[3], logits[4]);
                printf("  logits[8188..8191]: %.3f %.3f %.3f %.3f\n",
                       logits[8188], logits[8189], logits[8190], logits[8191]);
            }
            printf("[tok %d = %d, %lld ms] ", tokens_generated, token, (long long)elapsed_ms);
        }
    }

    return tokens_generated;
}
