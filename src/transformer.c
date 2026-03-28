#include "transformer.h"
#include "quantize.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

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

        // Rotate q
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        // Rotate k (only up to kv_dim, handled by caller passing correct dim)
        if (k) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i]     = k0 * cos_val - k1 * sin_val;
            k[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// Online softmax attention: streams over KV cache entries one at a time.
// Avoids materializing the full (seq_len x seq_len) attention matrix.
// q: [head_size], k_row/v_row: single KV entry [head_size]
// out: [head_size] accumulated weighted values
static void attention_online_init(float *max_score, float *sum_exp, float *out,
                                  int head_size) {
    *max_score = -1e30f;
    *sum_exp = 0.0f;
    memset(out, 0, head_size * sizeof(float));
}

static void attention_online_step(const float *q, const float *k_row,
                                  const float *v_row, float *max_score,
                                  float *sum_exp, float *out, int head_size) {
    // Compute attention score: dot(q, k) / sqrt(head_size)
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
        score += q[i] * k_row[i];
    }
    score /= sqrtf((float)head_size);

    // Online softmax update
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

float *forward(TransformerContext *ctx, int token, int pos) {
    Config *cfg = &ctx->config;
    RunState *s = &ctx->state;
    int dim = cfg->dim;
    int head_size = dim / cfg->n_heads;
    int kv_dim = head_size * cfg->n_kv_heads;

    (void)token;
    (void)pos;
    (void)kv_dim;

    // TODO: Load token embedding from SD card into s->x

    // Each layer: attention + FFN with residual connections
    for (int l = 0; l < cfg->n_layers; l++) {
        // TODO: Stream layer weights from SD via double-buffer

        // --- Attention block ---
        // rmsnorm(s->xb, s->x, layer_weights.attn_norm, dim);
        // matmul_q8(s->q, layer_weights.wq, layer_weights.wq_scale, s->xb, dim, dim);
        // matmul_q8(s->k, layer_weights.wk, layer_weights.wk_scale, s->xb, kv_dim, dim);
        // matmul_q8(s->v, layer_weights.wv, layer_weights.wv_scale, s->xb, kv_dim, dim);
        // rope(s->q, s->k, dim, head_size, pos);

        // TODO: Write K,V to SD card KV cache
        // TODO: Online attention over KV cache streamed from SD

        // matmul_q8(s->xb2, layer_weights.wo, layer_weights.wo_scale, s->xb, dim, dim);
        // Residual: x += attn_out
        // for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        // --- FFN block (SwiGLU) ---
        // rmsnorm(s->xb, s->x, layer_weights.ffn_norm, dim);
        // matmul_q8(s->hb, layer_weights.w_gate, layer_weights.w_gate_scale, s->xb, cfg->hidden_dim, dim);
        // matmul_q8(s->hb2, layer_weights.w_up, layer_weights.w_up_scale, s->xb, cfg->hidden_dim, dim);
        // for (int i = 0; i < cfg->hidden_dim; i++) {
        //     s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        // }
        // matmul_q8(s->xb2, layer_weights.w_down, layer_weights.w_down_scale, s->hb, dim, cfg->hidden_dim);
        // Residual: x += ffn_out
        // for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
    }

    // Final norm
    // rmsnorm(s->x, s->x, ctx->final_norm, dim);

    // Classifier: logits = wcls @ x
    // matmul_q8(s->logits, ctx->wcls, ctx->wcls_scale, s->x, cfg->vocab_size, dim);

    return s->logits;
}

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
        float *logits = forward(ctx, token, pos);

        if (pos < n_prompt - 1) {
            // Still processing prompt, force next prompt token
            token = prompt_tokens[pos + 1];
        } else {
            // Greedy sampling
            token = argmax(logits, ctx->config.vocab_size);
            tokens_generated++;
        }
    }

    return tokens_generated;
}
