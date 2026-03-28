#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>
#include <stdbool.h>

// Model configuration, read from the weight file header
typedef struct {
    int dim;        // d_model (e.g. 256)
    int hidden_dim; // FFN intermediate dim (e.g. 704)
    int n_layers;
    int n_heads;
    int n_kv_heads; // for grouped-query attention (GQA)
    int vocab_size;
    int max_seq_len;
} Config;

// Per-layer weight pointers (point into weight buffer, not owned)
typedef struct {
    // Attention
    float *attn_norm;   // [dim]
    int8_t *wq;         // [dim * dim]
    int8_t *wk;         // [dim * kv_dim]
    int8_t *wv;         // [dim * kv_dim]
    int8_t *wo;         // [dim * dim]
    float wq_scale;
    float wk_scale;
    float wv_scale;
    float wo_scale;

    // FFN
    float *ffn_norm;    // [dim]
    int8_t *w_gate;     // [hidden_dim * dim]
    int8_t *w_up;       // [hidden_dim * dim]
    int8_t *w_down;     // [dim * hidden_dim]
    float w_gate_scale;
    float w_up_scale;
    float w_down_scale;
} LayerWeights;

// Mutable inference state
typedef struct {
    // Activation buffers
    float *x;       // [dim] current activation
    float *xb;      // [dim] scratch buffer
    float *xb2;     // [dim] scratch buffer 2

    // Attention working buffers
    float *q;       // [dim]
    float *k;       // [kv_dim]
    float *v;       // [kv_dim]
    float *att;     // [n_heads] online attention accumulator

    // FFN working buffers
    float *hb;      // [hidden_dim] gate output
    float *hb2;     // [hidden_dim] up output

    // Output
    float *logits;  // [vocab_size]

    // KV cache offset tracking (cache lives on SD card)
    int kv_cache_len;
} RunState;

// Top-level context
typedef struct {
    Config config;
    RunState state;

    // Double-buffer pointers (managed by sdcard module)
    int8_t *weight_buf_a;
    int8_t *weight_buf_b;
    int8_t *active_buf;

    // Final layer norm + classifier
    float *final_norm;    // [dim]
    int8_t *wcls;         // [vocab_size * dim]
    float wcls_scale;
} TransformerContext;

// Compile-time RAM budget checks
_Static_assert(D_MODEL * sizeof(float) <= 1024,
    "d_model activation buffer exceeds 1KB");
_Static_assert(WEIGHT_BUF_SIZE <= 32768,
    "weight buffer exceeds 32KB");

// Core inference functions
void transformer_init(TransformerContext *ctx);
float *forward(TransformerContext *ctx, int token, int pos);
int generate(TransformerContext *ctx, int *prompt_tokens, int n_prompt, int max_tokens);

// Math primitives
void rmsnorm(float *out, const float *x, const float *weight, int size);
void rope(float *q, float *k, int dim, int head_size, int pos);
void softmax(float *x, int size);
float silu(float x);

#endif
