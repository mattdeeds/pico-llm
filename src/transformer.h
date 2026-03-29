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

// Byte offsets into the model file on SD card
typedef struct {
    uint32_t vocab_end;       // end of vocab section = start of layer weights
    uint32_t layer_bytes;     // size of one layer's weights in bytes
    uint32_t final_norm_off;  // byte offset to final_norm
    uint32_t wcls_off;        // byte offset to classifier weights
    uint32_t emb_off;         // byte offset to token embedding table
} ModelLayout;

// Forward declaration (defined in sdcard.h)
struct WeightBuf;

// Top-level context
typedef struct {
    Config config;
    RunState state;
    ModelLayout layout;

    // Double-buffer for streaming weights from SD
    struct WeightBuf *wb;

    // Final layer norm (small enough to keep in RAM)
    float *final_norm;    // [dim]
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

// Load a single token embedding from SD card into out[dim]
bool load_token_embedding(const TransformerContext *ctx, int token_id, float *out);

// Math primitives
void rmsnorm(float *out, const float *x, const float *weight, int size);
void rope(float *q, float *k, int dim, int head_size, int pos);
void softmax(float *x, int size);
float silu(float x);

#endif
