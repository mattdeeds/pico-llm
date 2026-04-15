#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdint.h>
#include <stdbool.h>
#include "tokenizer.h"

// Derived dimension macros (from compile-time constants)
#define Q_DIM  (N_HEADS * HEAD_DIM)
#define KV_DIM (N_KV_HEADS * HEAD_DIM)

// Model configuration, read from the weight file header (V2: 8 fields)
typedef struct {
    int dim;        // d_model (e.g. 1024)
    int hidden_dim; // FFN intermediate dim (e.g. 3072)
    int n_layers;
    int n_heads;
    int n_kv_heads; // for grouped-query attention (GQA)
    int vocab_size;
    int max_seq_len;
    int head_dim;   // explicit head dimension (e.g. 128, may differ from dim/n_heads)
} Config;

// Per-layer weight pointers (point into weight buffer, not owned)
typedef struct {
    // Attention
    float *attn_norm;   // [dim]
    int8_t *wq;         // [q_dim * dim]
    int8_t *wk;         // [kv_dim * dim]
    int8_t *wv;         // [kv_dim * dim]
    int8_t *wo;         // [dim * q_dim]
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
    float *xb;      // [Q_DIM] scratch buffer (also used for attention output)
    float *xb2;     // [dim] scratch buffer 2

    // Attention working buffers
    float *q;       // [Q_DIM]
    float *k;       // [KV_DIM]
    float *v;       // [KV_DIM]
    float *att;     // [n_heads] online attention accumulator

    // FFN working buffers
    float *hb;      // [hidden_dim] gate output
    float *hb2;     // [hidden_dim] up output

    // KV cache offset tracking (cache lives on SD card)
    int kv_cache_len;
} RunState;

// Byte offsets into the model file on SD card
typedef struct {
    uint32_t vocab_end;       // end of vocab section = start of layer weights
    uint32_t layer_bytes;     // size of one layer's weights in bytes
    uint32_t final_norm_off;  // byte offset to final_norm
    uint32_t wcls_off;        // byte offset to classifier weights
    uint32_t wcls_scale_off;  // byte offset to wcls scale (for tied embedding dequant)
    uint32_t kv_base_block;   // first SD block of KV cache region
} ModelLayout;

// Cursor for sequentially reading weights from the double-buffered SD stream
typedef struct {
    struct WeightBuf *wb;
    uint32_t sd_off;        // SD byte offset of next byte to consume
    uint32_t end_sd_off;    // SD byte offset past the end of this stream
    uint32_t buf_off;       // current read position within buffer
    uint32_t buf_valid;     // number of valid bytes in buffer
    int8_t *buf_ptr;        // pointer to current buffer data
    bool prefetch_pending;  // true if Core 1 has a prefetch in flight
} WeightStream;

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

    // Wcls scale for tied embedding dequantization
    float wcls_scale;
} TransformerContext;

// Compile-time RAM budget checks
_Static_assert(D_MODEL * sizeof(float) <= 4096,
    "d_model activation buffer exceeds 4KB");
_Static_assert(WEIGHT_BUF_SIZE <= 32768,
    "weight buffer exceeds 32KB");

// Core inference functions
void transformer_init(TransformerContext *ctx);
int forward(TransformerContext *ctx, int token, int pos);
int generate(TransformerContext *ctx, const Tokenizer *tok,
             int *prompt_tokens, int n_prompt, int max_tokens);

// Load a single token embedding from SD card into out[dim]
// (tied embeddings: reads int8 from wcls section, dequantizes)
bool load_token_embedding(const TransformerContext *ctx, int token_id, float *out);

// Weight stream functions
void ws_init(WeightStream *ws, struct WeightBuf *wb, uint32_t sd_byte_offset,
             uint32_t total_bytes);
void ws_ensure(WeightStream *ws, uint32_t min_bytes);
void ws_read_bytes(WeightStream *ws, void *out, uint32_t nbytes);
void ws_drain(WeightStream *ws);
void ws_resume(WeightStream *ws);

// Tiled streaming matmul: reads rows*cols int8 weights + scale from stream
void ws_tiled_matmul(WeightStream *ws, float *out, const int8_t *x_q,
                     float x_scale, int rows, int cols);

// Streaming argmax: processes classifier in chunks, returns best token ID
// without storing full logits vector (int32 comparison preserves ordering)
int ws_streaming_argmax(WeightStream *ws, const int8_t *x_q,
                        float x_scale, int rows, int cols);

// Math primitives
void rmsnorm(float *out, const float *x, const float *weight, int size);
void rope_split_half(float *q, float *k, int q_dim, int kv_dim,
                     int head_dim, int pos);
void qk_norm(float *q, float *k, const float *q_norm_w,
             const float *k_norm_w, int n_heads, int n_kv_heads, int head_dim);
void softmax(float *x, int size);
float silu(float x);

#endif
