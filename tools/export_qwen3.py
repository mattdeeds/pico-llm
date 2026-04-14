#!/usr/bin/env python3
"""Export Qwen3 model from HuggingFace to pico-llm V2 binary format.

V2 binary format (extends V1 with head_dim, QK-Norm, tied embeddings):

Header:
    8 x int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
               max_seq_len, head_dim

    Vocab data: for each token, uint16 length + UTF-8 bytes

Weights (streaming order per forward pass):
    For each layer:
        attn_norm:  float32[dim]
        q_norm:     float32[head_dim]          (QK-Norm)
        k_norm:     float32[head_dim]          (QK-Norm)
        wq:         int8[q_dim * dim]          + float32 scale
        wk:         int8[kv_dim * dim]         + float32 scale
        wv:         int8[kv_dim * dim]         + float32 scale
        wo:         int8[dim * q_dim]          + float32 scale
        ffn_norm:   float32[dim]
        w_gate:     int8[hidden_dim * dim]     + float32 scale
        w_up:       int8[hidden_dim * dim]     + float32 scale
        w_down:     int8[dim * hidden_dim]     + float32 scale

    final_norm: float32[dim]
    wcls:       int8[vocab_size * dim]         + float32 scale
    (No separate token_emb -- tied embeddings; firmware dequantizes wcls rows)

Where:
    q_dim  = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

Model-specific constants not in binary (handled by firmware compile-time flags):
    RoPE theta = 1,000,000
    RMSNorm eps = 1e-6
"""

import struct
import os
import argparse
import numpy as np


def quantize_int8(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization."""
    absmax = np.max(np.abs(tensor))
    if absmax == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    scale = absmax / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def write_config(f, config: dict):
    """Write V2 model config as 8 int32 values."""
    f.write(struct.pack('<8i',
        config['dim'],
        config['hidden_dim'],
        config['n_layers'],
        config['n_heads'],
        config['n_kv_heads'],
        config['vocab_size'],
        config['max_seq_len'],
        config['head_dim'],
    ))


def write_float32(f, arr: np.ndarray):
    """Write a float32 array."""
    f.write(arr.astype(np.float32).tobytes())


def write_int8_with_scale(f, arr: np.ndarray):
    """Quantize to int8 and write with scale factor."""
    q, scale = quantize_int8(arr)
    f.write(q.tobytes())
    f.write(struct.pack('<f', scale))


def load_vocab(tokenizer, vocab_size: int) -> list[bytes]:
    """Extract vocabulary as list[bytes] indexed by token ID.

    Uses convert_ids_to_tokens which handles tiktoken byte-level tokens
    correctly, including special tokens.
    """
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

    result = []
    for i, token in enumerate(tokens):
        if token is None:
            result.append(f"<{i}>".encode('utf-8'))
        else:
            result.append(token.encode('utf-8'))
    return result


def export_qwen3(output_path: str, model_name: str, max_seq_len: int):
    """Export a Qwen3 model from HuggingFace to pico-llm V2 binary."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    hf_config = model.config
    state = model.state_dict()

    config = {
        'dim': hf_config.hidden_size,
        'hidden_dim': hf_config.intermediate_size,
        'n_layers': hf_config.num_hidden_layers,
        'n_heads': hf_config.num_attention_heads,
        'n_kv_heads': hf_config.num_key_value_heads,
        'vocab_size': hf_config.vocab_size,
        'max_seq_len': max_seq_len,
        'head_dim': hf_config.head_dim,
    }

    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    head_dim = config['head_dim']
    q_dim = config['n_heads'] * head_dim
    kv_dim = config['n_kv_heads'] * head_dim
    vocab_size = config['vocab_size']

    print(f"  dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
          f"heads={config['n_heads']}, kv_heads={config['n_kv_heads']}, "
          f"head_dim={head_dim}, vocab={vocab_size}")
    print(f"  q_dim={q_dim}, kv_dim={kv_dim}")

    # Verify tied embeddings
    emb_key = 'model.embed_tokens.weight'
    cls_key = 'lm_head.weight'
    if cls_key in state and emb_key in state:
        if torch.equal(state[emb_key], state[cls_key]):
            print("  Tied embeddings: verified (embed_tokens == lm_head)")
        else:
            print("  WARNING: embed_tokens and lm_head weights differ!")
            print("  Using lm_head.weight for classifier/embedding")
    elif emb_key in state:
        print("  lm_head.weight not in state_dict, using embed_tokens.weight")
        state[cls_key] = state[emb_key]
    else:
        raise RuntimeError(
            "Neither lm_head.weight nor model.embed_tokens.weight found")

    # Verify expected weight shapes
    wq = state['model.layers.0.self_attn.q_proj.weight']
    assert wq.shape == (q_dim, dim), \
        f"wq shape {wq.shape} != expected ({q_dim}, {dim})"
    wo = state['model.layers.0.self_attn.o_proj.weight']
    assert wo.shape == (dim, q_dim), \
        f"wo shape {wo.shape} != expected ({dim}, {q_dim})"
    wk = state['model.layers.0.self_attn.k_proj.weight']
    assert wk.shape == (kv_dim, dim), \
        f"wk shape {wk.shape} != expected ({kv_dim}, {dim})"
    qn = state['model.layers.0.self_attn.q_norm.weight']
    assert qn.shape == (head_dim,), \
        f"q_norm shape {qn.shape} != expected ({head_dim},)"

    print(f"  Weight shapes verified")

    # Load vocabulary
    print(f"Loading tokenizer vocabulary ({vocab_size} tokens)...")
    vocab_tokens = load_vocab(tokenizer, vocab_size)

    # Write binary
    print(f"Writing to {output_path}...")
    with open(output_path, 'wb') as f:
        write_config(f, config)

        # Vocab section
        for token_bytes in vocab_tokens:
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)

        vocab_end = f.tell()
        print(f"  Vocab section: {vocab_end} bytes "
              f"({vocab_end / 1024 / 1024:.1f} MB)")

        # Per-layer weights in streaming consumption order
        for l in range(n_layers):
            prefix = f'model.layers.{l}'

            write_float32(f, state[
                f'{prefix}.input_layernorm.weight'].numpy())
            write_float32(f, state[
                f'{prefix}.self_attn.q_norm.weight'].numpy())
            write_float32(f, state[
                f'{prefix}.self_attn.k_norm.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.self_attn.q_proj.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.self_attn.k_proj.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.self_attn.v_proj.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.self_attn.o_proj.weight'].numpy())
            write_float32(f, state[
                f'{prefix}.post_attention_layernorm.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.mlp.gate_proj.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.mlp.up_proj.weight'].numpy())
            write_int8_with_scale(f, state[
                f'{prefix}.mlp.down_proj.weight'].numpy())

            layer_end = f.tell()
            if l == 0:
                layer_bytes = layer_end - vocab_end
                print(f"  Layer size: {layer_bytes} bytes "
                      f"({layer_bytes / 1024 / 1024:.1f} MB)")
            print(f"  Layer {l + 1}/{n_layers}", end='\r')

        print(f"  Layer {n_layers}/{n_layers}")

        # Final norm
        write_float32(f, state['model.norm.weight'].numpy())

        # Classifier (= embedding, tied)
        print(f"  Classifier ({vocab_size} x {dim})...")
        write_int8_with_scale(f, state[cls_key].numpy())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export Qwen3 model to pico-llm V2 binary format')
    parser.add_argument('--output', '-o', default='qwen3_model.bin',
                        help='Output file path')
    parser.add_argument('--model-name', default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model name')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length for KV cache')
    args = parser.parse_args()

    export_qwen3(args.output, args.model_name, args.max_seq_len)
