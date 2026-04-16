#!/usr/bin/env python3
"""Export Qwen3 model from HuggingFace with Q4_0 block quantization.

Q4_0 format (from ggml/llama.cpp):
  Block of 32 weights = 18 bytes:
    - 2 bytes: float16 scale (d = max_element / -8)
    - 16 bytes: 32 nibble-packed values
      qs[j] low nibble  = weight at position j     (0..15)
      qs[j] high nibble = weight at position j+16  (16..31)
  Dequantization: x = (nibble - 8) * d

Binary format (V2 header, Q4_0 weights):
  Header: 8 x int32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                      vocab_size, max_seq_len, head_dim)
  Vocab: per token, uint16 length + UTF-8 bytes
  Per layer:
    attn_norm:  float32[dim]
    q_norm:     float32[head_dim]
    k_norm:     float32[head_dim]
    wq:         Q4_0 blocks [q_dim rows]
    wk:         Q4_0 blocks [kv_dim rows]
    wv:         Q4_0 blocks [kv_dim rows]
    wo:         Q4_0 blocks [dim rows]
    ffn_norm:   float32[dim]
    w_gate:     Q4_0 blocks [hidden_dim rows]
    w_up:       Q4_0 blocks [hidden_dim rows]
    w_down:     Q4_0 blocks [dim rows]
  final_norm: float32[dim]
  wcls:       Q4_0 blocks [vocab_size rows]
  (No per-matrix scale — scales embedded in Q4_0 blocks)
"""

import struct
import os
import argparse
import numpy as np


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    """Quantize a 2D weight matrix to Q4_0 format.

    Each row is divided into blocks of 32 columns. Each block becomes
    18 bytes: 2 bytes float16 scale + 16 bytes packed nibbles.

    Args:
        tensor: float32 array of shape [rows, cols], cols must be multiple of 32
    Returns:
        bytes: Q4_0 encoded data
    """
    rows, cols = tensor.shape
    assert cols % 32 == 0, f"cols={cols} must be multiple of 32"

    result = bytearray()
    for r in range(rows):
        row = tensor[r]
        for b in range(cols // 32):
            block = row[b * 32:(b + 1) * 32]

            # Scale: d = max_element / -8 (ggml convention, preserves sign)
            idx = np.argmax(np.abs(block))
            max_val = float(block[idx])
            d = max_val / -8.0
            d_f16 = np.float16(d)
            result += d_f16.tobytes()

            # Quantize: q = clip(trunc(x / d + 8.5), 0, 15)
            d_f32 = float(d_f16)  # use the float16-rounded scale
            inv_d = 1.0 / d_f32 if d_f32 != 0 else 0.0

            # Pack: low nibble = first half (0..15), high nibble = second half (16..31)
            for j in range(16):
                x0 = float(block[j]) * inv_d + 8.5
                x1 = float(block[j + 16]) * inv_d + 8.5
                q0 = max(0, min(15, int(x0)))
                q1 = max(0, min(15, int(x1)))
                result.append(q0 | (q1 << 4))

    return bytes(result)


def write_config(f, config: dict):
    """Write V2 model config as 8 int32 values."""
    f.write(struct.pack('<8i',
        config['dim'], config['hidden_dim'], config['n_layers'],
        config['n_heads'], config['n_kv_heads'], config['vocab_size'],
        config['max_seq_len'], config['head_dim']))


def write_float32(f, arr: np.ndarray):
    """Write a float32 array."""
    f.write(arr.astype(np.float32).tobytes())


def write_q4_0(f, tensor: np.ndarray):
    """Quantize to Q4_0 and write."""
    data = quantize_q4_0(tensor.astype(np.float32))
    f.write(data)


def load_vocab(tokenizer, vocab_size: int) -> list[bytes]:
    """Extract vocabulary as list[bytes] indexed by token ID."""
    tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    result = []
    for i, token in enumerate(tokens):
        if token is None:
            result.append(f"<{i}>".encode('utf-8'))
        else:
            result.append(token.encode('utf-8'))
    return result


def export_qwen3_q4(output_path: str, model_name: str, max_seq_len: int):
    """Export a Qwen3 model with Q4_0 quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name} (this may take a while for 4B)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
        low_cpu_mem_usage=True)
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

    # Verify dimensions are multiples of 32 (Q4_0 block size)
    for name, val in [('dim', dim), ('q_dim', q_dim), ('kv_dim', kv_dim),
                      ('hidden_dim', hidden_dim)]:
        assert val % 32 == 0, f"{name}={val} must be multiple of 32 for Q4_0"

    # Verify tied embeddings
    emb_key = 'model.embed_tokens.weight'
    cls_key = 'lm_head.weight'
    if cls_key in state and emb_key in state:
        if torch.equal(state[emb_key], state[cls_key]):
            print("  Tied embeddings: verified")
        else:
            print("  WARNING: embed_tokens and lm_head differ")
    elif emb_key in state:
        print("  lm_head.weight not in state_dict, using embed_tokens.weight")
        state[cls_key] = state[emb_key]

    # Verify weight shapes
    wq = state['model.layers.0.self_attn.q_proj.weight']
    assert wq.shape == (q_dim, dim), f"wq shape {wq.shape} != ({q_dim}, {dim})"

    # Load vocabulary
    print(f"Loading tokenizer vocabulary ({vocab_size} tokens)...")
    vocab_tokens = load_vocab(tokenizer, vocab_size)

    # Compute expected sizes
    q4_row = lambda cols: (cols // 32) * 18
    layer_size = (dim * 4 + head_dim * 4 * 2  # norms
                  + q_dim * q4_row(dim)        # wq
                  + kv_dim * q4_row(dim) * 2   # wk, wv
                  + dim * q4_row(q_dim)         # wo
                  + dim * 4                     # ffn_norm
                  + hidden_dim * q4_row(dim) * 2  # w_gate, w_up
                  + dim * q4_row(hidden_dim))   # w_down

    print(f"  Layer size: {layer_size / 1024 / 1024:.1f} MB")
    print(f"  Total layers: {n_layers * layer_size / 1024 / 1024:.0f} MB")
    print(f"  Classifier: {vocab_size * q4_row(dim) / 1024 / 1024:.0f} MB")

    print(f"Writing to {output_path}...")
    with open(output_path, 'wb') as f:
        write_config(f, config)

        # Vocab section
        for token_bytes in vocab_tokens:
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)

        vocab_end = f.tell()
        print(f"  Vocab: {vocab_end / 1024 / 1024:.1f} MB")

        # Per-layer weights
        for l in range(n_layers):
            prefix = f'model.layers.{l}'

            # Norms (float32)
            write_float32(f, state[
                f'{prefix}.input_layernorm.weight'].numpy())
            write_float32(f, state[
                f'{prefix}.self_attn.q_norm.weight'].numpy())
            write_float32(f, state[
                f'{prefix}.self_attn.k_norm.weight'].numpy())

            # Weight matrices (Q4_0)
            write_q4_0(f, state[
                f'{prefix}.self_attn.q_proj.weight'].numpy())
            write_q4_0(f, state[
                f'{prefix}.self_attn.k_proj.weight'].numpy())
            write_q4_0(f, state[
                f'{prefix}.self_attn.v_proj.weight'].numpy())
            write_q4_0(f, state[
                f'{prefix}.self_attn.o_proj.weight'].numpy())

            write_float32(f, state[
                f'{prefix}.post_attention_layernorm.weight'].numpy())

            write_q4_0(f, state[f'{prefix}.mlp.gate_proj.weight'].numpy())
            write_q4_0(f, state[f'{prefix}.mlp.up_proj.weight'].numpy())
            write_q4_0(f, state[f'{prefix}.mlp.down_proj.weight'].numpy())

            print(f"  Layer {l + 1}/{n_layers}", end='\r')

        print(f"  Layer {n_layers}/{n_layers}")

        # Final norm
        write_float32(f, state['model.norm.weight'].numpy())

        # Classifier (tied embeddings, Q4_0)
        print(f"  Classifier ({vocab_size} x {dim})...")
        write_q4_0(f, state[cls_key].numpy())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export Qwen3 model with Q4_0 quantization')
    parser.add_argument('--output', '-o', default='qwen3_4b_q4.bin',
                        help='Output file path')
    parser.add_argument('--model-name', default='Qwen/Qwen3-4B-Thinking-2507',
                        help='HuggingFace model name')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length for KV cache')
    args = parser.parse_args()

    export_qwen3_q4(args.output, args.model_name, args.max_seq_len)
