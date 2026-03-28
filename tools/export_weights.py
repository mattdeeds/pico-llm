#!/usr/bin/env python3
"""Export a PyTorch transformer model to pico-llm binary format.

The binary format is designed for sequential streaming from SD card:

Header:
    Config struct: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len (7 x int32)
    Vocab data: for each token, uint16 length + utf-8 bytes

Weights (in exact consumption order per forward pass):
    For each layer:
        attn_norm:  float32[dim]
        wq:         int8[dim * dim]         + float32 scale
        wk:         int8[kv_dim * dim]      + float32 scale
        wv:         int8[kv_dim * dim]      + float32 scale
        wo:         int8[dim * dim]         + float32 scale
        ffn_norm:   float32[dim]
        w_gate:     int8[hidden_dim * dim]  + float32 scale
        w_up:       int8[hidden_dim * dim]  + float32 scale
        w_down:     int8[dim * hidden_dim]  + float32 scale
    final_norm: float32[dim]
    wcls:       int8[vocab_size * dim]      + float32 scale
    token_emb:  float32[vocab_size * dim]
"""

import struct
import numpy as np
import argparse


def quantize_int8(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization."""
    absmax = np.max(np.abs(tensor))
    if absmax == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    scale = absmax / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def write_config(f, config: dict):
    """Write model config as 7 int32 values."""
    f.write(struct.pack('<7i',
        config['dim'],
        config['hidden_dim'],
        config['n_layers'],
        config['n_heads'],
        config['n_kv_heads'],
        config['vocab_size'],
        config['max_seq_len'],
    ))


def write_float32(f, arr: np.ndarray):
    """Write a float32 array."""
    f.write(arr.astype(np.float32).tobytes())


def write_int8_with_scale(f, arr: np.ndarray):
    """Quantize to int8 and write with scale factor."""
    q, scale = quantize_int8(arr)
    f.write(q.tobytes())
    f.write(struct.pack('<f', scale))


def export_random_model(output_path: str, config: dict):
    """Export a random model for testing purposes."""
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    kv_dim = dim // config['n_heads'] * config['n_kv_heads']
    vocab_size = config['vocab_size']

    rng = np.random.default_rng(42)

    with open(output_path, 'wb') as f:
        write_config(f, config)

        # Placeholder vocab (single byte tokens for testing)
        for i in range(vocab_size):
            token = f"<{i}>".encode('utf-8')
            f.write(struct.pack('<H', len(token)))
            f.write(token)

        # Per-layer weights in consumption order
        for layer in range(n_layers):
            print(f"  Layer {layer}/{n_layers}")
            write_float32(f, rng.standard_normal(dim).astype(np.float32))           # attn_norm
            write_int8_with_scale(f, rng.standard_normal((dim, dim)))               # wq
            write_int8_with_scale(f, rng.standard_normal((kv_dim, dim)))            # wk
            write_int8_with_scale(f, rng.standard_normal((kv_dim, dim)))            # wv
            write_int8_with_scale(f, rng.standard_normal((dim, dim)))               # wo
            write_float32(f, rng.standard_normal(dim).astype(np.float32))           # ffn_norm
            write_int8_with_scale(f, rng.standard_normal((hidden_dim, dim)))        # w_gate
            write_int8_with_scale(f, rng.standard_normal((hidden_dim, dim)))        # w_up
            write_int8_with_scale(f, rng.standard_normal((dim, hidden_dim)))        # w_down

        # Final norm + classifier
        write_float32(f, rng.standard_normal(dim).astype(np.float32))               # final_norm
        write_int8_with_scale(f, rng.standard_normal((vocab_size, dim)))            # wcls
        write_float32(f, rng.standard_normal((vocab_size, dim)).astype(np.float32)) # token_emb

    print(f"Exported to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model weights for pico-llm')
    parser.add_argument('--output', '-o', default='model.bin', help='Output file path')
    parser.add_argument('--random', action='store_true', help='Generate random test model')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=704)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-kv-heads', type=int, default=4)
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--max-seq-len', type=int, default=512)
    args = parser.parse_args()

    config = {
        'dim': args.dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'n_kv_heads': args.n_kv_heads,
        'vocab_size': args.vocab_size,
        'max_seq_len': args.max_seq_len,
    }

    if args.random:
        export_random_model(args.output, config)
    else:
        print("Use --random to generate a test model, or implement PyTorch export.")
