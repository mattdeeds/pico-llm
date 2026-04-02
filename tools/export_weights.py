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
import json


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


def load_tokenizer_vocab(tokenizer_path: str, vocab_size: int) -> list[bytes]:
    """Load vocabulary from a HuggingFace tokenizers JSON file."""
    with open(tokenizer_path) as f:
        tok_data = json.load(f)

    # Build id -> token string mapping
    vocab = tok_data["model"]["vocab"]
    added = {t["id"]: t["content"] for t in tok_data.get("added_tokens", [])}

    # Invert vocab dict (string -> id) to (id -> string)
    id_to_str = {v: k for k, v in vocab.items()}
    id_to_str.update(added)

    result = []
    for i in range(vocab_size):
        token_str = id_to_str.get(i, f"<{i}>")
        result.append(token_str.encode("utf-8"))
    return result


def export_pytorch_model(output_path: str, model_path: str,
                         tokenizer_path: str):
    """Export a trained PyTorch model to pico-llm binary format."""
    import torch

    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    margs = ckpt["args"]

    config = {
        'dim': margs['dim'],
        'hidden_dim': margs['hidden_dim'],
        'n_layers': margs['n_layers'],
        'n_heads': margs['n_heads'],
        'n_kv_heads': margs['n_kv_heads'],
        'vocab_size': margs['vocab_size'],
        'max_seq_len': margs['max_seq_len'],
    }
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    kv_dim = dim // config['n_heads'] * config['n_kv_heads']
    vocab_size = config['vocab_size']

    print(f"  dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
          f"heads={config['n_heads']}, kv_heads={config['n_kv_heads']}, "
          f"vocab={vocab_size}")

    # Load vocabulary
    print(f"Loading tokenizer from {tokenizer_path}...")
    vocab_tokens = load_tokenizer_vocab(tokenizer_path, vocab_size)

    with open(output_path, 'wb') as f:
        write_config(f, config)

        # Write vocabulary
        for token_bytes in vocab_tokens:
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)

        # Per-layer weights in consumption order
        for l in range(n_layers):
            print(f"  Layer {l}/{n_layers}")
            prefix = f"layers.{l}"

            # attn_norm (float32)
            w = state[f"{prefix}.attn_norm.weight"].numpy()
            write_float32(f, w)

            # wq, wk, wv, wo (int8 + scale)
            for proj in ["wq", "wk", "wv", "wo"]:
                w = state[f"{prefix}.attention.{proj}.weight"].numpy()
                write_int8_with_scale(f, w)

            # ffn_norm (float32)
            w = state[f"{prefix}.ffn_norm.weight"].numpy()
            write_float32(f, w)

            # w_gate, w_up, w_down (int8 + scale)
            for proj in ["w_gate", "w_up", "w_down"]:
                w = state[f"{prefix}.ffn.{proj}.weight"].numpy()
                write_int8_with_scale(f, w)

        # Final norm (float32)
        write_float32(f, state["final_norm.weight"].numpy())

        # Classifier (int8 + scale)
        write_int8_with_scale(f, state["classifier.weight"].numpy())

        # Token embedding (float32)
        write_float32(f, state["token_embedding.weight"].numpy())

    print(f"Exported to {output_path}")
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model weights for pico-llm')
    parser.add_argument('--output', '-o', default='model.bin', help='Output file path')
    parser.add_argument('--random', action='store_true', help='Generate random test model')
    parser.add_argument('--model', default=None, help='Path to trained model.pt')
    parser.add_argument('--tokenizer', default=None, help='Path to tokenizer.json')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=704)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-kv-heads', type=int, default=4)
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--max-seq-len', type=int, default=512)
    args = parser.parse_args()

    if args.model:
        if not args.tokenizer:
            parser.error("--tokenizer is required when using --model")
        export_pytorch_model(args.output, args.model, args.tokenizer)
    elif args.random:
        config = {
            'dim': args.dim, 'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers, 'n_heads': args.n_heads,
            'n_kv_heads': args.n_kv_heads, 'vocab_size': args.vocab_size,
            'max_seq_len': args.max_seq_len,
        }
        export_random_model(args.output, config)
    else:
        print("Use --model MODEL --tokenizer TOKENIZER to export a trained model,")
        print("or --random to generate a test model.")
