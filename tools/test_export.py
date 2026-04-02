#!/usr/bin/env python3
"""Verify an exported pico-llm binary by reimplementing the C forward pass in numpy.

Reads the binary file sequentially (same as the C code), dequantizes weights,
runs a forward pass, and compares logits against saved reference.
"""

import argparse
import struct
import math

import numpy as np


# ---------------------------------------------------------------------------
# Binary reader (mirrors C sequential consumption)
# ---------------------------------------------------------------------------

class BinaryReader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        self.pos = 0

    def read_int32(self, n=1):
        vals = struct.unpack_from(f"<{n}i", self.data, self.pos)
        self.pos += 4 * n
        return vals[0] if n == 1 else vals

    def read_uint16(self):
        val = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return val

    def read_bytes(self, n):
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def read_float32_array(self, n):
        arr = np.frombuffer(self.data, dtype=np.float32, count=n, offset=self.pos)
        self.pos += 4 * n
        return arr.copy()

    def read_int8_array(self, n):
        arr = np.frombuffer(self.data, dtype=np.int8, count=n, offset=self.pos)
        self.pos += n
        return arr.copy()

    def read_int8_with_scale(self, shape):
        """Read int8 weights + float32 scale, return dequantized float32."""
        n = 1
        for s in shape:
            n *= s
        raw = self.read_int8_array(n).reshape(shape).astype(np.float32)
        scale = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return raw * scale


# ---------------------------------------------------------------------------
# Forward pass ops (matching transformer.c)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def rope(q, k, dim, head_size, pos):
    """Apply RoPE matching transformer.c:43-63."""
    q = q.copy()
    k = k.copy()
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1.0 / (10000.0 ** (float(head_dim) / float(head_size)))
        val = pos * freq
        cos_val = math.cos(val)
        sin_val = math.sin(val)

        q0, q1 = q[i], q[i + 1]
        q[i] = q0 * cos_val - q1 * sin_val
        q[i + 1] = q0 * sin_val + q1 * cos_val

        if k is not None and i < len(k):
            k0, k1 = k[i], k[i + 1]
            k[i] = k0 * cos_val - k1 * sin_val
            k[i + 1] = k0 * sin_val + k1 * cos_val
    return q, k


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def forward_pass(binary_path, input_ids):
    """Run a full forward pass on the exported binary, return logits."""
    r = BinaryReader(binary_path)

    # Read config
    dim = r.read_int32()
    hidden_dim = r.read_int32()
    n_layers = r.read_int32()
    n_heads = r.read_int32()
    n_kv_heads = r.read_int32()
    vocab_size = r.read_int32()
    max_seq_len = r.read_int32()

    head_size = dim // n_heads
    kv_dim = head_size * n_kv_heads

    print(f"Config: dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
          f"heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}")

    # Read vocab (skip over it)
    vocab = []
    for _ in range(vocab_size):
        length = r.read_uint16()
        token = r.read_bytes(length).decode("utf-8", errors="replace")
        vocab.append(token)

    # Record position after header — we need to re-read from here for each token
    weights_start = r.pos

    # Read all weights into memory (unlike C which streams)
    layers = []
    for l in range(n_layers):
        layer = {}
        layer["attn_norm"] = r.read_float32_array(dim)
        layer["wq"] = r.read_int8_with_scale((dim, dim))
        layer["wk"] = r.read_int8_with_scale((kv_dim, dim))
        layer["wv"] = r.read_int8_with_scale((kv_dim, dim))
        layer["wo"] = r.read_int8_with_scale((dim, dim))
        layer["ffn_norm"] = r.read_float32_array(dim)
        layer["w_gate"] = r.read_int8_with_scale((hidden_dim, dim))
        layer["w_up"] = r.read_int8_with_scale((hidden_dim, dim))
        layer["w_down"] = r.read_int8_with_scale((dim, hidden_dim))
        layers.append(layer)

    final_norm = r.read_float32_array(dim)
    wcls = r.read_int8_with_scale((vocab_size, dim))
    token_emb = r.read_float32_array(vocab_size * dim).reshape(vocab_size, dim)

    print(f"Loaded all weights ({r.pos} bytes read)")

    # Run autoregressive forward pass
    seq_len = len(input_ids)
    all_logits = np.zeros((seq_len, vocab_size), dtype=np.float32)

    # KV cache (in memory for this test, unlike on-device SD)
    k_cache = np.zeros((n_layers, max_seq_len, kv_dim), dtype=np.float32)
    v_cache = np.zeros((n_layers, max_seq_len, kv_dim), dtype=np.float32)

    for pos in range(seq_len):
        token = input_ids[pos]
        x = token_emb[token].copy()

        for l in range(n_layers):
            layer = layers[l]

            # Attention
            xn = rms_norm(x, layer["attn_norm"])
            q = layer["wq"] @ xn
            k = layer["wk"] @ xn
            v = layer["wv"] @ xn

            q, k = rope(q, k, kv_dim, head_size, pos)

            k_cache[l, pos] = k
            v_cache[l, pos] = v

            # Multi-head attention
            out = np.zeros(dim, dtype=np.float32)
            for h in range(n_heads):
                kv_h = h * n_kv_heads // n_heads  # GQA head mapping
                q_h = q[h * head_size:(h + 1) * head_size]

                scores = np.zeros(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    k_h = k_cache[l, t, kv_h * head_size:(kv_h + 1) * head_size]
                    scores[t] = np.dot(q_h, k_h) / math.sqrt(head_size)

                attn = softmax(scores)

                head_out = np.zeros(head_size, dtype=np.float32)
                for t in range(pos + 1):
                    v_h = v_cache[l, t, kv_h * head_size:(kv_h + 1) * head_size]
                    head_out += attn[t] * v_h

                out[h * head_size:(h + 1) * head_size] = head_out

            xb2 = layer["wo"] @ out
            x = x + xb2

            # FFN
            xn = rms_norm(x, layer["ffn_norm"])
            gate = layer["w_gate"] @ xn
            up = layer["w_up"] @ xn
            hb = silu(gate) * up
            x = x + layer["w_down"] @ hb

        x = rms_norm(x, final_norm)
        logits = wcls @ x
        all_logits[pos] = logits

        if pos % 4 == 0:
            print(f"  Position {pos}/{seq_len}", end="\r")

    print()
    return all_logits


def main():
    parser = argparse.ArgumentParser(description="Test exported pico-llm binary")
    parser.add_argument("binary", help="Path to exported .bin file")
    parser.add_argument("--reference", default="models/reference_logits.npz",
                        help="Reference logits from validate_model.py")
    args = parser.parse_args()

    # Load reference
    ref = np.load(args.reference)
    input_ids = ref["input_ids"].tolist()
    ref_logits = ref["logits"][0]  # [seq_len, vocab]

    print(f"Input tokens: {input_ids}")
    print(f"Reference logits shape: {ref_logits.shape}")

    # Run forward pass on binary
    test_logits = forward_pass(args.binary, input_ids)

    # Compare
    max_err = np.max(np.abs(test_logits - ref_logits))
    mean_err = np.mean(np.abs(test_logits - ref_logits))
    print(f"\nMax absolute error:  {max_err:.6f}")
    print(f"Mean absolute error: {mean_err:.6f}")

    # Check argmax agreement (most important for generation)
    ref_preds = np.argmax(ref_logits, axis=-1)
    test_preds = np.argmax(test_logits, axis=-1)
    agree = np.sum(ref_preds == test_preds)
    print(f"Argmax agreement:    {agree}/{len(ref_preds)} "
          f"({100*agree/len(ref_preds):.1f}%)")

    if max_err < 1.0:
        print("\nPASS: Quantization error within expected range.")
    else:
        print("\nWARNING: Large error detected — check export correctness.")


if __name__ == "__main__":
    main()
