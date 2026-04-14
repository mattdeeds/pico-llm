#!/usr/bin/env python3
"""Verify an exported Qwen3 pico-llm V2 binary against HuggingFace model output.

Reads the binary file, dequantizes weights, runs a forward pass matching the
Qwen3 architecture (QK-Norm, GQA, decoupled head_dim, RoPE theta=1M), and
compares logits against the HuggingFace transformers model.

Memory note: loads ~2.5 GB of dequantized weights from binary plus the HF
model (~2.4 GB). Requires ~5 GB RAM on the host machine.
"""

import argparse
import struct
import math
import numpy as np


# ---------------------------------------------------------------------------
# Binary reader (mirrors firmware sequential consumption)
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
        arr = np.frombuffer(self.data, dtype=np.float32, count=n,
                            offset=self.pos)
        self.pos += 4 * n
        return arr.copy()

    def read_int8_with_scale(self, shape):
        """Read int8 weights + float32 scale, return dequantized float32."""
        n = 1
        for s in shape:
            n *= s
        raw = np.frombuffer(self.data, dtype=np.int8, count=n,
                            offset=self.pos).reshape(shape).astype(np.float32)
        self.pos += n
        scale = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return raw * scale


# ---------------------------------------------------------------------------
# Forward pass ops (Qwen3 architecture)
# ---------------------------------------------------------------------------

ROPE_THETA = 1_000_000.0
RMSNORM_EPS = 1e-6


def rms_norm(x, weight, eps=RMSNORM_EPS):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def apply_qk_norm(q, k, q_norm_w, k_norm_w, n_heads, n_kv_heads, head_dim):
    """Apply per-head RMSNorm to Q and K vectors."""
    q = q.copy()
    k = k.copy()
    for h in range(n_heads):
        s = h * head_dim
        q[s:s + head_dim] = rms_norm(q[s:s + head_dim], q_norm_w)
    for h in range(n_kv_heads):
        s = h * head_dim
        k[s:s + head_dim] = rms_norm(k[s:s + head_dim], k_norm_w)
    return q, k


def apply_rope(q, k, n_heads, n_kv_heads, head_dim, pos):
    """Apply RoPE to Q [q_dim] and K [kv_dim].

    Uses the HF/Qwen3 split-half convention: pairs element i with element
    i + head_dim/2 within each head (not consecutive pairs).
    """
    q = q.copy()
    k = k.copy()
    half = head_dim // 2

    # Precompute frequencies for head_dim/2 pairs
    inv_freq = 1.0 / (ROPE_THETA ** (
        np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = pos * inv_freq
    cos_vals = np.cos(angles).astype(np.float32)
    sin_vals = np.sin(angles).astype(np.float32)

    for h in range(n_heads):
        s = h * head_dim
        q1 = q[s:s + half].copy()
        q2 = q[s + half:s + head_dim].copy()
        q[s:s + half] = q1 * cos_vals - q2 * sin_vals
        q[s + half:s + head_dim] = q2 * cos_vals + q1 * sin_vals

    for h in range(n_kv_heads):
        s = h * head_dim
        k1 = k[s:s + half].copy()
        k2 = k[s + half:s + head_dim].copy()
        k[s:s + half] = k1 * cos_vals - k2 * sin_vals
        k[s + half:s + head_dim] = k2 * cos_vals + k1 * sin_vals

    return q, k


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_pass(binary_path, input_ids):
    """Run a full Qwen3 forward pass on the exported V2 binary."""
    r = BinaryReader(binary_path)

    # Read V2 config (8 fields)
    dim = r.read_int32()
    hidden_dim = r.read_int32()
    n_layers = r.read_int32()
    n_heads = r.read_int32()
    n_kv_heads = r.read_int32()
    vocab_size = r.read_int32()
    max_seq_len = r.read_int32()
    head_dim = r.read_int32()

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"Config: dim={dim}, hidden={hidden_dim}, layers={n_layers}, "
          f"heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}, "
          f"head_dim={head_dim}")
    print(f"  q_dim={q_dim}, kv_dim={kv_dim}")

    # Skip vocab section
    for _ in range(vocab_size):
        length = r.read_uint16()
        r.pos += length

    # Read all layer weights into memory
    print("Loading layer weights...")
    layers = []
    for l in range(n_layers):
        layer = {}
        layer["attn_norm"] = r.read_float32_array(dim)
        layer["q_norm"] = r.read_float32_array(head_dim)
        layer["k_norm"] = r.read_float32_array(head_dim)
        layer["wq"] = r.read_int8_with_scale((q_dim, dim))
        layer["wk"] = r.read_int8_with_scale((kv_dim, dim))
        layer["wv"] = r.read_int8_with_scale((kv_dim, dim))
        layer["wo"] = r.read_int8_with_scale((dim, q_dim))
        layer["ffn_norm"] = r.read_float32_array(dim)
        layer["w_gate"] = r.read_int8_with_scale((hidden_dim, dim))
        layer["w_up"] = r.read_int8_with_scale((hidden_dim, dim))
        layer["w_down"] = r.read_int8_with_scale((dim, hidden_dim))
        layers.append(layer)
        print(f"  Layer {l + 1}/{n_layers}", end='\r')
    print(f"  Layer {n_layers}/{n_layers}")

    final_norm = r.read_float32_array(dim)
    print("Loading classifier weights...")
    wcls = r.read_int8_with_scale((vocab_size, dim))

    print(f"Loaded all weights ({r.pos} bytes, {r.pos / 1024**2:.1f} MB)")

    # Tied embeddings: use dequantized wcls rows as token embeddings
    token_emb = wcls

    # Run autoregressive forward pass
    seq_len = len(input_ids)
    all_logits = np.zeros((seq_len, vocab_size), dtype=np.float32)

    k_cache = np.zeros((n_layers, max_seq_len, kv_dim), dtype=np.float32)
    v_cache = np.zeros((n_layers, max_seq_len, kv_dim), dtype=np.float32)

    for pos in range(seq_len):
        token = input_ids[pos]
        x = token_emb[token].copy()

        for l in range(n_layers):
            ly = layers[l]

            # Attention
            xn = rms_norm(x, ly["attn_norm"])
            q = ly["wq"] @ xn
            k = ly["wk"] @ xn
            v = ly["wv"] @ xn

            # QK-Norm (per-head RMSNorm)
            q, k = apply_qk_norm(q, k, ly["q_norm"], ly["k_norm"],
                                 n_heads, n_kv_heads, head_dim)

            # RoPE
            q, k = apply_rope(q, k, n_heads, n_kv_heads, head_dim, pos)

            # Cache
            k_cache[l, pos] = k
            v_cache[l, pos] = v

            # Multi-head attention with GQA
            out = np.zeros(q_dim, dtype=np.float32)
            for h in range(n_heads):
                kv_h = h * n_kv_heads // n_heads
                q_h = q[h * head_dim:(h + 1) * head_dim]

                scores = np.zeros(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    k_h = k_cache[l, t,
                                  kv_h * head_dim:(kv_h + 1) * head_dim]
                    scores[t] = np.dot(q_h, k_h) / math.sqrt(head_dim)

                attn = softmax(scores)

                head_out = np.zeros(head_dim, dtype=np.float32)
                for t in range(pos + 1):
                    v_h = v_cache[l, t,
                                  kv_h * head_dim:(kv_h + 1) * head_dim]
                    head_out += attn[t] * v_h

                out[h * head_dim:(h + 1) * head_dim] = head_out

            xb2 = ly["wo"] @ out
            x = x + xb2

            # FFN
            xn = rms_norm(x, ly["ffn_norm"])
            gate = ly["w_gate"] @ xn
            up = ly["w_up"] @ xn
            hb = silu(gate) * up
            x = x + ly["w_down"] @ hb

        x = rms_norm(x, final_norm)
        logits = wcls @ x
        all_logits[pos] = logits

        print(f"  Position {pos + 1}/{seq_len}", end='\r')

    print()
    return all_logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate exported Qwen3 pico-llm V2 binary "
                    "against HuggingFace model")
    parser.add_argument("binary", help="Path to exported .bin file")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model for reference logits")
    parser.add_argument("--prompt", default="The quick brown fox",
                        help="Test prompt")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load HF model
    print(f"Loading HF model {args.model_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hf_model.eval()

    # Tokenize
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"Prompt: {args.prompt!r}")
    print(f"Token IDs ({len(input_ids)}): {input_ids}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in input_ids]}")

    # HF reference logits
    print("\nRunning HF reference forward pass...")
    with torch.no_grad():
        hf_out = hf_model(torch.tensor([input_ids]))
        ref_logits = hf_out.logits[0].numpy()

    print(f"Reference logits shape: {ref_logits.shape}")

    # Our forward pass on the exported binary
    print(f"\nRunning NumPy forward pass on {args.binary}...")
    test_logits = forward_pass(args.binary, input_ids)

    # --- Comparison ---
    print("\n=== Results ===")

    max_err = np.max(np.abs(test_logits - ref_logits))
    mean_err = np.mean(np.abs(test_logits - ref_logits))
    print(f"Max absolute error:  {max_err:.4f}")
    print(f"Mean absolute error: {mean_err:.4f}")

    # Argmax agreement
    ref_preds = np.argmax(ref_logits, axis=-1)
    test_preds = np.argmax(test_logits, axis=-1)
    agree = np.sum(ref_preds == test_preds)
    total = len(ref_preds)
    print(f"Argmax agreement:    {agree}/{total} "
          f"({100 * agree / total:.1f}%)")

    # Top-5 overlap
    top5_sum = 0.0
    for i in range(total):
        ref_top5 = set(np.argsort(ref_logits[i])[-5:])
        test_top5 = set(np.argsort(test_logits[i])[-5:])
        top5_sum += len(ref_top5 & test_top5) / 5.0
    print(f"Top-5 overlap:       {100 * top5_sum / total:.1f}%")

    # Per-position detail
    print(f"\nPer-position predictions:")
    for i in range(total):
        match = "  OK" if ref_preds[i] == test_preds[i] else "MISS"
        ref_tok = tokenizer.decode([ref_preds[i]])
        test_tok = tokenizer.decode([test_preds[i]])
        print(f"  pos {i}: ref={ref_preds[i]:6d} ({ref_tok!r:20s})  "
              f"test={test_preds[i]:6d} ({test_tok!r:20s})  {match}")

    # Verdict (thresholds tuned for 600M param model with per-tensor int8)
    if agree == total:
        print("\nPASS: Perfect argmax agreement.")
    elif agree / total >= 0.75 and max_err < 15.0:
        print("\nPASS: Argmax agreement and error within expected range "
              "for per-tensor int8 quantization of a large model.")
    elif agree / total >= 0.50:
        print("\nWARNING: Moderate agreement -- per-channel quantization "
              "may be needed for this model.")
    else:
        print("\nFAIL: Low agreement -- likely a bug in export or "
              "forward pass.")


if __name__ == "__main__":
    main()
