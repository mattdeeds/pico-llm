#!/usr/bin/env python3
"""Export Bonsai-1.7B (1-bit Qwen3) from GGUF to pico-llm V2 binary.

Source: prism-ml/Bonsai-1.7B-gguf (native 1-bit, Q1_0_g128 = GGML dtype 41).

Q1_0_g128 block format (18 bytes per 128 weights):
  bytes 0..1:   float16 block scale d
  bytes 2..17:  16 sign bytes, bit k of byte j → weight at (j*8 + k)
                bit = 1 → +d,  bit = 0 → -d

Since GGUF already stores rows row-major with each row = (cols/128) blocks of
18 bytes, we copy the raw Q1 bytes verbatim — no requantization.

Output layout (matches export_qwen3_q4.py's V2 format; only block format
differs — 128-weight / 18-byte blocks instead of 32-weight / 18-byte):
  Header: 8 × int32
  Vocab:  per token uint16 len + UTF-8 bytes
  Per layer: attn_norm, q_norm, k_norm (F32) → wq, wk, wv, wo (Q1)
             → ffn_norm (F32) → w_gate, w_up, w_down (Q1)
  final_norm (F32)
  wcls (Q1) — Bonsai-1.7B has tie_word_embeddings=true, so wcls = token_embd.
"""

import argparse
import os
import struct
import sys
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# Minimal GGUF reader (spec v3)
# -----------------------------------------------------------------------------

GGUF_T_UINT8, GGUF_T_INT8 = 0, 1
GGUF_T_UINT16, GGUF_T_INT16 = 2, 3
GGUF_T_UINT32, GGUF_T_INT32 = 4, 5
GGUF_T_FLOAT32 = 6
GGUF_T_BOOL = 7
GGUF_T_STRING = 8
GGUF_T_ARRAY = 9
GGUF_T_UINT64, GGUF_T_INT64 = 10, 11
GGUF_T_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_Q1_0_G128 = 41


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_infos()

    def close(self):
        self.f.close()

    def _u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def _u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def _i32(self): return struct.unpack("<i", self.f.read(4))[0]
    def _i64(self): return struct.unpack("<q", self.f.read(8))[0]
    def _f32(self): return struct.unpack("<f", self.f.read(4))[0]
    def _f64(self): return struct.unpack("<d", self.f.read(8))[0]
    def _u8(self): return self.f.read(1)[0]
    def _i8(self): return struct.unpack("<b", self.f.read(1))[0]
    def _u16(self): return struct.unpack("<H", self.f.read(2))[0]
    def _i16(self): return struct.unpack("<h", self.f.read(2))[0]
    def _bool(self): return bool(self._u8())

    def _str(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def _value(self, t):
        if t == GGUF_T_UINT8: return self._u8()
        if t == GGUF_T_INT8: return self._i8()
        if t == GGUF_T_UINT16: return self._u16()
        if t == GGUF_T_INT16: return self._i16()
        if t == GGUF_T_UINT32: return self._u32()
        if t == GGUF_T_INT32: return self._i32()
        if t == GGUF_T_FLOAT32: return self._f32()
        if t == GGUF_T_BOOL: return self._bool()
        if t == GGUF_T_STRING: return self._str()
        if t == GGUF_T_UINT64: return self._u64()
        if t == GGUF_T_INT64: return self._i64()
        if t == GGUF_T_FLOAT64: return self._f64()
        if t == GGUF_T_ARRAY:
            inner = self._u32()
            n = self._u64()
            return [self._value(inner) for _ in range(n)]
        raise ValueError(f"Unknown GGUF value type: {t}")

    def _read_header(self):
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Bad GGUF magic: {magic!r}")
        self.version = self._u32()
        self.tensor_count = self._u64()
        self.metadata_kv_count = self._u64()

    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.metadata_kv_count):
            key = self._str()
            t = self._u32()
            self.metadata[key] = self._value(t)

    def _read_tensor_infos(self):
        self.tensors = {}
        for _ in range(self.tensor_count):
            name = self._str()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            dtype = self._u32()
            offset = self._u64()
            self.tensors[name] = {"name": name, "dims": dims,
                                  "dtype": dtype, "offset": offset}
        align = self.metadata.get("general.alignment", 32)
        pos = self.f.tell()
        pad = (align - (pos % align)) % align
        self.data_start = pos + pad

    def read_tensor_bytes(self, name, expected_nbytes=None):
        t = self.tensors[name]
        self.f.seek(self.data_start + t["offset"])
        if expected_nbytes is None:
            # Infer from dtype+dims
            if t["dtype"] == GGML_TYPE_F32:
                nelts = 1
                for d in t["dims"]:
                    nelts *= d
                expected_nbytes = nelts * 4
            elif t["dtype"] == GGML_TYPE_Q1_0_G128:
                cols, rows = t["dims"][0], t["dims"][1] if len(t["dims"]) > 1 else 1
                assert cols % 128 == 0, f"{name}: cols={cols} not div 128"
                expected_nbytes = rows * (cols // 128) * 18
            else:
                raise ValueError(f"{name}: unsupported dtype {t['dtype']}")
        return self.f.read(expected_nbytes)


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

def encode_gguf_token(tok: str, idx: int) -> bytes:
    """Encode a GGUF tokenizer token (str) to bytes for our V2 vocab section.

    GGUF stores tokens as Python-style strings, already containing the GPT-2
    byte-level encoding (Ġ for leading space, Ċ for newline, etc.). We just
    encode them as UTF-8. Empty or None tokens get a placeholder.
    """
    if not tok:
        return f"<{idx}>".encode("utf-8")
    return tok.encode("utf-8")


def export_bonsai(output_path: str, max_seq_len: int):
    repo = "prism-ml/Bonsai-1.7B-gguf"
    fname = "Bonsai-1.7B.gguf"
    print(f"Downloading {repo}/{fname} (cached after first run)...")
    gguf_path = hf_hub_download(repo_id=repo, filename=fname)
    print(f"  path: {gguf_path}  size: {os.path.getsize(gguf_path)/1e6:.1f} MB\n")

    r = GGUFReader(gguf_path)
    md = r.metadata

    # ---- Pull architecture constants from GGUF metadata ----
    dim = md["qwen3.embedding_length"]
    hidden_dim = md["qwen3.feed_forward_length"]
    n_layers = md["qwen3.block_count"]
    n_heads = md["qwen3.attention.head_count"]
    n_kv_heads = md["qwen3.attention.head_count_kv"]
    head_dim = md["qwen3.attention.key_length"]
    vocab_size = len(md["tokenizer.ggml.tokens"])
    rope_theta = md["qwen3.rope.freq_base"]
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print("=== Architecture (from GGUF metadata) ===")
    print(f"  dim={dim} hidden={hidden_dim} layers={n_layers}")
    print(f"  heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}")
    print(f"  q_dim={q_dim} kv_dim={kv_dim}")
    print(f"  vocab={vocab_size}  rope_theta={rope_theta}")
    print()

    # Check dims divisible by 128 (Q1_0_g128 block requirement)
    for name, val in [("dim", dim), ("q_dim", q_dim), ("kv_dim", kv_dim),
                      ("hidden_dim", hidden_dim)]:
        assert val % 128 == 0, f"{name}={val} must be multiple of 128 for Q1_0_g128"

    # Check expected tensors are present
    required = [
        "output_norm.weight", "token_embd.weight",
    ]
    for l in range(n_layers):
        for suf in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
            required.append(f"blk.{l}.{suf}.weight")
        for suf in ("attn_q", "attn_k", "attn_v", "attn_output",
                    "ffn_gate", "ffn_up", "ffn_down"):
            required.append(f"blk.{l}.{suf}.weight")
    missing = [n for n in required if n not in r.tensors]
    if missing:
        raise SystemExit(f"Missing tensors in GGUF: {missing[:5]}...")

    # ---- Vocab ----
    tokens = md["tokenizer.ggml.tokens"]
    print(f"Preparing vocab ({len(tokens)} tokens)...")
    vocab_bytes = [encode_gguf_token(t, i) for i, t in enumerate(tokens)]

    # ---- Compute expected sizes ----
    q1_row = lambda cols: (cols // 128) * 18
    layer_bytes = (
        dim * 4 + head_dim * 4 * 2           # attn_norm, q_norm, k_norm
        + q_dim * q1_row(dim)                # wq
        + kv_dim * q1_row(dim) * 2           # wk, wv
        + dim * q1_row(q_dim)                # wo
        + dim * 4                            # ffn_norm
        + hidden_dim * q1_row(dim) * 2       # w_gate, w_up
        + dim * q1_row(hidden_dim)           # w_down
    )
    print(f"  Per-layer bytes: {layer_bytes/1024/1024:.1f} MB")
    print(f"  Total layers:    {n_layers*layer_bytes/1024/1024:.0f} MB")
    print(f"  Classifier:      {vocab_size*q1_row(dim)/1024/1024:.0f} MB")
    print()

    # ---- Helper: read a Q1 tensor and verify its size matches expectation ----
    def q1_tensor(name, rows, cols):
        t = r.tensors[name]
        assert t["dtype"] == GGML_TYPE_Q1_0_G128, \
            f"{name}: dtype {t['dtype']}, expected 41"
        # GGUF dims are [cols, rows] (inner first)
        g_cols, g_rows = t["dims"][0], t["dims"][1]
        assert g_cols == cols and g_rows == rows, \
            f"{name}: GGUF dims [{g_cols}, {g_rows}] != expected [{cols}, {rows}]"
        nbytes = rows * (cols // 128) * 18
        return r.read_tensor_bytes(name, nbytes)

    def f32_tensor(name, nelts):
        t = r.tensors[name]
        assert t["dtype"] == GGML_TYPE_F32, \
            f"{name}: dtype {t['dtype']}, expected F32"
        total = 1
        for d in t["dims"]:
            total *= d
        assert total == nelts, f"{name}: {total} elts, expected {nelts}"
        return r.read_tensor_bytes(name, nelts * 4)

    # ---- Write output ----
    print(f"Writing to {output_path}...")
    with open(output_path, "wb") as f:
        # Header: 8 × int32
        f.write(struct.pack("<8i",
            dim, hidden_dim, n_layers, n_heads, n_kv_heads,
            vocab_size, max_seq_len, head_dim))

        # Vocab
        for tb in vocab_bytes:
            f.write(struct.pack("<H", len(tb)))
            f.write(tb)
        vocab_end = f.tell()
        print(f"  Vocab section: {vocab_end/1024/1024:.1f} MB")

        # Per-layer weights
        for l in range(n_layers):
            p = f"blk.{l}"
            f.write(f32_tensor(f"{p}.attn_norm.weight", dim))
            f.write(f32_tensor(f"{p}.attn_q_norm.weight", head_dim))
            f.write(f32_tensor(f"{p}.attn_k_norm.weight", head_dim))

            f.write(q1_tensor(f"{p}.attn_q.weight",      q_dim, dim))
            f.write(q1_tensor(f"{p}.attn_k.weight",      kv_dim, dim))
            f.write(q1_tensor(f"{p}.attn_v.weight",      kv_dim, dim))
            f.write(q1_tensor(f"{p}.attn_output.weight", dim,    q_dim))

            f.write(f32_tensor(f"{p}.ffn_norm.weight", dim))

            f.write(q1_tensor(f"{p}.ffn_gate.weight", hidden_dim, dim))
            f.write(q1_tensor(f"{p}.ffn_up.weight",   hidden_dim, dim))
            f.write(q1_tensor(f"{p}.ffn_down.weight", dim,        hidden_dim))

            print(f"  Layer {l+1}/{n_layers}", end="\r")
        print(f"  Layer {n_layers}/{n_layers}")

        # Final norm + classifier (tied: token_embd == wcls)
        f.write(f32_tensor("output_norm.weight", dim))
        print(f"  Classifier (tied, {vocab_size}×{dim})...")
        f.write(q1_tensor("token_embd.weight", vocab_size, dim))

    r.close()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export Bonsai-1.7B GGUF to pico-llm V2 binary (Q1_0_g128).")
    ap.add_argument("--output", "-o", default="bonsai_1p7b_q1.bin")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    export_bonsai(args.output, args.max_seq_len)
