#!/usr/bin/env python3
"""Inspect the Bonsai-1.7B GGUF file: download, parse header, list tensors.

Confirms:
  - Q1_0_g128 dtype number (expected 41).
  - Block layout: 18 bytes per 128-weight block (2-byte fp16 scale + 16 sign bytes).
  - Tensor names present (per-layer projections, norms, wcls/embed).
  - Metadata keys (arch, RoPE, tokenizer).

Writes nothing — read-only inspection.
"""

import os
import struct
import sys
from huggingface_hub import hf_hub_download

REPO = "prism-ml/Bonsai-1.7B-gguf"
FILE = "Bonsai-1.7B.gguf"

# GGUF metadata value types (spec v3)
GGUF_T_UINT8, GGUF_T_INT8 = 0, 1
GGUF_T_UINT16, GGUF_T_INT16 = 2, 3
GGUF_T_UINT32, GGUF_T_INT32 = 4, 5
GGUF_T_FLOAT32 = 6
GGUF_T_BOOL = 7
GGUF_T_STRING = 8
GGUF_T_ARRAY = 9
GGUF_T_UINT64, GGUF_T_INT64 = 10, 11
GGUF_T_FLOAT64 = 12

# Known ggml tensor dtypes (standard llama.cpp set + Bonsai custom types)
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    # Custom PrismML/Bonsai:
    41: "Q1_0_g128",
}


class GGUFReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self._read_header()
        self._read_metadata()
        self._read_tensor_infos()

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
        assert magic == b"GGUF", f"Bad magic: {magic!r}"
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
        self.tensors = []
        for _ in range(self.tensor_count):
            name = self._str()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            dtype = self._u32()
            offset = self._u64()
            self.tensors.append({"name": name, "dims": dims,
                                 "dtype": dtype, "offset": offset})
        # tensor data starts aligned after current position
        align = self.metadata.get("general.alignment", 32)
        pos = self.f.tell()
        pad = (align - (pos % align)) % align
        self.data_start = pos + pad


def main():
    print(f"Downloading {REPO}/{FILE} (cached after first run)...")
    path = hf_hub_download(repo_id=REPO, filename=FILE)
    print(f"  path: {path}")
    print(f"  size: {os.path.getsize(path) / 1e6:.1f} MB\n")

    r = GGUFReader(path)
    print(f"GGUF v{r.version}  tensors={r.tensor_count}  metadata={r.metadata_kv_count}")
    print(f"Tensor data starts at offset {r.data_start}\n")

    # Interesting metadata keys
    print("=== Key metadata ===")
    keys_of_interest = [
        "general.architecture", "general.name", "general.quantization_version",
        "general.file_type", "general.alignment",
        "qwen3.block_count", "qwen3.context_length",
        "qwen3.embedding_length", "qwen3.feed_forward_length",
        "qwen3.attention.head_count", "qwen3.attention.head_count_kv",
        "qwen3.attention.key_length", "qwen3.attention.value_length",
        "qwen3.attention.layer_norm_rms_epsilon", "qwen3.rope.freq_base",
        "qwen3.rope.scaling.type", "qwen3.rope.scaling.factor",
        "tokenizer.ggml.model", "tokenizer.ggml.pre",
    ]
    for k in keys_of_interest:
        if k in r.metadata:
            v = r.metadata[k]
            if isinstance(v, list) and len(v) > 5:
                v = f"[list len={len(v)}, first={v[:3]}]"
            print(f"  {k}: {v}")
    print()

    # All other metadata keys (short form)
    other = [k for k in r.metadata if k not in keys_of_interest]
    print(f"Other metadata keys ({len(other)}):")
    for k in sorted(other):
        v = r.metadata[k]
        if isinstance(v, list):
            print(f"  {k}: <list len={len(v)}>")
        elif isinstance(v, str):
            print(f"  {k}: {v[:80]!r}")
        else:
            print(f"  {k}: {v}")
    print()

    # Tensor summary by dtype
    print("=== Tensor dtypes summary ===")
    dtype_counts = {}
    for t in r.tensors:
        dtype_counts.setdefault(t["dtype"], []).append(t)
    for dtype, ts in sorted(dtype_counts.items()):
        name = GGML_TYPE_NAMES.get(dtype, f"UNKNOWN({dtype})")
        total_bytes = 0
        print(f"  dtype {dtype} ({name}): {len(ts)} tensors")
    print()

    # Full tensor list
    print("=== All tensors ===")
    for t in r.tensors[:20]:
        name = GGML_TYPE_NAMES.get(t["dtype"], f"UNKNOWN({t['dtype']})")
        print(f"  {t['name']:<45} dims={t['dims']}  dtype={name}  off={t['offset']}")
    if len(r.tensors) > 20:
        print(f"  ... ({len(r.tensors) - 20} more)")
        # Print last 5 to see wcls/embed area
        for t in r.tensors[-5:]:
            name = GGML_TYPE_NAMES.get(t["dtype"], f"UNKNOWN({t['dtype']})")
            print(f"  {t['name']:<45} dims={t['dims']}  dtype={name}  off={t['offset']}")
    print()

    # Pick first Q1_0_g128 tensor and dump first block
    q1 = [t for t in r.tensors if t["dtype"] == 41]
    if not q1:
        print("WARNING: no dtype-41 tensors found")
    else:
        t = q1[0]
        rows, cols = t["dims"][1], t["dims"][0]  # GGUF uses column-major dims [n, m]
        # Actually GGUF tensor dims for a 2D matrix [rows, cols] in GGML layout are stored
        # as (cols, rows). Row-major: cols first. Clarify by printing both:
        print(f"=== First Q1_0_g128 tensor: {t['name']} ===")
        print(f"  dims (raw): {t['dims']}")
        print(f"  interpreted (row, col): ({rows}, {cols})  "
              f"blocks/row = cols/128 = {cols // 128 if cols % 128 == 0 else 'NOT DIV 128'}")

        # Compute expected size and dump first 36 bytes (two blocks)
        expected_bytes = rows * (cols // 128) * 18
        print(f"  expected bytes = rows*blocks*18 = {expected_bytes}")

        # Next tensor's offset - this tensor's offset should equal expected_bytes
        if len(r.tensors) > 1:
            idx = r.tensors.index(t)
            if idx + 1 < len(r.tensors):
                next_off = r.tensors[idx + 1]["offset"]
                actual_bytes = next_off - t["offset"]
                print(f"  actual bytes (next offset - this offset) = {actual_bytes}")
                print(f"  match: {expected_bytes == actual_bytes}")

        # Read first 36 bytes and dump
        r.f.seek(r.data_start + t["offset"])
        b = r.f.read(36)
        print(f"  first 36 bytes (hex): {b.hex()}")
        # Interpret as 2 blocks
        import numpy as np
        for i in range(2):
            blk = b[i * 18:(i + 1) * 18]
            scale = np.frombuffer(blk[:2], dtype=np.float16)[0]
            sign_bytes = blk[2:]
            print(f"  block {i}: fp16_scale={float(scale):+.6f}  sign_bytes={sign_bytes.hex()}")


if __name__ == "__main__":
    main()
