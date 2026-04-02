#!/usr/bin/env python3
"""Train a small LLaMA-style model on TinyStories for pico-llm.

Architecture exactly matches the C inference engine in src/transformer.c:
RMSNorm, RoPE, SwiGLU FFN, grouped-query attention, no bias.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, base=10000.0):
    """Precompute RoPE frequencies matching transformer.c:43-63."""
    # The C code iterates i in range(0, dim, 2) with head_dim = i % head_size
    # Since n_kv_heads == n_heads, kv_dim == dim, so all of q/k gets rotated.
    # freq = 1 / 10000^(head_dim / head_size) where head_dim steps 0,2,4,...
    # This is standard per-head RoPE.
    head_size = dim // 4  # dim / n_heads = 256/4 = 64
    freqs = []
    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1.0 / (base ** (float(head_dim) / float(head_size)))
        freqs.append(freq)
    freqs = torch.tensor(freqs, dtype=torch.float32)  # [dim/2]
    t = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    angles = torch.outer(t, freqs)  # [max_seq_len, dim/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin  # both [max_seq_len, dim/2]


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: [batch, seq, dim]."""
    # Split into pairs: x[..., 0::2] and x[..., 1::2]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    # cos/sin are [seq, dim/2], broadcast over batch
    c = cos[:x.shape[1]]  # [seq, dim/2]
    s = sin[:x.shape[1]]
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    # Interleave back
    out = torch.stack([out0, out1], dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_size = dim // n_heads
        self.kv_dim = self.head_size * n_kv_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.wq(x)  # [B, T, dim]
        k = self.wk(x)  # [B, T, kv_dim]
        v = self.wv(x)  # [B, T, kv_dim]

        # Apply RoPE to full q and k vectors (matching C code which passes kv_dim=dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Reshape to heads
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)

        # Expand KV heads if grouped-query attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                 vocab_size, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.classifier = nn.Linear(dim, vocab_size, bias=False)

        # Precompute RoPE
        cos, sin = precompute_rope(dim, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TinyStoriesDataset:
    """Tokenizes TinyStories and serves packed sequences of fixed length."""

    def __init__(self, tokenizer_path, seq_len, split="train"):
        self.seq_len = seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        print(f"Loading TinyStories ({split})...")
        ds = load_dataset("roneneldan/TinyStories", split=split)

        print("Tokenizing...")
        all_ids = []
        for i, example in enumerate(ds):
            ids = self.tokenizer.encode(example["text"]).ids
            all_ids.append(self.bos_id)
            all_ids.extend(ids)
            all_ids.append(self.eos_id)
            if (i + 1) % 500000 == 0:
                print(f"  {i+1} examples tokenized ({len(all_ids)} tokens)")

        self.tokens = np.array(all_ids, dtype=np.int32)
        self.n_tokens = len(self.tokens)
        print(f"Total tokens: {self.n_tokens:,}")

    def get_batch(self, batch_size, device):
        """Get a random batch of packed sequences."""
        ix = torch.randint(0, self.n_tokens - self.seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.tokens[i:i+self.seq_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.tokens[i+1:i+1+self.seq_len].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Train pico-llm model")
    # Model
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=704)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=512)
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # IO
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--output", "-o", default="models/model.pt")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Model
    model = Transformer(
        dim=args.dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        vocab_size=args.vocab_size, max_seq_len=args.max_seq_len,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer (weight decay only on non-embedding, non-norm params)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # Data
    train_data = TinyStoriesDataset(args.tokenizer, args.max_seq_len, split="train")
    val_data = TinyStoriesDataset(args.tokenizer, args.max_seq_len, split="validation")

    # Resume
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resuming from step {start_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.max_steps} steps...")
    t0 = time.time()
    tokens_processed = 0

    for step in range(start_step, args.max_steps):
        # Learning rate schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Validation
        if step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_batches):
                    x, y = val_data.get_batch(args.batch_size, device)
                    logits = model(x)
                    val_loss += F.cross_entropy(
                        logits.view(-1, args.vocab_size), y.view(-1)
                    ).item()
                val_loss /= args.val_batches
            elapsed = time.time() - t0
            tps = tokens_processed / elapsed if elapsed > 0 else 0
            print(f"step {step:>6d} | val_loss {val_loss:.4f} | "
                  f"lr {lr:.2e} | {tps:.0f} tok/s")
            model.train()

        # Checkpoint
        if step > 0 and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Forward + backward
        x, y = train_data.get_batch(args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_processed += args.batch_size * args.max_seq_len

    # Save final model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
    }, args.output)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/3600:.1f} hours")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
