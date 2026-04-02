#!/usr/bin/env python3
"""Validate a trained pico-llm model before export.

Computes validation perplexity, generates sample text, and saves reference
logits for verifying the exported binary.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import Transformer, TinyStoriesDataset


def generate(model, tokenizer, prompt_ids, max_new_tokens=200,
             temperature=1.0, top_p=0.9, device="cpu"):
    """Generate text with nucleus sampling."""
    model.eval()
    ids = list(prompt_ids)
    eos_id = tokenizer.token_to_id("<|eos|>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]  # [1, vocab]

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                # Nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
            else:
                next_token = logits.argmax(dim=-1)

            next_id = next_token.item()
            ids.append(next_id)
            if next_id == eos_id:
                break

    return ids


def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--model", default="models/model.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--reference-output", default="models/reference_logits.npz")
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    # Load model
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    margs = ckpt["args"]
    model = Transformer(
        dim=margs["dim"], hidden_dim=margs["hidden_dim"],
        n_layers=margs["n_layers"], n_heads=margs["n_heads"],
        n_kv_heads=margs["n_kv_heads"], vocab_size=margs["vocab_size"],
        max_seq_len=margs["max_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model ({model.count_parameters():,} params)")

    tokenizer = Tokenizer.from_file(args.tokenizer)

    # --- Validation perplexity ---
    print("\nComputing validation perplexity...")
    val_data = TinyStoriesDataset(args.tokenizer, margs["max_seq_len"],
                                  split="validation")
    total_loss = 0.0
    with torch.no_grad():
        for i in range(args.val_batches):
            x, y = val_data.get_batch(args.batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, margs["vocab_size"]), y.view(-1)
            )
            total_loss += loss.item()
    avg_loss = total_loss / args.val_batches
    perplexity = math.exp(avg_loss)
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    # --- Sample generation ---
    print("\n--- Greedy generation ---")
    bos_id = tokenizer.token_to_id("<|bos|>")
    prompt = [bos_id]
    ids = generate(model, tokenizer, prompt, max_new_tokens=200,
                   temperature=0, device=device)
    print(tokenizer.decode(ids))

    print("\n--- Nucleus sampling (T=0.8, p=0.9) ---")
    for i in range(3):
        ids = generate(model, tokenizer, prompt, max_new_tokens=200,
                       temperature=0.8, top_p=0.9, device=device)
        print(f"\nSample {i+1}:")
        print(tokenizer.decode(ids))

    # --- Reference logits ---
    print("\nSaving reference logits...")
    # Use a fixed input: BOS followed by first 15 tokens of greedy output
    ref_ids = generate(model, tokenizer, [bos_id], max_new_tokens=15,
                       temperature=0, device=device)
    ref_input = torch.tensor([ref_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        ref_logits = model(ref_input).cpu().numpy()

    np.savez(args.reference_output,
             input_ids=np.array(ref_ids, dtype=np.int32),
             logits=ref_logits.astype(np.float32))
    print(f"Saved reference to {args.reference_output}")
    print(f"  Input tokens: {ref_ids}")
    print(f"  Logits shape: {ref_logits.shape}")


if __name__ == "__main__":
    main()
