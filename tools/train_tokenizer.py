#!/usr/bin/env python3
"""Train a BPE tokenizer on TinyStories for pico-llm.

Produces a tokenizer.json compatible with the HuggingFace tokenizers library,
with vocab_size=8192 and special tokens for pad/bos/eos/unk.
"""

import argparse
import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on TinyStories")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--output", "-o", default="models/tokenizer.json")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Limit training examples (default: use all)")
    args = parser.parse_args()

    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Build iterator over text
    def text_iterator():
        count = 0
        for example in ds:
            yield example["text"]
            count += 1
            if args.num_examples and count >= args.num_examples:
                break

    print(f"Training BPE tokenizer with vocab_size={args.vocab_size}...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Verify special token IDs
    for i, tok in enumerate(special_tokens):
        tid = tokenizer.token_to_id(tok)
        assert tid == i, f"Expected {tok} at id {i}, got {tid}"
        print(f"  {tok} -> {tid}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}")

    # Quick sanity check
    test = "Once upon a time, there was a little girl named Lucy."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nSanity check:")
    print(f"  Input:   {test}")
    print(f"  Tokens:  {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
