#!/usr/bin/env python3
"""Encode a text prompt to token IDs for pico-llm.

Since the device-side BPE encoder is not yet implemented, this script
encodes on the host. Output can be hardcoded into firmware or sent via UART.
"""

import argparse
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode text to token IDs")
    parser.add_argument("text", nargs="?", default=None, help="Text to encode")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--add-bos", action="store_true", default=True)
    parser.add_argument("--format", choices=["list", "c-array", "comma"],
                        default="list")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos_id = tokenizer.token_to_id("<|bos|>")

    if args.text is None:
        # Interactive mode
        print("Enter text (Ctrl-D to exit):")
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            ids = tokenizer.encode(text).ids
            if args.add_bos:
                ids = [bos_id] + ids
            print(f"  Tokens ({len(ids)}): {ids}")
            print(f"  Decoded: {tokenizer.decode(ids)}")
    else:
        ids = tokenizer.encode(args.text).ids
        if args.add_bos:
            ids = [bos_id] + ids

        if args.format == "list":
            print(ids)
        elif args.format == "c-array":
            elements = ", ".join(str(i) for i in ids)
            print(f"int prompt[] = {{{elements}}};")
            print(f"int prompt_len = {len(ids)};")
        elif args.format == "comma":
            print(",".join(str(i) for i in ids))

        print(f"\n// {len(ids)} tokens")
        print(f"// Decoded: {tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
