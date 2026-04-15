#!/usr/bin/env python3
"""Serial helper for pico-llm Qwen3 firmware.

Handles tokenization and detokenization so the user can type natural
language prompts. Communicates with the board over USB serial, sending
comma-separated token IDs and decoding the [TOKEN_ID] output back to text.

Usage:
    python tools/serial_chat.py                      # auto-detect port
    python tools/serial_chat.py --port /dev/cu.usbmodem1234
    python tools/serial_chat.py --model-name Qwen/Qwen3-0.6B

Requires: pip install pyserial transformers torch
"""

import argparse
import re
import sys
import time

import serial
import serial.tools.list_ports
from transformers import AutoTokenizer


def find_pico_port():
    """Auto-detect the Pico USB serial port."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        # Pico shows up as "Board in FS mode" or similar
        if "board in fs mode" in desc or "pico" in desc:
            return port.device
        # Also match by VID/PID (Raspberry Pi Pico USB CDC)
        if port.vid == 0x2E8A:
            return port.device
    # Fallback: look for cu.usbmodem on macOS
    for port in serial.tools.list_ports.comports():
        if "usbmodem" in (port.device or ""):
            return port.device
    return None


def wait_for_prompt(ser, timeout=60):
    """Read serial output until we see '> ', printing boot messages."""
    buf = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
            buf += chunk
            # Print boot messages
            lines = buf.split("\n")
            for line in lines[:-1]:
                print(f"  {line.rstrip()}")
            buf = lines[-1]
            if buf.endswith("> "):
                return True
        else:
            time.sleep(0.05)
    return False


def send_and_receive(ser, token_ids, tokenizer):
    """Send token IDs and decode the response in real-time."""
    # Send comma-separated token IDs
    payload = ",".join(str(t) for t in token_ids) + "\n"
    ser.write(payload.encode("ascii"))

    # Read response, decode tokens as they arrive
    buf = ""
    generated_tokens = []
    done = False

    while not done:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
            buf += chunk

            # Extract and decode [TOKEN_ID] patterns as they appear
            while True:
                m = re.search(r'\[(\d+)\]', buf)
                if not m:
                    break
                token_id = int(m.group(1))
                generated_tokens.append(token_id)
                text = tokenizer.decode([token_id])
                sys.stdout.write(text)
                sys.stdout.flush()
                # Remove everything up to and including the match
                buf = buf[m.end():]

            # Check for end-of-generation summary
            if re.search(r'\[\d+ tokens in \d+ ms', buf):
                done = True
            # Also stop if we see the next prompt
            if "> " in buf:
                done = True
        else:
            time.sleep(0.02)

    # Print timing summary
    summary = re.search(r'\[(\d+) tokens in (\d+) ms, (\d+) ms/tok\]', buf)
    if summary:
        n_tok = summary.group(1)
        total_ms = summary.group(2)
        per_tok = summary.group(3)
        print(f"\n  [{n_tok} tokens, {total_ms} ms total, {per_tok} ms/tok]")

    return generated_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Serial chat helper for pico-llm Qwen3 firmware")
    parser.add_argument("--port", default=None,
                        help="Serial port (auto-detect if not specified)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model for tokenizer")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate (firmware default)")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Find serial port
    port = args.port or find_pico_port()
    if not port:
        print("ERROR: No serial port found. Use --port to specify.")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        sys.exit(1)

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, args.baud, timeout=0.1)

    print("Waiting for board (press reset if needed)...")
    if not wait_for_prompt(ser):
        print("ERROR: Timed out waiting for '> ' prompt.")
        sys.exit(1)

    print("\nReady. Type a prompt and press Enter. Ctrl-C to quit.\n")

    try:
        while True:
            try:
                text = input("You: ")
            except EOFError:
                break

            if not text.strip():
                continue

            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            print(f"  [{len(token_ids)} tokens: {token_ids}]")
            print("Bot: ", end="", flush=True)

            # Send and decode response
            send_and_receive(ser, token_ids, tokenizer)
            print()

    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
