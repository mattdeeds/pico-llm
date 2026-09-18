#!/usr/bin/env python3
"""Serial helper for pico-llm Qwen3 firmware.

Handles tokenization and detokenization so the user can type natural
language prompts. Communicates with the board over USB serial, sending
comma-separated token IDs and decoding the [TOKEN_ID] output back to text.

Usage:
    python tools/serial_chat.py                      # auto-detect port
    python tools/serial_chat.py --port /dev/cu.usbmodem1234
    python tools/serial_chat.py --model-name Qwen/Qwen3-0.6B
    python tools/serial_chat.py --raw                # no chat template

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
    # The Raspberry Pi Debug Probe shares the Pico's USB vendor ID; skip it.
    ports = [p for p in serial.tools.list_ports.comports()
             if "cmsis-dap" not in (p.description or "").lower()]
    for port in ports:
        desc = (port.description or "").lower()
        # Pico shows up as "Board in FS mode" or similar
        if "board in fs mode" in desc or "pico" in desc:
            return port.device
        # Also match by VID/PID (Raspberry Pi Pico USB CDC)
        if port.vid == 0x2E8A:
            return port.device
    # Fallback: look for cu.usbmodem on macOS
    for port in ports:
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


# Firmware reads the prompt into a 512-byte buffer (src/main.c) and stops
# reading at 511 characters, leaving the newline behind. Keep the
# comma-separated token IDs to 510 characters so the newline is consumed.
MAX_PAYLOAD_CHARS = 510


def build_prompt_ids(tokenizer, text, raw=False, think=False):
    """Tokenize a user prompt, wrapped in the model's chat template unless raw.

    The firmware starts every prompt at position 0, so each prompt is a
    single-turn conversation. Qwen3's hybrid models think by default, which
    would use up the firmware's 64-token budget, so thinking is off unless
    requested. Qwen3-4B-Thinking-2507's template always opens a <think> block.
    """
    if raw:
        return tokenizer.encode(text, add_special_tokens=False)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def send_and_receive(ser, token_ids, tokenizer, stop_ids=()):
    """Send token IDs and decode the response in real-time.

    Output after a stop token (end of turn) is not printed. The firmware has
    no stop-token check, so it keeps generating until its token limit.
    """
    # Send comma-separated token IDs
    payload = ",".join(str(t) for t in token_ids) + "\n"
    ser.write(payload.encode("ascii"))

    # Read response, decode tokens as they arrive
    buf = ""
    generated_tokens = []
    done = False
    stopped = False

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
                if token_id in stop_ids and not stopped:
                    stopped = True
                    sys.stdout.write("\n  [end of turn; board keeps "
                                     "generating until its token limit]")
                    sys.stdout.flush()
                elif not stopped:
                    sys.stdout.write(tokenizer.decode([token_id]))
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
    parser.add_argument("--raw", action="store_true",
                        help="Send the prompt as plain text, without the "
                             "chat template")
    parser.add_argument("--think", action="store_true",
                        help="Enable thinking mode in the chat template")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Prompt format: {'raw text' if args.raw else 'chat template'}"
          + ("" if args.raw else f", thinking {'on' if args.think else 'off'}"))
    stop_ids = () if args.raw else (tokenizer.convert_tokens_to_ids("<|im_end|>"),)

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

    print("\nReady. Type a prompt and press Enter. Ctrl-C to quit.")
    print("Commands: /temp N, /rep N, /greedy (sent directly to board)\n")

    try:
        while True:
            try:
                text = input("You: ")
            except EOFError:
                break

            if not text.strip():
                continue

            # Pass slash commands directly to board
            if text.startswith("/"):
                ser.write((text + "\n").encode("ascii"))
                time.sleep(0.2)
                if ser.in_waiting:
                    resp = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
                    print(f"  {resp.strip()}")
                # Wait for next prompt
                wait_for_prompt(ser, timeout=5)
                continue

            # Tokenize
            token_ids = build_prompt_ids(tokenizer, text, args.raw, args.think)
            payload_len = len(",".join(str(t) for t in token_ids))
            if payload_len > MAX_PAYLOAD_CHARS:
                print(f"  Prompt too long: {len(token_ids)} tokens is "
                      f"{payload_len} chars, firmware accepts "
                      f"{MAX_PAYLOAD_CHARS}. Try a shorter prompt.")
                continue
            print(f"  [{len(token_ids)} tokens: {token_ids}]")
            print("Bot: ", end="", flush=True)

            # Send and decode response
            send_and_receive(ser, token_ids, tokenizer, stop_ids)
            print()

    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
