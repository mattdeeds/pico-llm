# pico-llm

Transformer LLM inference on the RP2350.

**Write-up:** [From 9M to 4B Parameters on a $1 Microcontroller](https://mattdeeds.com/writing/pico-llm.html)

## Measured results

All on an RP2350 at 250 MHz, 62.5 MHz SDIO bus, same firmware.

| Model | Quant | On card | Time/token | Bottleneck |
|---|---|---|---|---|
| Custom 9M test model | int8 | 15 MB | 0.69 s | SD I/O |
| Qwen3-0.6B | int8 | 570 MB | 55 s | SD I/O |
| Qwen3-4B-Thinking-2507 | Q4_0 | 2.16 GB | 3–6 min | SD I/O |
| Bonsai-1.7B | Q1_0_g128 | 232 MB | 29.1 s | compute |
| **Qwen3-0.6B** | **Q4_0** | **321 MB** | **19.4 s** | **SD I/O** |

## Hardware

- Raspberry Pi Pico 2 (RP2350), or the board in `pico-llm-kicad/v1/`
- microSD card, **U3 / V30 or better**
- SDIO wiring: CLK=GPIO10, CMD=GPIO11, D0–D3=GPIO12–15 (`docs/sd-card-pinmap.md`)

## Build

Requires the Pico SDK (`PICO_SDK_PATH`) and an arm-none-eabi toolchain.

```sh
cmake -B build -DPICO_BOARD=pico2
cmake --build build --target pico_llm
picotool load -f -x build/pico_llm.uf2
```

Options:

| Flag | Effect |
|---|---|
| `-DPICO_LLM_MODEL=bonsai` | Build for Bonsai-1.7B (Q1_0_g128) instead of Qwen3-0.6B |
| `-DPICO_LLM_PROFILE=ON` | Per-token timing breakdown over serial |
| `-DPICO_LLM_BENCH=ON` | Also build `bench_sd`, a raw SD read benchmark |
| `-DSDIO_DATA_CLK_DIVIDER=5` | Force a 50 MHz (in-spec) bus instead of the 62.5 MHz default |

## Preparing a card

Export the model, then write it to the card's **raw blocks** — there is no
filesystem.

```sh
python3 tools/export_qwen3_q4.py -o models/qwen3_0.6b_q4.bin
python3 tools/export_bonsai.py   -o models/bonsai_1p7b_q1.bin   # optional
```

```sh
diskutil unmountDisk /dev/diskN        # macOS; use umount on Linux
sudo dd if=models/qwen3_0.6b_q4.bin of=/dev/rdiskN bs=1m
```

Raw block devices only accept writes in multiples of 512 bytes, and the exported
files are not a whole number of blocks. `dd` with `bs=1m` writes every full
megabyte and then fails on the remainder with `Invalid argument`, leaving the
tail of the classifier missing. Either pad the file to a 512-byte boundary
first, or write the tail separately and confirm `dd` reports equal records in
and out.

## Talking to it

There is no tokenizer on the device, vocabulary does not fit in RAM.
The host does tokenization, the chat template and detokenization:

```sh
python3 tools/serial_chat.py --model-name Qwen/Qwen3-0.6B
```

The chat template matters. Without it the model autocompletes the prompt instead
of answering it.

## Validating an export

```sh
python3 tools/test_qwen3_q4.py models/qwen3_0.6b_q4.bin \
  --model-name Qwen/Qwen3-0.6B --rope-theta 1000000 --prompt "<a long prompt>"
```

## Credit

Inspired by Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).
SDIO by [SDIO_RP2350](https://github.com/rabbitholecomputing/SDIO_RP2350).
Q4_0 follows llama.cpp's block format.
