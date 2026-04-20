# Blog Post Outline: From 9M to 4B Parameters on a $1 Microcontroller

## Hook

A climbing-the-ladder story: starting from Andrej Karpathy's llama2.c as inspiration, I built a bare-metal LLM inference engine for the RP2350 (dual Cortex-M33, 520 KB RAM) and progressively scaled up — first a custom 9M parameter toy, then Qwen3-0.6B (a real open-source model), then Qwen3-4B-Thinking (a reasoning-tuned 4B), then a natively-trained 1-bit model (Bonsai-1.7B). At each step a new wall appeared. The plot twist: the fastest result came from going *back* to the smallest real model with better quantization and an overclocked chip — 15.5 seconds per token, 2.5× faster than the 1-bit model, with better output quality.

No OS, no frameworks, no GPU. Just C, an SD card, and a lot of streaming.

## 1. The Premise: What if you didn't care about speed?

- Modern LLMs require expensive GPUs. What's the absolute cheapest hardware you could run one on?
- Imagine a future where an ASI model exists that can solve problems better than any human — you'd want to run it no matter the cost, even on a few-dollar MCU
- The RP2350 (Raspberry Pi Pico 2): dual Cortex-M33 at 150 MHz, 520 KB SRAM, ~$1
- The naive ratio: a 600M parameter int8 model is 570 MB. RAM is 0.5 MB. That's 1000:1. Push to a 4B model in Q4_0 and it's 4000:1. Clearly impossible in the usual sense.
- The plan: climb the ladder. Start tiny, prove the concept, scale up until the chip says no.

## 2. The Key Insight: Stream Everything

- Inspiration: Andrej Karpathy's llama2.c — a single-file C transformer inference engine
- LLM inference consumes weights sequentially — each layer's weights are used once per token, in order
- If we serialize weights in consumption order on an SD card, we can stream them through a tiny buffer
- 32 KB double buffer: while we compute on one tile of weights, DMA fills the next
- We never hold more than a fraction of the model in RAM

## 3. Building the Engine (Mar 28-30)

- Bare-metal C on RP2350, no RTOS or libc bloat
- PIO-based 4-bit SDIO driver at 25 MHz (much faster than SPI)
- Three-phase per-layer pipeline:
  1. Stream attention weights → Q/K/V projections with tiled int8 matmul → RoPE
  2. Pause streaming → write KV cache to SD → online softmax attention (single-pass, never materializes attention matrix)
  3. Resume streaming → output projection → SwiGLU FFN
- Int8 quantization: weights as int8 with per-tensor scales, int8×int8 with int32 accumulate, float32 only for norms and softmax
- KV cache lives on SD card too — RAM only holds the current activation vector (~1 KB for the small model)

## 4. Ladder rung #1: A Model to Test With (Apr 1)

- Can't validate the engine without a model — but downloading a real one comes later
- Custom 9M parameter LLaMA-architecture model: dim=256, 6 layers, 4 heads, 8K vocab
- Trained on TinyStories: 15K steps, 7.5 hours, perplexity 4.96
- Full toolchain: BPE tokenizer training → model training → validation → int8 binary export
- 14.7 MB on the SD card, 100% argmax agreement with float32 reference
- NumPy test tool reimplements the full forward pass to verify the export byte-for-byte
- Why start small: I need to prove the streaming engine works end-to-end before trusting it with a real model

## 5. Custom PCB (Mar 30 - Apr 1)

- KiCad schematic + layout: RP2350 + SDIO SD card slot + USB
- Ordered from JLCPCB
- Minimal design — just enough to run the inference engine

## 6. Hardware Bring-Up: Everything That Went Wrong (Apr 13)

- First power-on: SD card won't respond (ACMD41 never completes) — hardware issue on PCB, fixed with soldering
- Only 1 of 3 SD cards works in the slot (32GB SDHC U1)
- **The INTERFACE bug**: SDIO library compiled with example GPIO pins (34-39) instead of the project's (10-15). Root cause: one word in CMakeLists.txt — `INTERFACE` should have been `PRIVATE`. Hours of debugging for a one-word fix.
- USB CDC serial: switched from UART, added connection wait so boot output isn't lost
- Post-write busy wait: SD card enters programming state after KV cache writes, needed CMD13 polling
- First inference runs! But all output is token 8190, logits 1000x too large...

## 7. The Buffer Boundary Bug (Apr 13)

- The pivotal debugging session
- When a weight matrix row spanned two 32 KB buffers, the partial tail was silently discarded
- Every subsequent weight read was shifted, corrupting the entire forward pass
- Three interrelated fixes: boundary-spanning row fallback, ws_drain re-read from correct offset, prefetch overflow clamp
- After the fix: "Once upon a time there was a little girl..." — coherent English text, ~1 sec/token
- First light moment

## 8. Going Dual-Core (Apr 13)

- SD reads and matmul compute were sequential — each waiting for the other
- RP2350 has two Cortex-M33 cores. Core 0 owns all SD I/O (DMA + IRQ chaining), Core 1 runs matmul
- True overlap: while Core 1 multiplies the current weight tile, DMA fills the next buffer
- 1.46x speedup: ~690 ms/token, down from ~1010 ms/token
- Attempted Core 1 SD access — PIO commands hang (library is Core 0-specific). Workaround: Core 0 does all I/O.

## 9. Making It Interactive (Apr 13)

- On-device BPE token decoding: byte-level BPE → readable text printed over USB serial
- Greedy longest-match tokenizer for prompt encoding
- Interactive loop: type a prompt, get streamed text output in real time
- A working LLM chatbot on a microcontroller — but only 9M parameters, trained on TinyStories. Interesting as a demo, but not a "real" model. Time to climb the next rung.

## 10. Ladder rung #2: Qwen3-0.6B — a real open-source model (Apr 14)

- Goal: run a real, open-source model, not just a custom toy
- Evaluated Qwen3.5-0.8B first — rejected because its hybrid DeltaNet/GQA architecture is too different from standard transformers
- Qwen3-0.6B: standard transformer, same SwiGLU + RMSNorm + RoPE stack. 600M params, 28 layers, 151K vocab.
- The scale jump: 570 MB vs 14.7 MB, dim 1024 vs 256, vocab 151K vs 8K
- The wall we hit first: the 608 KB logits problem

### The 608 KB Problem

- `float logits[151936]` = 608 KB — exceeds the chip's entire 520 KB RAM
- Can't store logits. Can't do normal softmax sampling.
- Solution: streaming argmax — process the 151K-row classifier in 256-row chunks
- Int32 accumulator comparison is valid because dequantization (multiply by positive scale) preserves ordering
- Zero bytes of logits storage. Total RAM: 150 KB.

### Architectural Differences

- GQA: 16 query heads, 8 KV heads (2:1 grouping) — the attention loop already handled this
- QK-Norm: per-head RMSNorm on Q and K vectors — new operation, small code addition
- Decoupled head_dim: Qwen3 has head_dim=128 but dim/n_heads=64. Q output is 2048, not 1024. Non-square weight matrices.
- Split-half RoPE: Qwen3 pairs element i with i+head_dim/2 (not consecutive pairs). Caught during validation — accuracy jumped from 25% to 82% after fix.
- Tied embeddings: embedding table = classifier weights. No separate copy — saves 590 MB of SD space. Firmware reads int8 from classifier section and dequantizes.
- 151K vocab won't fit in RAM for tokenization — off-device tokenization via Python serial helper

### Export & Validation

- Python export tool downloads Qwen3-0.6B from HuggingFace, int8 quantizes, writes V2 binary format
- NumPy validation tool reimplements entire Qwen3 forward pass and compares against HuggingFace
- 82% argmax agreement — validates the architecture port is correct, remaining error is quantization noise

### First Real Model Output

- 570 MB written to SD card, firmware flashed
- "Hello world" → varied, coherent English text
- ~55 seconds per token — slow, but it works. A real 600M parameter model on a microcontroller.

## 11. Sampling Without Logits (Apr 15)

- Greedy decoding produces repetitive loops ("I'm going to be a part of..." forever)
- Standard temperature sampling needs the full logits vector — which we can't store
- The Gumbel-max trick: `argmax(logit/T + Gumbel_noise)` is mathematically equivalent to `sample(softmax(logit/T))`
- Works in the streaming classifier — just add noise to each chunk's scores before comparing
- Repetition penalty: reduce logit magnitude for tokens in a 64-token circular history
- Result: non-repetitive, varied output. Configurable via serial commands (/temp, /rep, /greedy).
- But Qwen3-0.6B is a base model: no instruction tuning, no reasoning capability. The output is coherent but not useful. One more rung to climb.

## 12. Ladder rung #3: Qwen3-4B-Thinking with Q4_0 (Apr 15-16)

- Target: Qwen3-4B-Thinking-2507 — a 4B parameter reasoning-tuned model (7x larger, fine-tuned for chain-of-thought)
- Why this one: it's a real, useful model. If it works, we've gone from "cool demo" to "actually reasoning on a microcontroller."
- Architecturally identical to Qwen3-0.6B (same `Qwen3ForCausalLM`) — just bigger:
  - dim 2560 vs 1024, hidden 9728 vs 3072, 36 vs 28 layers, 32 vs 16 heads
  - rope_theta 5M vs 1M (different positional encoding scale)
- The engine port is trivial — change compile-time constants. The challenge is the weights.

### The int8 problem

- Int8 Qwen3-4B would be ~3.9 GB — right up against our uint32_t SD offset limit and ~6.5 min/tok
- Needed a better quantization scheme, both for quality and size

### Why our naive quantization wasn't good enough

- Per-tensor int8: one scale factor for millions of weights (e.g., 10M values with one scale for the FFN)
- If a few outliers dominate the range, all the small values get poor resolution
- Fine for small models, marginal for 600M, likely broken for 4B

### GGUF Q4_0: per-block scales

- llama.cpp's Q4_0 format: weights in blocks of 32, each block has its own float16 scale
- 18 bytes per 32 weights (2 for scale + 16 packed nibbles) = 4.5 bits/weight
- Much better quality per bit than per-tensor because outliers only affect one block
- Same nibble-packing trick: low nibble = position j (0..15), high nibble = position j+16 (16..31)
- Dequantize: `weight = (nibble - 8) * block_scale`

### Implementation

- Wrote our own Q4_0 export tool (load safetensors from HuggingFace, quantize per block)
- Matched ggml's exact algorithm: `d = max_element / -8` (preserves sign), split-half nibble packing
- New firmware matmul kernel: int4 × int8 inner loop (still fast integer math), per-block float scale multiply
- Core 1 compute worker updated for float accumulator (Q4_0 has per-block dequant built in)
- NumPy validation tool to verify byte-for-byte correctness against HuggingFace
- 75% argmax agreement on 4-token prompt — validates Q4_0 implementation

### The Result

- 2.16 GB model on SD card (down from 3.9 GB int8)
- 330 KB RAM used (still plenty of headroom in 520 KB)
- Generating text from a 4B parameter reasoning-tuned model on a $1 MCU — but at 3–6 min/tok, too slow to be pleasant. If we want bigger *and* usable, the next rung can't just add parameters. We need to shrink the *weights themselves*.

## 13. Ladder rung #4: Bonsai-1.7B at 1.125 bits per weight (Apr 16)

- Target: `prism-ml/Bonsai-1.7B` — a Qwen3-1.7B architecture model, but trained end-to-end at 1 bit per weight. Not post-hoc quantized. The ±scale is baked into training.
- Why this one: speed is now the real wall, not size. Every byte streamed from SD is compute we wait on. Going from 4.5 bits/weight (Q4_0) to 1.125 bits/weight (Q1_0_g128) cuts SD reads by 4× without changing the parameter count.
- The reframe: we've been climbing "more parameters." Bonsai climbs a different axis — "fewer bits per parameter" — which translates directly to speed on our I/O-bound chip.

### The Q1_0_g128 format (PrismML / llama.cpp extension)

- 18 bytes per block of **128 weights** (vs Q4_0's block of 32): 2-byte fp16 scale + 16 bytes of sign bits
- Bit k of byte j → weight at position `j*8 + k`: bit=1 → +scale, bit=0 → −scale
- Same structural pattern as Q4_0 (per-block fp16 scale + 16 bytes of packed data), just 4× denser
- Effective 1.125 bits/weight (1 sign bit + 16-bit scale / 128 values)
- Works because Bonsai's weights were *trained* to live in {−d, +d} per block — no quantization error. Max absolute logit error vs FP16 reference: **0.55** (Q4_0 on Qwen3-4B was 9.57).

### Source: parse the GGUF directly

- `prism-ml/Bonsai-1.7B-gguf` ships a 248 MB GGUF file with custom dtype **41** (`Q1_0_g128`)
- Wrote a minimal GGUF parser in Python (~200 lines): the standard `gguf-py` package doesn't know this dtype
- Transcribe raw Q1 block bytes verbatim into our V2 binary — no requantization
- Tokenizer, architecture constants, and RoPE theta all come from the GGUF metadata

### Firmware swap

- The streaming/tiling/dual-core scaffold carries over unchanged
- New inner kernel `matmul_q1_0_g128_tile`: sign × int8 → int32, per-block fp16 scale. Trick: `block_sum = 2*sum(x where bit=1) − sum(x)` avoids per-bit branching
- One-line math change: `Q4_ROW_BYTES(cols) = (cols/32)*18` → `Q1_ROW_BYTES(cols) = (cols/128)*18`
- Compile-time constants for Qwen3-1.7B: dim=2048, hidden=6144, 28 layers, 16 heads, 8 KV heads, vocab=151669, rope_theta=1M

### Validation: 100% argmax

- NumPy forward pass reading the exported binary, compared against HF `prism-ml/Bonsai-1.7B-unpacked`
- 4/4 argmax agreement on "Once upon a time", max abs error 0.55, top-5 overlap 95%
- The trained-at-1-bit nature makes this much cleaner than any post-hoc quant we've tested

### The Result

- **232 MB** model on SD card (down from 570 MB for Qwen3-0.6B int8 — smaller *and* 3× more parameters)
- **~230 KB RAM** used — the smallest footprint of any Qwen3 model we've run
- **~38 s/tok** on hardware after optimization — the fastest real-model token rate yet
- Coherent English on the first prompt: "Hello world!" → "I'm a:\". I think that's all you'd need. Just let me know..." (base-model chatter, as expected)
- The speed regime has flipped: for the first time, **compute is the bottleneck**, not SD I/O. 232 MB/tok at ~25 MB/s (50 MHz SDIO) should be ~9.3 s of reads; we're at 38 s, so compute (~1.7B multiply-adds per token on a 200 MHz Cortex-M33) still dominates even after optimization.

### Optimization: 50 MHz SDIO and ARM DSP intrinsics (Apr 16-17)

Two optimizations stacked for a combined **39% speedup** (62.7 → 38.1 s/tok):

**1. 50 MHz high-speed SDIO** (25% speedup, 62.7 → 46.8 s/tok)
- Issued CMD6 SWITCH_FUNC to put the SD card into SDR25/HS mode
- Bumped GPIO drive strength 8 → 12 mA on CLK/CMD/DAT0-3
- Pinned sysclk to 200 MHz for exact integer PIO divider (200/4 = 50 MHz)
- Added multi-block smoke read after HS init with graceful 25 MHz fallback
- Previous attempt (Apr 13) hit CRC errors — the drive strength bump fixed it

**2. ARM DSP intrinsics in the matmul kernel** (19% speedup, 46.8 → 38.1 s/tok)
- Benchmarked three kernel variants against baseline using DWT cycle counter
- Winner: split-phase approach — tight `total` loop (no sign-byte dependency) + 2×-unrolled `pos` loop with GE-flag byte selection
- Key DSP instructions: `SADD8`/`SEL` for nibble-indexed byte selection via GE flags, `SXTB16`/`SXTAB16`/`SADD16` for packed int16 horizontal sums
- Inline ASM for `SXTAB16 Rd, Rn, Rm, ROR #8` — GCC doesn't fold the rotation operand into the instruction, wasting 4 instructions per sign byte
- `-O3 -funroll-loops` for quantize.c (rest of firmware stays at -O2)
- Result: **5.07 → 4.07 cycles/weight** (20% kernel speedup, measured via DWT)
- Correctness verified: exact match with baseline kernel output

## 14. The Plot Twist: Fewer Parameters, Faster and Smarter (Apr 17)

- Bonsai-1.7B at 1-bit was our fastest — but was it our *best*?
- Key realization: 1-bit quantization reduces storage but not compute. Bonsai has 2.8× more parameters than Qwen3-0.6B. Compute went *up* even though disk size went *down*.
- The right question isn't "how many parameters can we cram in?" — it's "what's the best quality per second?"
- Qwen3-0.6B at Q4_0: fewer parameters (less compute), moderate quantization (much better quality), 321 MB on disk (only 40% larger than Bonsai's 232 MB)

### Bottleneck analysis

| Model | Quant | I/O time | Compute time | Total | Quality |
|---|---|---|---|---|---|
| Bonsai-1.7B | Q1_0 | ~9s | ~29s | 38s | Worst |
| Qwen3-0.6B | Q4_0 | ~16s | ~2.5s | 18.3s | Good |
| Qwen3-0.6B | Q4_0 @ 250 MHz | ~16s | ~2s | **15.5s** | Good |

- Q4 matmul kernel: int4×int8 per block of 32, per-block fp16 scale. Same structural pattern as Q1_0 but with actual multiplies instead of conditional adds.
- The Q4_0 export script already existed from the Qwen3-4B port — just pointed it at Qwen3-0.6B

### 250 MHz overclock

- RP2350 at 200 MHz → 250 MHz: 25% more clock cycles per second
- Catch: 250 MHz at boot caused ACMD41 timeout during SD card init
- Fix: init SD at 200 MHz, switch to 250 MHz after init. SDIO driver auto-adjusts PIO divider (250/5 = 50 MHz bus speed).
- Linear speedup on the compute-bound portion: 18.3 → 15.5 s/tok

### The twist

- We climbed four rungs of model size (9M → 600M → 4B → 1.7B) chasing parameters
- The fastest and most practical result came from going *back* to the smallest real model with better quantization and a faster clock
- 15.5 s/tok with coherent, grammatical English — 2.5× faster than Bonsai, better output quality

## 15. The Numbers

| Metric | Custom 9M | Qwen3-0.6B int8 | Qwen3-4B Q4_0 | Bonsai-1.7B Q1_0 | Qwen3-0.6B Q4_0 |
|--------|-----------|-----------------|----------------|------------------|-----------------|
| Parameters | 9M | 600M | 4B | 1.7B | 600M |
| Bits/weight | 8 | 8 (per-tensor) | 4.5 (per-block) | 1.125 (per-block) | 4.5 (per-block) |
| Model size | 14.7 MB | 570 MB | 2.16 GB | 232 MB | 321 MB |
| Layers | 6 | 28 | 36 | 28 | 28 |
| Dimensions | 256 | 1024 | 2560 | 2048 | 1024 |
| Vocab | 8,192 | 151,936 | 151,936 | 151,669 | 151,936 |
| Token rate | ~690 ms | ~55 s | ~3–6 min | ~38 s | **~15.5 s** |
| RAM used | ~114 KB | ~150 KB | ~330 KB | ~230 KB | ~140 KB |
| SD reads/tok | ~10 MB | ~570 MB | ~2.16 GB | ~232 MB | ~321 MB |
| SDIO speed | 25 MHz | 25 MHz | 25 MHz | 50 MHz | 50 MHz |
| Sys clock | 150 MHz | 150 MHz | 150 MHz | 200 MHz | **250 MHz** |
| Bottleneck | SD I/O | SD I/O | SD I/O | compute | SD I/O |
| Hardware cost | ~$5 | same | same | same | same |

## 16. What's Next

- **DSP-optimized Q4_0 kernel**: the current Q4_0 kernel is scalar C. ARM DSP intrinsics (SMLAD for dual int16 multiply-accumulate) could cut cycles/weight significantly. We proved this works for Q1_0 — same approach applies.
- **Dual-core compute**: Core 0 is mostly idle waiting for DMA during matmul. It could process half the tile rows in parallel with Core 1, for ~1.5× compute throughput.
- **300 MHz overclock**: 250 MHz is stable. Some RP2350 chips run at 300 MHz — another 20% if the silicon cooperates.
- **Bonsai-8B**: same Q1_0_g128 pipeline with bigger constants (dim=4096, 36 layers). ~1.15 GB on disk. The port is a few-line change, but at 1-bit the compute-bound regime makes it slower per token than Qwen3-0.6B Q4 despite better intelligence.
- **Natively trained 1-bit small models**: if a BitNet-style model existed at 0.6B params, it could combine the compute advantage of fewer params with the I/O advantage of 1-bit storage. None exist yet in the open-weight ecosystem.

## 17. What I Learned

- **You don't need the model in RAM** — you just need it in order. Sequential streaming makes the impossible possible.
- **More parameters isn't always better**: Bonsai-1.7B at 1-bit had 3× more params than Qwen3-0.6B but was 2.5× slower and lower quality. The optimal is to minimize *both* storage and parameter count — moderate quantization on a small model beats extreme quantization on a large one.
- **The bottleneck shifts**: SD I/O dominated for everything through 4B Q4_0. At 1-bit, compute took over. Switching back to Q4_0 on a smaller model made I/O the bottleneck again. Each regime needs different optimizations.
- **Quantization granularity matters**: per-tensor int8 was fine for 600M, 4B needed per-block Q4_0, and Bonsai showed that *training*-time quantization crushes *post-hoc* quantization on error (0.55 vs 9.57 max logit error).
- **Training for the target beats adapting after the fact**: Bonsai's weights are 1-bit because they were trained to be — the "quantization" is lossless.
- **The Gumbel-max trick is beautiful**: temperature sampling without storing logits, using a one-line mathematical identity.
- **Standard formats pay off**: reusing llama.cpp's block layouts (Q4_0, Q1_0_g128) meant firmware changes were ~50 lines per quantization scheme and we could validate against HuggingFace reference models.
- **Hardware debugging is humbling**: one word in a CMakeLists.txt (`INTERFACE` vs `PRIVATE`) cost hours. The buffer boundary bug was three interrelated issues masking each other.
- **Bare metal is freeing**: no OS overhead, no memory allocator surprises, every byte accounted for. 140 KB of RAM running a 600M parameter model — over 4000× its size.
