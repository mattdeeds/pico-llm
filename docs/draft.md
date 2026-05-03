# pico-llm: From 9M to 4B Parameters on a $1 Microcontroller

I ran a 600M parameter LLM on a $1 microcontroller with only 520 KB of RAM. It generates one token every 15.5 seconds — and the most interesting part is that the fastest result came from going *back* to a smaller model, not forward to a bigger one. I climbed a ladder of four models, each one hitting a new wall, and the answer was waiting at rung two with better tools.

The idea of running a real LLM on something with the performance and power budget of a Bluetooth speaker — with no operating system, no frameworks, and no GPU — consumed me. Not necessarily for practical reasons. Imagine a not-so-distant future where an ASI model can solve problems better than any human. You'd run it no matter the cost, even waiting days for the answer to a civilization-scale question. We don't live in that world yet, but the experiment is interesting regardless. Here's how I did it.

## The Setup

Our target is the RP2350, a dual-core Cortex-M33 running at 150 MHz with 520 KB of RAM. The chip itself is about a dollar in volume; I'm running it on the Raspberry Pi Pico 2 development board, which is closer to $5 once you add the USB connector, regulator, and headers. That's roughly 1/15,000th the RAM of your iPhone, and none of the dedicated neural hardware Apple builds into their chips for exactly this kind of workload. The situation isn't all bad though. The RP2350 is decently capable for a low-cost microcontroller, and because we're going bare-metal there's no OS or framework overhead eating into our budget.

The obstacle is that any modern LLM won't physically fit in 520 KB. The smallest models AI labs are releasing are nearing, or already in, the gigabyte range. A 600M parameter int8 model is 570 MB — a 1,000:1 ratio against our available RAM. At 4B parameters with Q4_0 quantization it's closer to 4,000:1. Clearly impossible in the traditional sense.

The key insight that makes everything work: LLM inference consumes model weights sequentially. For each token the model outputs, each layer's weights are used exactly once, in order. If we serialize those weights onto an SD card in consumption order, we can stream them through a small buffer in RAM — at any moment holding only a fraction of the model. We never need the whole thing at once.

For storage I went with a microSD card. They're cheap, ubiquitous, and someone has already written a great open-source library for 4-bit SDIO mode on the RP2350 that gets us close to 25 MB/s transfer rates — fast enough that we won't be completely I/O bound. More on that later.

## The Engine

No OS means we build the complete inference pipeline ourselves. We'll take inspiration from Andrej Karpathy's llama2.c — a single-file C inference engine — and implement the standard llama-style transformer in C on bare metal. We won't even have a filesystem on the SD card. But no OS, framework, or filesystem also means no overhead slowing us down.

The engine runs a 32 KB double buffer: while Core 1 computes on one tile of weights, DMA fills the next buffer. We never hold more than a small slice of the model in RAM.

The per-layer pipeline:

1. Stream attention weights from SD → Q/K/V projections with tiled int8 matmul → RoPE (the positional encoding most modern transformers use)
2. Pause streaming → write the KV cache to SD → online softmax attention
3. Resume streaming → output projection → SwiGLU feed-forward network

Step 2's softmax is "online" — single-pass, never materializing the full attention matrix. A normal implementation builds an N×N matrix of attention scores in memory and then softmaxes it. We can't afford that. Instead we walk over past tokens one at a time, keeping a running max and a running denominator, and rescaling on the fly. Standard trick, but load-bearing here: it's what lets the KV cache live on the SD card at all.

That second step is worth pausing on. The KV cache — the record of past tokens the model needs to stay coherent — lives on the SD card too. The RP2350 RAM holds only the current activation vector, around 1 KB for the small model. When I tell people this they're usually skeptical. It works because we write each KV entry once and read it back once during attention, then it's done. It's slow, but it's correct.

The hardware at this point is just the Pico 2 dev board wired to a microSD card adapter on a breadboard. That breadboard setup will cause problems.

## A Model to Test With

Before throwing a real open-source model at the engine I needed a way to validate the full pipeline quickly — a real model would mean waiting potentially hours between tests. So I trained a custom 9M parameter LLaMA-architecture model: 256 dimensions, 6 layers, 4 attention heads, 8K vocab. Trained on the TinyStories dataset for 15K steps on my MacBook. It's large enough to produce coherent English, which is all we need for validation.

The model export tool writes the weights directly to the raw SD card in consumption order as an int8 quantized binary — 14.7 MB of data. A NumPy reimplementation of the full forward pass validates the export byte-for-byte: 100% argmax agreement with the float32 reference. Time to test on real hardware.

The SD card wouldn't get past initialization. Signal integrity issues running at SD speeds on a breadboard. I designed and ordered a custom PCB — essentially just the Pico 2 interfaced to a microSD slot, nothing more. When the boards arrived, the card still wouldn't init. Hours of debugging traced it to a single word in a CMakeLists.txt: the SDIO library's pin definitions were declared `INTERFACE` instead of `PRIVATE`, so my project was silently using the library's example pins (GPIO 34–39) instead of my PCB's wiring (GPIO 10–15). One word, one rebuild, card initialized.

Then inference ran end-to-end and every single output was token 8190, with logits about 1,000 times too large at every position. The cause was subtler than it first looked: when a weight matrix row spanned the boundary between two 32 KB streaming buffers, the partial tail at the end of the first buffer was silently discarded. Every subsequent weight read was shifted by the size of that tail, corrupting the entire forward pass — and inflating logit magnitudes by orders of magnitude. The fix required three interrelated changes: a boundary-spanning row fallback, re-reading from the correct offset after draining the buffer, and clamping the prefetch so it wouldn't overflow. Brutal to diagnose because everything silently runs and produces wrong numbers with no obvious pattern.

After all of that: "Once upon a time there was a little girl..." — coherent English, around 1 second per token. On a custom PCB I had just soldered together, no OS, no frameworks, a $1 chip. It's hard to overstate how good that moment felt after the previous few days of debugging.

## Going Dual-Core

The firmware at this point was purely sequential. Core 0 handled SD reads, then waited while Core 1 ran matmul, then Core 1 waited while Core 0 did the next read. The RP2350 has two Cortex-M33 cores — we should be overlapping those.

The redesign: Core 0 owns all SD I/O and DMA, Core 1 runs all matrix multiplications. While Core 1 is multiplying the current weight tile, Core 0's DMA is already filling the next buffer. True pipeline overlap. I tried letting Core 1 also handle the KV cache SD writes to overlap those too, but the SDIO library is Core 0-specific and those commands just hung. Core 0 owns all I/O.

This gave us a 1.46× speedup — from ~1,010 ms/token to ~690 ms/token on the 9M model. Now we're really cooking. This looks modest at under a second per token, but when we're waiting minutes per token on a larger model every multiplier counts.

## Making It Interactive

Raw token IDs over USB serial aren't satisfying. I implemented an on-device BPE token decoder so the output streams as readable text, a greedy longest-match tokenizer for encoding prompts, and wrapped it all in an interactive loop: type a prompt over USB serial, get streamed text output back in real time.

A working LLM chatbot on a microcontroller. But with only 9M parameters and TinyStories training data, it's a proof of concept, not a useful tool. Time for the next rung.

## Qwen3-0.6B — A Real Open-Source Model

The Qwen3 family was the obvious target: open-source, recent, and using the same standard transformer architecture as our engine with some differences I'd need to handle. Qwen3-0.6B has 600M parameters, 28 layers, 1,024 dimensions, and a 151K vocabulary. We're going from 14.7 MB on the SD card to 570 MB, and from 8K vocab to 151K.

The first wall: `float logits[151936]` is 608 KB. The chip's entire RAM is 520 KB. We can't even allocate the logits array. The solution is the same as everything else here — stream it. We run the 151K-row output classifier in 256-row chunks, tracking a running argmax. Zero bytes of logit storage in RAM. Total RAM usage stays around 150 KB.

There were four architectural differences to handle compared to our 9M toy:

- **Grouped-query attention**: 16 query heads but only 8 KV heads — two query heads share each KV pair. The attention loop already handled arbitrary groupings.
- **QK-Norm**: per-head RMSNorm applied to Q and K before the attention scores. A small new operation.
- **Decoupled head dimensions**: Qwen3 uses head_dim=128 even though dim/n_heads=64, so the Q projection output is 2,048 rather than 1,024. Non-square weight matrices, but the tiled matmul handles arbitrary dimensions.
- **Split-half RoPE**: Qwen3 pairs element *i* with element *i + head_dim/2* for positional encoding instead of consecutive pairs. This one bit me — argmax agreement jumped from 25% to 82% after fixing it.

The model uses tied embeddings, meaning the embedding table and the output classifier are the same weights. This saves 590 MB of SD space. For tokenization the 151K vocabulary won't fit in RAM for an on-device tokenizer, so prompt encoding happens off-device via a Python serial helper.

With 570 MB written to the SD card and the firmware updated, I sent "Hello world" and waited. Fifty-five seconds later a token appeared. Then another. A real 600M parameter model — the kind of thing that's supposed to need a GPU — running on a chip with less RAM than a 1990s graphing calculator. The TinyStories model had been a proof of concept; this was the first time the project felt like it had actually arrived somewhere.

Then I read the output. "I'm going to be a part of... I'm going to be a part of..." forever.

## Sampling Without Logits

Greedy decoding — always pick the highest-probability token — gets stuck in loops. The fix is temperature sampling: add controlled randomness so the model doesn't always pick the same token. Standard temperature sampling needs the full probability distribution across all 151K tokens. We can't store that.

The Gumbel-max trick solves this elegantly. Sampling from `softmax(logits / T)` is mathematically equivalent to `argmax(logits / T + Gumbel_noise)`, where Gumbel noise is just `-log(-log(uniform(0,1)))`. The key is that this equivalence holds element-wise — we can add the noise during our streaming argmax computation, one chunk at a time. No full logit vector, no extra memory.

I also added a repetition penalty: tokens appearing in the last 64 outputs have their score reduced before comparison. The whole thing is configurable over USB serial with `/temp`, `/rep`, and `/greedy` commands. With sampling enabled the output immediately became varied and non-repetitive.

But Qwen3-0.6B is a base model. No instruction tuning, no chat capability. It would talk at you, not with you. One more rung.

## Qwen3-4B — A Reasoning Model

Qwen3-4B-Thinking is a 4B parameter reasoning-tuned model — roughly 7× the size of our 0.6B, fine-tuned for chain-of-thought reasoning. Architecturally it's identical to Qwen3-0.6B, just bigger: dim 2,560 vs 1,024, 36 vs 28 layers. The engine port is a compile-time constant change. The challenge is the weights.

Int8 quantization of Qwen3-4B is 3.9 GB — close to the uint32_t SD address limit, and around 6.5 minutes per token. We need better quantization.

Per-tensor int8 — what we'd been using — assigns one scale factor to an entire weight matrix. If a few outlier values dominate the range, all the small values get poor resolution. Fine for 9M parameters, marginal for 600M, likely broken for 4B.

llama.cpp's Q4_0 format solves this: weights are grouped into blocks of 32, each block with its own float16 scale. That's 18 bytes per 32 weights (4.5 bits/weight). Outliers in one block only affect that one block. The dequantization is: unpack two nibbles per byte, subtract 8 from each, multiply by the block scale. Integer math per-element, one float multiply per block.

After writing the export tool, matching ggml's exact nibble-packing layout byte-for-byte, updating the firmware kernel, and validating with NumPy: 75% argmax agreement. The result on hardware: a 2.16 GB model, 330 KB RAM, generating text from a 4B reasoning model on a $1 chip — at 3–6 minutes per token.

Technically it works. Practically it's unusable. If we want more capability, adding parameters isn't the right axis anymore. We need fewer bits per weight.

## Bonsai-1.7B — 1.125 Bits Per Weight

`prism-ml/Bonsai-1.7B` is a 1.7B parameter model trained end-to-end at 1 bit per weight. Not post-hoc quantized — the ±scale structure is baked into training. This makes it different from everything we'd done before. The "quantization error" is essentially zero because the model was designed to be 1-bit.

The reason to try this: every byte we read from SD is latency we pay. Going from Q4_0's 4.5 bits/weight to 1.125 bits/weight should cut SD reads by 4× without changing the parameter count. Speed is the wall now, not size.

The Q1_0_g128 format stores 128 weights per block: a 2-byte fp16 scale, followed by 16 bytes of sign bits (bit = 1 means +scale, bit = 0 means -scale). That's 18 bytes for 128 weights. PrismML ships the model as a GGUF file with a custom dtype that standard parsing tools don't know, so I wrote a 200-line Python parser to read it directly and transcribe the raw bytes into our binary format without requantization.

The new inner kernel computes `block_sum = 2 × Σ(x where sign bit = 1) − Σ(x)` to avoid per-bit branching. Validation: 100% argmax agreement, max absolute logit error 0.55 — compared to 9.57 for Q4_0 on Qwen3-4B. Training-time 1-bit quantization is dramatically cleaner than post-hoc.

The model is 232 MB on SD — smaller than Qwen3-0.6B at 570 MB int8, with 3× more parameters. First result: 62.7 seconds per token. The SD reads account for about 9 seconds of that; the remaining 53 seconds is compute — 1.7B multiply-adds per token at 200 MHz. For the first time, compute is the bottleneck, not I/O.

Two optimizations stacked for a 39% speedup:

**50 MHz SDIO** (62.7 → 46.8 s/tok): Issued CMD6 to switch the SD card into high-speed mode, bumped GPIO drive strength from 8 to 12 mA on the clock and data lines, and pinned the system clock to 200 MHz for an exact integer PIO divider. A previous attempt at this failed with CRC errors; the drive strength bump fixed it.

**ARM DSP intrinsics** (46.8 → 38.1 s/tok): The Cortex-M33 has packed-SIMD instructions that operate on four 8-bit or two 16-bit values at once — basically a poor-man's SSE for microcontrollers. Rewriting the inner loop to use them (`SEL` for branch-free byte selection driven by sign bits, `SXTAB16` for packed horizontal sums) cut the work per weight from 5.07 cycles to 4.07. One instruction needed inline assembly because GCC wouldn't fold an operand rotation that the hardware actually supports for free. I benchmarked three variants with the cycle counter to pick the winner.

Final rate: **38.1 s/tok** on Bonsai-1.7B. Faster than anything before it — but notice what just happened. Compute is now 29 of those 38 seconds. We traded the I/O bottleneck for a compute bottleneck.

## The Plot Twist: We Optimized the Wrong Thing

Here's the thing I didn't expect: Qwen3-0.6B at Q4_0 is **321 MB** on disk. Bonsai-1.7B is **232 MB**. The Q4_0 model is 40% larger. And yet it's 2.5× faster.

The 1-bit trick did exactly what it promised — it cut the bytes on disk by more than half compared to int8. What it didn't cut was the parameter count. Bonsai has 2.8× more parameters than Qwen3-0.6B, which means 2.8× more multiply-adds per token. Shrinking the file shifted the bottleneck from I/O to compute, and compute turned out to be worse. We optimized for the wrong thing.

Qwen3-0.6B at Q4_0 sidesteps this entirely. Fewer parameters means less compute. Q4_0's moderate quantization is good enough that quality holds up. And 321 MB at 50 MHz SDIO is only ~16 seconds of reads per token, versus Bonsai's ~9 seconds — a smaller I/O advantage than you'd expect, because Q4_0 is 4.5 bits/weight and not far behind Q1_0's 1.125 bits/weight in practice when the model itself is 3× smaller.

The Q4_0 export tool already existed from the Qwen3-4B port — just pointed at 0.6B. One wrinkle: initializing the SD card at 250 MHz caused ACMD41 timeout failures. Fix was to init at 200 MHz, then switch to 250 MHz after the card is ready. The SDIO PIO clock divider adjusts automatically (250/5 = 50 MHz bus speed).

The result: **15.5 seconds per token**. Better output quality than Bonsai. A bigger file, a simpler quantization scheme, and a faster clock — that's all it took to beat the 1-bit model we'd spent days optimizing.

## The Numbers

| Model | Quant | Bits/wt | Size | Sys clk | Speed | RAM | Bottleneck |
|-------|-------|---------|------|---------|-------|-----|------------|
| Custom 9M | int8 | 8 | 14.7 MB | 150 MHz | ~690 ms/tok | ~114 KB | SD I/O |
| Qwen3-0.6B | int8 | 8 | 570 MB | 150 MHz | ~55 s/tok | ~150 KB | SD I/O |
| Qwen3-4B | Q4_0 | 4.5 | 2.16 GB | 150 MHz | ~3–6 min/tok | ~330 KB | SD I/O |
| Bonsai-1.7B | Q1_0 | 1.125 | 232 MB | 200 MHz | ~38 s/tok | ~230 KB | compute |
| **Qwen3-0.6B** | **Q4_0** | **4.5** | **321 MB** | **250 MHz** | **~15.5 s/tok** | **~140 KB** | **SD I/O** |

## What's Next

**DSP-optimized Q4_0 kernel**: the current Q4_0 matmul is scalar C. The same ARM DSP approach I used for Bonsai's Q1_0 kernel (5.07 → 4.07 cycles/weight) should transfer directly — the inner loops have the same structure. That could push Qwen3-0.6B Q4_0 below 10 s/tok.

**Dual-core compute**: Core 0 is mostly idle during matmul, waiting on DMA. Processing half the weight tile rows in parallel on Core 0 while Core 1 handles the other half would give roughly 1.5× compute throughput with no hardware changes.

**300 MHz overclock**: 250 MHz is stable on the chips I've tested. Some RP2350 silicon reportedly runs at 300 MHz — another free ~20% if the chip cooperates.

**Natively trained small 1-bit models**: the ideal would be a BitNet-style model at 0.6B parameters — combining the compute advantage of fewer parameters with the I/O advantage of 1-bit storage. Nothing like this exists in the open-weight ecosystem yet.

## What I Learned

**You don't need the model in RAM — you just need it in order.** Sequential weight streaming makes a 1,000:1 RAM deficit irrelevant. The insight is that transformer inference is embarrassingly sequential at the per-token level: each layer's weights are consumed exactly once, in order. That's all you need.

**The bottleneck shifts, and optimizing for the wrong one makes things worse.** SD I/O dominated through Qwen3-4B. Bonsai's 1-bit trick cut the I/O bill — and handed it to compute, which turned out to be bigger: 232 MB on disk vs Qwen3-0.6B Q4_0's 321 MB, but 2.5× slower per token. Switching back to Q4_0 on a smaller model made I/O the bottleneck again, and that was the regime where the chip ran fastest. A model that's cheaper to read isn't necessarily faster to run. Figure out which resource is actually scarce before you start saving the wrong one.

**Quantization granularity matters more at scale.** Per-tensor int8 was fine for 9M parameters. Per-block Q4_0 was necessary at 4B. Bonsai showed that training-time 1-bit quantization produces a max logit error of 0.55 compared to 9.57 for post-hoc Q4_0. The closer quantization is to training, the cleaner the result.

**The Gumbel-max trick is beautiful.** Temperature sampling without storing logits — a one-line mathematical identity that works naturally in a streaming argmax loop. I keep thinking about other places this pattern applies.

**Hardware debugging is humbling.** One word in a CMakeLists.txt — `INTERFACE` vs `PRIVATE` — cost hours. The buffer boundary bug was three interrelated issues masking each other. Bare-metal debugging without memory sanitizers or print statements is a specific kind of pain.

**Bare metal is freeing.** No OS overhead, no framework surprises, every byte accounted for. 140 KB of RAM running a 600M parameter model — over 4,000× its size. The constraint forces clarity.
