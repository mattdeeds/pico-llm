# Optimization Ideas

Current performance: ~38 s/tok (Bonsai-1.7B Q1_0_g128, RP2350 @ 200 MHz, 50 MHz SDIO)

## Recommendation

**Qwen3-0.6B at Q4 + 250 MHz system clock** is the highest-impact next move.
Estimated ~25 s/tok with noticeably better output quality than Bonsai Q1.

The key insight: 1-bit quantization reduces storage but not compute. Bonsai has
2.8x more parameters than Qwen3-0.6B, so despite smaller on-disk size, compute
is higher. And 1-bit quality is substantially worse than int4 or int8.

| Model | Quant | Size | Est. I/O | Est. Compute | Est. Total | Quality |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Q8 | 570 MB | ~44s | ~16s | ~60s | Best |
| Bonsai-1.7B | Q1_0 | 200 MB | ~9s | ~29s | ~38s | Worst |
| **Qwen3-0.6B** | **Q4** | **~285 MB** | **~22s** | **~10s** | **~32s** | **Good** |
| Qwen3-0.6B | Q4 + 250MHz | ~285 MB | ~18s | ~7s | **~25s** | Good |

Qwen3-0.6B Q4 wins on both speed AND quality. Fewer parameters means fewer
MACs, int4 preserves far more model quality, and storage is only 40% larger
than Bonsai. We already have the full Qwen3 export pipeline working.

## Model choice: why not 1-bit?

Current bottleneck breakdown at ~38 s/tok (Bonsai-1.7B Q1_0):
- ~9 s I/O (streaming ~200 MB at 50 MHz SDIO)
- ~29 s compute (1.7B sign-bit MACs across 28 layers)

Compare Qwen3-0.6B at int8 (~60 s/tok):
- ~44 s I/O (streaming ~570 MB)
- ~16 s compute (0.6B int8 MACs across 28 layers)

The ideal is to minimize BOTH storage and param count. Extreme quantization on a
large model gives small storage but lots of compute. Moderate quantization on a
small model gives moderate storage but little compute — and much better quality.

### Natively trained 1-bit models

Post-training 1-bit quantization destroys quality. Models trained natively with
ternary/binary weights (like Microsoft's BitNet b1.58) maintain quality much
better because training accounts for quantization. Worth watching for open-weight
BitNet models in the 0.5–1B range, but none are available as of mid-2025.

## I/O optimizations

### SDIO_HIGHSPEED_OVERCLOCK (75 MHz)

The SDIO_RP2350 driver has an experimental `SDIO_HIGHSPEED_OVERCLOCK` mode that
runs the SD bus at 75 MHz — 50% above the SDR25 spec. Out-of-spec, so expect
CRC errors on some cards or wiring configurations. Worth testing with the 12 mA
drive strength already in place. If CRC errors appear, could try 16 mA or
shorter traces on a future PCB rev.

DDR50 (dual data rate at 50 MHz, 1.8V) is NOT supported by the SDIO driver. It
would require 1.8V signaling hardware, new PIO programs for dual-edge clocking,
and CMD6 mode 4. Not feasible without a PCB redesign.

### Reduce SD reads per token

Each token currently streams all model weights from SD. Possible approaches:

- **Layer caching**: keep the smallest layers (embedding, final norm, classifier)
  in RAM permanently. Final norm is already cached; embedding and classifier
  together are ~44 MB — won't fit in 520 KB SRAM.
- **KV-cache on SD**: already done (KV cache lives on SD). No further win here.
- **Speculative skipping**: skip layers whose contribution is below a threshold.
  Risky for quality but could cut reads substantially. Needs research.
- **Weight deduplication / sparse streaming**: if weight blocks repeat or are
  mostly zero, skip reading them. Unlikely with 1-bit quantization since every
  bit matters. More interesting at int4/int8 where structured sparsity exists.

## Compute optimizations

### System clock overclock (250+ MHz)

RP2350 runs at 150 MHz stock, currently set to 200 MHz. Many users report stable
operation at 250–300 MHz. Benefits:

- Directly speeds up matmul kernel (cycles/weight stays same, but cycles are
  faster). 250 MHz = 25% speedup, 300 MHz = 50%.
- Also speeds up SDIO PIO clock if the divider stays the same (but may need
  adjustment to keep SDIO at 50 MHz spec)

Risks: instability, increased power draw, flash XIP timing may need adjustment
(flash divider). Test with `set_sys_clock_khz(250000, true)` and run a long
generation to check for corruption.

### Core 0 helping with compute

Currently Core 0 manages DMA while Core 1 does matmul. But Core 0 is mostly
idle waiting for DMA to finish. It could compute on half the tile rows while
waiting. Potentially ~1.5–1.8x compute throughput. Complexity: need to split the
accumulator and synchronize carefully to avoid contention on the FIFO.

### Further kernel optimization

Current DSP kernel runs at ~4.07 cycles/weight. Potential next steps:

- **Inline ASM for the full inner loop**: bypass compiler entirely for the
  hottest 20 instructions. Could save 0.5–1.0 cyc/wt from better register
  allocation. We already proved GCC can't fold ROR into SXTAB16.
- **DMA prefetch of weight buffers**: overlap SD DMA with compute on the
  previous tile. Already using double-buffering, but timing could be tighter.
- **int4 activation quantization**: reduce activation precision from int8 to
  int4, halving the activation bandwidth and enabling packed SIMD on twice as
  many values per instruction. Needs quality evaluation.

### Float32 → fixed-point for non-matmul ops

All non-matmul code in `transformer.c` uses float32 for RMSNorm, RoPE, SiLU,
and softmax. The M33 has a single-precision FPU so it's not terrible, but these
operations run 28 times per token (once per layer). Fixed-point Q16.16 would be
faster for the simpler arithmetic ops. Lower priority since matmul dominates.
