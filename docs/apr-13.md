Session Summary

  Hardware bring-up: complete
  - Fixed SDIO library pin config (INTERFACE → PRIVATE) — was using GPIO 34-39 instead of 10-15
  - Enabled USB CDC serial for debug output
  - SD card initializes successfully (SDHC, 25 MHz standard speed)
  - Post-write CMD13 busy wait eliminates CMD18 timeouts

  Inference engine: working correctly
  - Single-core mode: ~1 sec/token, 33 tokens generated
  - Output is coherent English text (TinyStories-style)
  - Fixed weight stream buffer-boundary bug (see below)

  Bugs fixed this session:
  1. Weight stream misalignment at buffer boundaries (ws_tiled_matmul)
     - When a weight row spanned two 32KB buffers, the partial tail was
       discarded and the next buffer was read from a wrong offset
       (skip = sd_off % 512 didn't match the prefetch start address)
     - Fixed by falling back to ws_read_bytes for rows that span boundaries
     - Also fixed ws_drain to re-read from sd_off instead of loading a
       stale prefetch buffer (fixes Phase 1→3 transition data loss)
     - Also clamped weightbuf_start_prefetch block_count to prevent
       buffer overflow when sd_byte_offset is not block-aligned

  2. High-speed SDIO (50 MHz): CMD6 switch works but CRC errors on data
     - CMD6 SWITCH_FUNC to SDR25 succeeds, verification single-block read passes
     - Multi-block reads at 50 MHz get CRC errors — likely PCB signal integrity
     - Code is written (ifdef'd out), needs hardware fix (termination resistors
       or shorter traces on D0-D3)

  3. Core 1 SDIO prefetch: PIO commands hang from Core 1
     - CMD18 issued from Core 1 never returns (PIO state machine stall)
     - DMA IRQ handler race (registered on Core 0) may contribute
     - Disabling DMA IRQ didn't help — CMD18 itself hangs before DMA starts
     - Root cause likely in SDIO_RP2350 library PIO init being Core 0-specific

  Current performance: ~1 sec/token (single-core, 25 MHz SDIO, 6-layer model)
  Breakdown (estimated): ~500ms SD reads + ~500ms compute

  Next session priorities:
  1. Performance: overlap IO and compute via single-core async DMA
     (start CMD18+DMA, compute on previous buffer, poll for completion)
     — OR use Core 1 for matmul compute while Core 0 handles all SD IO
  2. High-speed SDIO: investigate PCB signal integrity (scope D0-D3,
     check trace lengths, consider 33Ω series termination)
  3. Add on-device token decoding (tokenizer_decode is implemented but
     not called during generation)
  4. Prompt input via USB serial (currently hardcoded BOS token)
