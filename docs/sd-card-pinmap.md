# Micro SD Card to RP2350 Pin Map (SDIO 4-bit Mode)

## Pin Connections

| Micro SD Pin | SD Name | RP2350 GPIO | Function       |
|:------------:|:-------:|:-----------:|:---------------|
| 1            | DAT2    | GPIO 14     | Data Line 2    |
| 2            | DAT3    | GPIO 15     | Data Line 3    |
| 3            | CMD     | GPIO 11     | Command        |
| 4            | VDD     | 3.3V        | Power Supply   |
| 5            | CLK     | GPIO 10     | Clock          |
| 6            | VSS     | GND         | Ground         |
| 7            | DAT0    | GPIO 12     | Data Line 0    |
| 8            | DAT1    | GPIO 13     | Data Line 1    |

## Notes

- Data lines D0-D3 (GPIO 12-15) must be consecutive GPIOs for PIO-based SDIO.
- All signal pins use fast slew rate and 8 mA drive strength.
- SDIO runs on PIO1, state machine 0, with DMA channels 4 and 5.
- Default clock speed: 50 MHz (high-speed mode), up to 75 MHz overclocked.

## Wiring Diagram (top view of micro SD connector)

```
         ┌─────────────────────┐
         │ micro SD connector  │
         │  (contact side up)  │
         │                     │
   Pin 1 │  DAT2  ── GPIO 14  │
   Pin 2 │  DAT3  ── GPIO 15  │
   Pin 3 │  CMD   ── GPIO 11  │
   Pin 4 │  VDD   ── 3.3V     │
   Pin 5 │  CLK   ── GPIO 10  │
   Pin 6 │  VSS   ── GND      │
   Pin 7 │  DAT0  ── GPIO 12  │
   Pin 8 │  DAT1  ── GPIO 13  │
         │                     │
         └─────────────────────┘
```

## Configuration Reference

Defined in `src/sdio_rp2350_config.h`.
