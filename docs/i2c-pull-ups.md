# I2C Pull-up Resistor Selection

The board has 10k pull-ups on both I2C buses (I2C0 and I2C1). This works for Standard Mode (100 kHz) but may be too weak for Fast Mode (400 kHz), especially with cables on the JST-SH connectors adding capacitance.

If you experience communication errors at 400 kHz, swap the pull-ups to 4.7k or 2.2k.

| I2C Speed | Recommended Pull-up |
|-----------|-------------------|
| 100 kHz (Standard) | 10k (current value) |
| 400 kHz (Fast) | 4.7k |
| 1 MHz (Fast-Mode Plus) | 2.2k |
