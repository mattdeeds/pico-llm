# Title

Running modern LLMs on low cost Microcontrollers
pico-llm: From 9M to 4B Parameters on a $1 Microcontroller

## Subtitle

What is possible for modern LLM inferencing on extremely underpowered hardware?

### Hook

Running a modern LLM locally at "realtime" speeds requires expensive hardware and I don't have a B200 sitting at home. But what if you didn't care about tokens per second? What if getting a response from the model at the lowest possible power consumption was your only concern?
Imagine a not so distant future where an ASI (Artificial Super Inteligence) model exists and can solve problems better than any human. You could argue that it would be imparitive to run this model no matter the cost, even on a low cost, low power MCU. In this situation you could happily wait hours/days/years for the answer to a civilization scale query. The power a system like this would consume would also be in the milliwatt range so powering it for years would be trivial. The stakes in this scenario are much higher than asking ChatGPT to proofread an email to a coworker but lets take this idea an run with it. If you had to run a LLM on something like your bluetooth speaker or smartbulb what performance could you expect and how would you do it with no operating system, no frameworks, and no GPU? 

### The Setup

Our target is the RP2350, a dual core Cortex-M33 running at 150 MHz with 520 KB RAM. Specifically the Raspberry Pi Pico 2 dev board. For scale, that's around 15,000 times less RAM than your iPhone. Not to mention all of the dedicated GPU and Neural Accelerator hardware built into an iPhone processor that is built for exactly this workload. The situation is not all bad though, the RP2350 is still decently performant in terms of low cost microcontrollers and many thousand of times more powerful than something like the Apollo Guidance Computer from the 1960s. Still, running a extremely computationally intensive tasks that we typically use million dollar GPU clusters for will involve overcoming a lot of roadblocks.

Our first and biggest obstical is that any modern LLM will not physically fit into the limited RAM of the RP2350. The  smallest models AI Labs are releasing are in nearing or in the GB range. So clearly this is impossible, right? Not necessarily.
On a very basic level, the two operations we need to run or "inference" an LLM are loading the model weights into memory and running a simple math operation on that data. We typically load an the model into the massive high speed RAM of a GPU or a some other AI accelorator then we use the immense parallelization of the hardware to do all those math operations simultaniously. We hardly have any RAM and we only have two cores.
The solution: stream everything. The inference process consumes the model weights sequentially and for each token output by the model each layers weights are used once in order.
If we serialized the model weights in consumption order, we can stream them through a tiny buffer in RAM. So for every token we generate we are only every holding a fraction of the model in RAM.
That still leaves us with the question, where are we storing the model that we are streaming from? There are many different options available to us especially since we are using a microcontroller and programming on the bare-metal. Using the PIO peripheral on the RP2350 (sort of like a programmable hardware interface) we could use eMMC memory, an SD card, NAND/NOR Flash, or even a USB drive. All have their pros and cons but for this project I decided on a microSD card. They are cheap, ubiquitous, relatively high speed, and high capacity. Most importantly someone has already written a great opensource library for 4-bit SDIO mode access on the RP2350. With the standard configuration this should get us at least data transfer rates close to 25 MB/s. Very important if we don't want to be IO bound when streaming the model.
With the memory roadblock taken care of we can move on to building the inference engine.

### The Engine

Like I mentioned up top, we have no OS or frameworks to help us so we'll need to build the complete infrence pipeline ourselves. We won't even have a filesystem on the SD Card. But this also means no OS, framework, or filesystem overhead to slow us down.
We'll take inspiration from Andrej Karpathy's llama2.c project, a single-file C inference engine. Our inference engine will also be written in C and will similarly implement the standard llama style transformer. This way we should easily be able to run various opensource models on our engine.
For those interested in the deatils the engine follows the three-phase per-layer pipeline below:
1. Stream the attention weights from the microSD card -> Q/K/V projections with tiled int8 matmul -> RoPE
2. Pause streaming -> write the KV cache to the micrSD card -> online softmax attention
3. Resume streaming -> output projection -> SwiGLU FFN 
An important note from the pipeline, the KV cache lives on the SD card as well. The RP2350 RAM only holds the current activation vector.
**MAYBE EXPLAIN MORE**

On the hardware side its pretty simple at this point, just the Pico 2 dev board wired to the microSD card through a SD card adapter on a breadboard. Programming and communication are both done over USB for simplicity.

### A Model to Test with

We have our inference engine, our hardware setup, but no model to validate the streaming inference engine. From a modern model like Qwen3, I am expecting to get token output speeds of minutes per token and it's not very easy to test the full pipeline when you could be waiting hours for each complete response to be output.
To get around this I trained a custom 9M parameter LLaMa-architecture model with 256 dimenstions, 6 layers, 4 heads, and 8k vocab. This model was trained on the TinyStories dataset for 15k steps on my MacBook. It's large enough to produce coherent english text which is all we really need for this test and it shold be "relatively" fast on our hardware.
This first 9M param model is not just a validation for the inference engine but the model export tooling as well. As I alluded to in the setup, our model can't be formated in any of the standard formats, we'll write the model weights to the raw SD card serially in consumption order as an int8 quantized binary. That just means we want **FIX THIS AND EXPLAIN BETTER**...

The result is ~15MB of data on the SD card and a test of the model on my laptop verifies the export is 100% argmax agreement with the original. It's time for testing on the real hardware and I immediately run into an issue. The SD card won't get pass initalization. Lets call it signal integraty issues trying to run this on a breadboard. I designed and ordered a custom PCB which essentially just interfaces the Pico 2 board with an microSD card slot. This would be a critical step for acheiving the data transfer speeds we need from the card.
After some time debugging integration isues, our first success! Inference is running but the only output is token 8190. At this point I don't have a tokenizer running, the model is just outputting raw token values. It turns out the logits were 1000x too large **EXPlAIN**...

After fixing that bug and a few more related to weight streaming into our RAM buffers, conherent english text generation was finally achieved. "Once upon a time there was..." generated at around 1 sec per token. At this point it finally felt like I was making some progress. But with progress comes the opportunity for optimization. How much performance are we leaving on the table? And while we don't have the thousands of cores of a GPU we do have more than more core.

### Going Dual-Core

Right now the firmware was operating purely sequentially. One core was handling the SD card reads and the matmul compute one after the other and each waiting for the other to finish. But the RP2350 is a dual-core chip. What if we split the workload and Core 0 owns all the SD card I/O and Core 1 runs the matmul? That way while Core 1 multiplies the current weight tile, the DMA fills the next RAM buffer.
Now we're really cooking. This gives us around a 1.5x speedup taking us from ~1 sec/token to ~0.7 sec/token on the 9M param model.
This speedup may seem modest now but when we are waiting minutes for a token to be generated any increase helps.
It would also be helpful if instead of just raw token values like <1234> being output, we had more of a chatbot style interface for interacting with the model. This involved implementing a greedy longest match tokenizer for prompt encoding, an on-device BPE token decoder for printing readable text back over USB serial, and wrapping this all in an interactive loop. This resulted in being able to type a prompt over USB serial, and getting streamed text output back in real time.
A working LLM chatbot on a microcontroller! But with only 9M parameters and very simple training data, it's not a "real" LLM that is actually useful, it's just a proof on concept.

### Qwen3-0.6B - A Real Open-source Model

With the inference engine and tooling validated it was time to aim higher and the Qwen3 family of models was a great target. First they are opensource, relatively new, and fits the same standard transformer architecture of the testing model with some minor differences that will become important later. Qwen3-0.6B features the same SwiGLU + RMSNorm + RoPE stack but with ~66 times the parameter count. We are going from ~15 MB on the SD card to 570 MB, 256 dimensions to 1024, and 8K vocab to 151K. This is a real model that can answer questions and talk to you like a human. It still very small compared to proprietary models from OpenAI or Antropic but it's a stepping stone to something bigger.

The first wall we hit was the 608 KB logits problem. float logits[151936] = 608 KB of memory which exceeds the chips entire 520 KB of RAM. We can't even store the logits on device, let alone have room in the RAM needed for any computation. The solution to this problem is the same as the rest so far, streaming. We can stream argmax from the SD card and process the 151K-row classifier in 256-row chunks. This rsesults in zero bytes of logit storage in RAM leaving our total RAM usage so far around 150 KB.

The remaining architectural differences... **EXPLAIN HERE**

The model export and formatting tool does a int8 quantization to the base model before output resulting in a 82% argmax agreement. So the archtecture port is correct but we are getting some noise from the quantization. It's not something that should stop us now but something to keep in mind for the future.

With the new firmware flashed the Qwen3 model loaded on the SD card all that was left to do was wait. 55 seconds later a token appeared in the serial terminal. A real 600M parameter model was running on a microcontroller and outputing "real" text. Real in quotes because the text was repetitive and confusing. Not what I expected from a model this much larger than the text model. Surely the output should be better than this?
It turns out we are missing something that all LLMs use to sound more human, temperature sampling. The current greedy decoding produces repetitive loops. We need temperature sampling and a repetition penalty to get a non-repetitative, varied output. Standard temperature sampling needs the full logits vector but we already know we cant store that in memory. 

