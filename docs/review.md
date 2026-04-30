# Review of `post.md`

Honest feedback on the rough draft. I read both `post.md` and the `blog-outline.md` it's based on, so I'll also flag where the draft is missing material the outline promises.

## TL;DR

The draft has a genuinely interesting story and a conversational voice that works. But right now it reads like a first pass — it stops roughly 40% of the way through the planned arc, it's carrying a lot of typos and small grammar errors, and it hasn't yet committed to the strongest version of its narrative. The biggest structural issue isn't length: it's that the **"plot twist"** the outline sets up (going *back* to Qwen3-0.6B with Q4_0 beats the bigger 1-bit model) is the most interesting idea in the whole post, and the current draft doesn't foreshadow it at all.

---

## Structural / Narrative Feedback

### 1. The draft is unfinished

It ends mid-thought at line 70, during the "we can't store the full logits" problem. The outline has 17 sections; the draft covers roughly sections 1–11. Everything after Qwen3-0.6B — Gumbel-max sampling, Qwen3-4B with Q4_0, Bonsai-1.7B at 1-bit, the Q1_0 DSP kernel work, **the plot twist**, the summary table, and the "what I learned" wrap-up — is missing. That's a lot, and several of those are the most interesting parts.

There are also four explicit `**EXPLAIN**` / `**FIX THIS**` placeholders still in the text (lines 35, 43, 46, 64).

### 2. The opening has too much scaffolding

The current top reads:

```
# Title
Running modern LLMs on low cost Microcontrollers
pico-llm: From 9M to 4B Parameters on a $1 Microcontroller

## Subtitle
What is possible for modern LLM inferencing on extremely underpowered hardware?

### Hook
...
```

Those `# Title`, `## Subtitle`, `### Hook` headers look like a template that was never stripped. Pick a real H1 title, drop the subtitle or fold it into a dek line under the title, and delete the "Hook" header (the opening paragraph just *is* the hook).

Also: you have two title candidates stacked on top of each other. Commit to one. "pico-llm: From 9M to 4B Parameters on a $1 Microcontroller" is punchier and more specific than "Running modern LLMs on low cost Microcontrollers."

### 3. The hook is intellectually interesting but asks a lot of the reader

The ASI-on-a-smartbulb thought experiment is a cool frame, but it takes the whole first paragraph to set up, and the reader has to accept several "imagine this future" premises before they know what the post is actually about. I'd consider one of two moves:

- **Shorter hook, concrete result upfront.** Lead with the punchline: "I ran a 600M-parameter LLM on a $1 microcontroller with 520 KB of RAM. It generates one token every 15.5 seconds. Here's how." Then use the ASI framing as a one-sentence philosophical aside, not the load-bearing motivation.
- **Or keep the hook but tighten it.** Three sentences instead of seven. The bluetooth-speaker / smartbulb line is the best part — move it earlier.

Either way, the reader should know within the first ~100 words that (a) this is a real thing you built, (b) it runs real models, and (c) there's a surprising result at the end. Right now (c) is entirely absent.

### 4. The "climbing the ladder" frame from the outline isn't in the draft

The outline's spine is: 9M toy → 0.6B real → 4B reasoning → 1.7B 1-bit → *back to* 0.6B Q4_0 as the winner. That's a great structure — it's a series with an unexpected ending. The draft follows the first steps but never tells the reader "we're going to climb a ladder," so each new model feels like a separate chapter rather than part of an arc. Even one sentence in the intro ("I climbed a ladder of models — each one hit a new wall — and the fastest result came from the step I least expected") would massively improve the reader's sense of where they are.

### 5. Foreshadow the plot twist

Related to above: the single most interesting claim in the outline is that *going smaller with better quantization beats going bigger with extreme quantization*. This is counterintuitive and genuinely insightful. The current draft gives no hint this is coming. A one-line teaser at the end of the intro, or at the end of the first Qwen3-0.6B section, would pay off enormously when the twist lands.

### 6. Pacing: the engine section is dry, the model sections are livelier

Section "The Engine" drops into a numbered pipeline list with phrases like "tiled int8 matmul → RoPE" that assume the reader already knows what RoPE is. The rest of the post is written for a curious generalist. Either commit to the generalist voice (brief one-line glosses: "RoPE, the positional encoding trick most modern transformers use") or commit to the deep-dive voice throughout. Right now the register shifts.

The `**MAYBE EXPLAIN MORE**` note at line 35 is right — this section needs either more detail *or* less. The middle ground it's currently in is the worst of both.

### 7. The "first light" moment is underplayed

"Once upon a time there was..." generated at 1 sec/token on a $1 chip you hand-soldered onto a custom PCB is a genuinely dramatic moment. The draft dispatches it in half a sentence. This is the emotional peak of the first half of the post — linger on it. What did it feel like? How long had you been debugging? The INTERFACE-vs-PRIVATE and buffer-boundary bugs from the outline are exactly the kind of war stories that make a technical post memorable, and they're not in the draft yet.

### 8. The Qwen3-0.6B section introduces a problem then stops

The draft builds up to "we can't even store logits — we need temperature sampling — standard temperature sampling needs the full logits vector but we already know we cant store that in memory" and then just ends. The Gumbel-max trick is the payoff and it's missing. If you have to stop writing mid-draft, stop at a section boundary, not a cliffhanger.

---

## Grammar, Spelling, and Line Edits

There are enough typos that a full pass with spellcheck is worth doing before the next review. Non-exhaustive list of what I noticed:

**Spelling:**
- "imparitive" → imperative (L13)
- "obstical" → obstacle (L19)
- "accelorator" → accelerator (L20)
- "simultaniously" → simultaneously (L20)
- "micrSD" → microSD (L32)
- "infrence" → inference (L28)
- "deatils" → details (L30)
- "dimenstions" → dimensions (L42)
- "shold" → should (L42)
- "formated" → formatted (L43)
- "initalization" → initialization (L45, and "pass" should be "past")
- "integraty" → integrity (L45)
- "isues" → issues (L46)
- "acheiving" → achieving (L45)
- "conherent" → coherent (L48)
- "outputing" → outputting (L68)
- "archtecture" → architecture (L66)
- "Antropic" → Anthropic (L60)
- "rsesults" → results (L62)
- "non-repetitative" / "repetative" → non-repetitive / repetitive (L69)
- "cant" → can't (L69)
- "opensource" → open-source (multiple) — both are accepted, but pick one and be consistent

**Grammar / wording:**
- "lets take this idea an run with it" → "let's take this idea and run with it" (L13)
- "a extremely computationally intensive tasks" → "extremely computationally intensive tasks" (L17, drop "a", and the sentence structure is off — "running [tasks] will involve overcoming roadblocks" has a subject/verb issue)
- "The smallest models AI Labs are releasing are in nearing or in the GB range" → "The smallest models AI labs are releasing are nearing, or already in, the GB range." (L19)
- "running a simple math operation on that data" → inference is actually many operations; "performing a lot of simple math" is more honest (L20)
- "we typically load an the model" → drop "an" (L20)
- "for each token output by the model each layers weights are used once in order" → needs a comma and an apostrophe: "...for each token the model outputs, each layer's weights are used once, in order." (L21)
- "we are only every holding" → "only ever holding" (L22)
- "someone has already written a great opensource library" — name them and link it; credit matters and it also lends credibility (L23)
- "more than more core" → "more than one core" (L48)
- "proof on concept" → "proof of concept" (L56)
- "It still very small" → "It's still very small" (L60)
- "Surely the output should be better than this?" — question mark is fine, but the preceding sentence ends with a period where it should be a question too (L68)

**Punctuation / style:**
- Many comma splices in the longer paragraphs. E.g. L17: "...for scale, that's around 15,000 times less RAM than your iPhone. Not to mention all of the dedicated GPU and Neural Accelerator hardware built into an iPhone processor that is built for exactly this workload." The second "sentence" is a fragment, and "built... built" repeats.
- "Neural Accelerator" — no need to capitalize; it's not a proper noun.
- "It's large enough to produce coherent english text" — "English" is capitalized.
- Hyphenate compound modifiers: "low-cost microcontroller", "million-dollar GPU clusters", "real-time speeds".
- "realtime" → "real-time" (L12)
- En-dashes vs hyphens: for ranges like "minutes per token" vs "hours for each complete response," consistency is fine, but "1 sec/token" and "~0.7 sec/token" mix units casually — pick "s/tok" or "sec/token" and stick to it (the outline uses "s/tok", which is cleaner).

---

## Content / Technical Notes

A few places where the technical framing could be sharper:

- **L13: "power consumption in the milliwatt range"** — this is a strong claim and it's unsupported. The RP2350 at 150 MHz draws more like 50–100 mW under load; an SD card in active read mode adds another 100+ mW. "Low power" is fair; "milliwatt range" is optimistic. Either soften or cite.
- **L17: "15,000 times less RAM than your iPhone"** — good comparison, but "less RAM" is awkward for ratios; "1/15,000th the RAM" is cleaner.
- **L20: "the two operations we need to run or 'inference' an LLM are loading the model weights into memory and running a simple math operation on that data"** — this oversimplifies to the point of being misleading. Loading weights isn't an "operation" of inference; it's a prerequisite. The real framing you want is: "inference is just a long chain of matrix multiplies, and the weights for each multiply are used exactly once per token." That's the insight that makes streaming work.
- **L23: the list of storage options** (eMMC / SD / NAND / NOR / USB) is a detour. You picked microSD; you can say "I considered a few options and picked microSD because X, Y, Z" in one sentence.
- **L31–33: the three-phase pipeline** is presented as a bare list without explanation of *why* it's three phases. The answer is "because we can't stream and write the KV cache at the same time" — that reasoning is interesting and it's not stated.
- **L34: "the KV cache lives on the SD card as well"** — this is a wild claim (most people expect KV cache in RAM) and deserves to land harder. Two sentences on why this works and what it costs.
- **L46: "the logits were 1000x too large"** — this is the fun bug; tell the reader what was actually happening. The outline mentions the buffer-boundary bug separately; make sure you disambiguate.
- **L69: "something that all LLMs use to sound more human, temperature sampling"** — temperature sampling doesn't really make LLMs sound human; it makes them sound *less deterministic*. The human-sounding part is the training data. A cleaner framing: "greedy decoding gets stuck in loops; we need sampling to break ties."

---

## Summary: What I'd Do Next

In order of impact:

1. **Finish the draft.** Sections 12–17 from the outline are missing, including the plot twist that anchors the whole post.
2. **Foreshadow the twist in the intro.** One sentence. The post is much stronger if the reader knows there's a surprise coming.
3. **Spellcheck pass.** There are enough typos that they distract; most are catchable automatically.
4. **Strip the template scaffolding** (Title/Subtitle/Hook headers) and commit to one title.
5. **Expand the "first light" moment and the INTERFACE / buffer-boundary bug stories.** These are the most human parts of the story and they're currently compressed into half-sentences.
6. **Decide on a technical register** (generalist vs. deep-dive) and apply it consistently.
7. **Resolve the four `**EXPLAIN**` placeholders** — each one is in a spot where the reader would genuinely want more detail.

The foundation is good. There's a real story here, with a real twist, and you have the numbers to back it up. It mostly needs finishing and polishing, not restructuring.
