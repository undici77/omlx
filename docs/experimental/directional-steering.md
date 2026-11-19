# Directional steering on Qwen3.8 Flash Next: results

## TL;DR

A verbosity pack cuts answer length 30–46% with every quality check inside its gate; the default scope costs no measurable decode speed and about 2 points of MTP acceptance, steering every token about 5% decode. User guide: [steering.md](../steering.md).

## Setup

| Item | Value |
|---|---|
| Machine | MacBook Pro 14" M4 Max, 128 GB |
| Model | Qwen3.8 Flash Next, oQ4e + MTP (`qwen4_exp`, 48 layers) |
| Memory mode | PLE n-gram table (79.5 GB) on SSD; it does not fit next to the 70.9 GB of weights |
| Pack | `--preset verbosity`: 100 DS4 succinct/verbose pairs, attention branch, `project` method |
| Pack stability | split-half cosine 0.92–0.99 on all 48 layers |
| Sampling | temperature 0 |
| Runs | three unattended campaigns on 2026-11-18/19; decode gates with MTP off |

## Effect and quality

`omlx steer verify --quick`, scope `prefill+k`, against the unsteered model (strength 0).

| Strength | Mean words | Effect | KL (nats/token) | NLL change | Repeated 4-grams | Valid JSON | Gate |
|---|---|---|---|---|---|---|---|
| 0 | 114.5 | — | 0 | — | 0.85% | 100% | reference |
| -0.5 | 80.1 | **-30%** | 0.021 | -0.29% | 0.36% | 100% | pass |
| -1 | 61.8 | **-46%** | 0.045 | +0.16% | 0.14% | 100% | pass (KL near the 0.05 limit) |

Shorter answers did not turn repetitive: the repeated 4-gram rate went down. The quality gate wrote `{"attn": -1.0}` at `prefill+k` into the pack as the recommended setting.

## Decode cost per scope

Decode tok/s at strength -1, compared with the pack installed at strength 0, which costs nothing (see the last row). Positive = slower. Baseline about 59 tok/s with MTP off.

| Scope | Run 1 | Run 2 | Run 3 | Mean words |
|---|---|---|---|---|
| `prefill+k` | +0.16% | +0.42% | +0.16% | 78 |
| `all` | +4.92% | +5.02% | +5.11% | 42 |
| `all` at strength -0.0001 (hook cost, text unchanged) | — | — | +3.89% | 100 |
| `prefill` | +2.54% | +3.02% | +4.55% | 95 |
| pack at strength 0 vs no pack | -0.83% | -1.05% | -0.59% | 102 |
| run-to-run noise (same config twice) | ±0.57% | ±0.58% | ±1.50% | — |

| Finding | Reading |
|---|---|
| `prefill+k` is free | The hooks pass through once the first K=32 tokens are generated |
| `all` costs ~5%, ~4% of it with the text unchanged | One extra small GPU op per layer per token (48 per step) outside the model's compiled decode paths; fusing it into those paths should remove most of it |
| `prefill` reads slower although it does no decode work | Not hook cost. On this machine decode speed depends on which tokens are generated, because the PLE table is read from SSD; not resolved |

## MTP

Draft acceptance from the server log, MTP on, scope `prefill`, strength -1.

| Run | Without pack | Steered | Change |
|---|---|---|---|
| 1 | 73.2% | 70.9% | -2.3 points |
| 2 | 72.8% | 70.6% | -2.3 points |
| 3 | 73.3% | 71.2% | -2.1 points |

The MTP speedup ratio falls more (about 1.28× → 1.15×), but that ratio is confounded by length: steered answers average 60 tokens vs 153, and MTP gains less on short requests.

## Measurement notes

| Issue on this setup | How the numbers above handle it |
|---|---|
| Shorter steered answers make tokens/wall-clock look slower | Equal-length decode (`ignore_eos`) and the server's own generation rate |
| PLE pages read from SSD depend on the generated tokens | An untimed pass decodes the same tokens first |
| Slow drift over a 40-minute run (up to -5%, macOS thermal state "nominal") | Each scope is compared with the mean of references measured just before and after it |
| Prompt-cache hits inflated prefill speed | A unique prefix per prefill request |
| Prefill noise stays ±6–7% (GPU-bound, chassis throttling) | Prefill gates are not reported; the prefill hook adds one op per layer per chunk |

## Open questions

| Question | Why it matters |
|---|---|
| Is a fixed engine-wide pack useful, or is per-request steering (`extra_body`) needed? | The scale machinery is already per row; the API is the missing part |
| Which behaviours beyond verbosity are worth packs (refusal style, formality, language)? | Only verbosity was measured |
| Which other architectures should come next? | Adapters for Qwen3.5/3.6 and Llama-style models exist but are disabled until measured like this |
| Is ~5% acceptable for `all`, or should fusing the hook into the compiled decode path come first? | `prefill+k` already gives most of the effect for free |
