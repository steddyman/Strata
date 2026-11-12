# Mac Pro 2019 on Linux: Radeon Pro Vega II Duo (gfx906)

This fork ([steddyman/Strata](https://github.com/steddyman/Strata)) adds hand-written gfx906 kernels and a gfx906 desktop
app (AppImage). It was built and measured on one machine, and the defaults are tuned for it:

| | |
| --- | --- |
| Computer | Mac Pro 2019 (MacPro7,1) |
| GPU | Radeon Pro Vega II Duo: two Vega 20 (gfx906) dies, 32 GB HBM2 each (64 GB in total), about 1 TB/s each, joined by an Infinity Fabric (XGMI) link |
| CPU | Intel Xeon W-3245, 16 cores / 32 threads, AVX-512 (Cascade Lake) |
| Memory | 192 GB DDR4-2933, six channels |
| OS | Ubuntu 26.04 with the t2linux kernel (`7.3.3-2-t2-resolute`), the kernel's own `amdgpu` driver |
| ROCm | 7.14 for gfx906 from TheRock's nightly wheels (AMD's own releases no longer ship gfx906) |

Other gfx906 cards (Instinct MI50 / MI60, Radeon VII, Radeon Pro VII) use the same kernels, but none of the numbers below
were measured on them. A smaller GPU or less RAM changes what fits: DeepSeek in particular needs the 192 GB.

## Before you start: let the GPUs clock up

**This is the most important setting on this machine.** The amdgpu driver has a power level per card,
`/sys/class/drm/cardN/device/power_dpm_force_performance_level`. On this Mac Pro a udev rule
(`/etc/udev/rules.d/30-amdgpu-pm.rules`) set it to `low`, which holds the shader clock at **999 MHz**. The Vega II boosts to
**1,730 MHz** on `auto`. With the same engine, Qwen3.8-Flash-Next went from 50.9 to 70.9 tok/s (37.3 → 27.3 ms a verify
window) when the level went from `low` to `auto`. Every number in this guide is measured on `auto`.

Check it:

```bash
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
```

Each AMD card should print `auto`. If one prints `low` (or `profile_min_sclk` / `profile_min_mclk`):

1. Set `auto` now (until the next reboot):

   ```bash
   echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
   ```

2. Make it stick. Find the rule that sets it:

   ```bash
   grep -rl power_dpm_force_performance_level /etc/udev/rules.d /lib/udev/rules.d
   ```

   In that file change `"low"` to `"auto"`. This machine's rule now reads:

   ```
   KERNEL=="card[012]", SUBSYSTEM=="drm", DRIVERS=="amdgpu", ATTR{device/power_dpm_force_performance_level}="auto"
   ```

   Then reload the rules (or reboot):

   ```bash
   sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=drm
   ```

3. Restart Strata.

**The app warns you.** When an AMD card reports one of the low levels, the **Model** page shows "Your AMD GPU clocks are
held at their lowest setting", naming the cards, with these commands under **How to fix it**. The check reads the level when
the app starts, so restart the app after changing it.

The cards draw more power and run warmer on `auto`; the **Monitor** page shows each card's temperature, clock and power if
you want to check them.

## Install

1. Build the AppImage, or take one built from this fork:

   ```bash
   packaging/build_appimage_gfx906.sh --appimage <official Strata-<ver>-linux-x86_64.AppImage>
   ```

   It writes `Strata-<ver>-gfx906-linux-x86_64.AppImage` with the gfx906 engine package inside
   (see [AMD_HIP.md](AMD_HIP.md#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)). The engine package holds
   the Qwen engine (Qwen3.8-Flash-Next and Qwen3.6-35B-A3B), the DeepSeek-V4.1 engine (`strata-dsv41`) and the picture
   readers. Its version is `<engine version>.<GFX906_REV>`, for example 0.1.59.18.
2. Start it. The app installs the bundled engine on first start; nothing is fetched from the update feed for gfx906.
3. Pick a model, set one die as **Main** and the other as **Helper**, and load it. Models already in `~/Documents/Strata`
   are checked locally, not downloaded again.

## What runs, and how fast

Decode is greedy generation on a single request at the stated context; prompt is prefill speed.

### Qwen3.8-Flash-Next (both dies, layer split, MTP drafting)

| Pack | Decode | Engine |
| --- | ---: | --- |
| ISTA-DASLab IQ3_XXS | **~83 tok/s** | 0.1.59.15 |
| ISTA-DASLab IQ3_S | **~74 tok/s** | 0.1.59.11 |
| IQ3_S before this work (stock 0.1.59.1, clocks `low`) | 33–39 tok/s | |

Drafting on **Auto** uses the MTP layer, not DFlash2, on gfx906 (MTP was 71–77 tok/s against DFlash2's 55–58), with
two pipelined verify windows. Expect less on long, creative text with reasoning on: a ~500-word story at 64K context read
about 60 tok/s in the app, because fewer drafted tokens are accepted. Q2_0 and IQ2_XS packs have gfx906 kernels too, but were
not measured end to end. How it was done: [GFX906_KERNEL_PLAN.md](GFX906_KERNEL_PLAN.md).

### Qwen3.6-35B-A3B (128K context)

| | Q4 (UD-Q4_K_XL), one die | Q4, both dies | Q8_0, both dies |
| --- | ---: | ---: | ---: |
| Decode, MTP drafting (the app's default) | 128–196 tok/s | 115–188 | 107–160 |
| Decode, no drafting | 114 | 106 | 102 |
| Prompt, 4K tokens | 1,345 | 1,318 | 1,182 |

- Q4 fits on one die and is about 7 % faster there; select both cards for Q6 / Q8_0, which need both (the layers split
  across the dies, every expert in VRAM).
- Checked against llama.cpp (KL 3.3e-3 for Q4, 3.8e-3 for Q8_0); Q4 on two dies is bit-for-bit the one-die output, and
  drafting gives the same text as no drafting.
- Not tested on two dies: Q6, pictures.
- Details: [QWEN36.md](QWEN36.md#gfx906-vega-20-radeon-pro-vii-mi50-vega-ii).

### DeepSeek-V4.1-Flash (both dies, tensor parallel, ISTA-DAS 2.00 bpw general set)

| | Decode |
| --- | ---: |
| 8K context, drafting **Off** | **~29 tok/s** |
| 128K context, in the app | ~20 tok/s |
| First run on gfx906 (layer split) | 19–21 tok/s |

- About 50 GB of the 132 GB of experts sit in VRAM; the rest run on the Xeon's AVX-512 (a Cascade Lake build of the i-quant
  kernels, since the W-3245 lacks VBMI / VPOPCNTDQ). The Engram table is read from the SSD. This is why it needs the 192 GB.
- Both dies compute every layer (tensor parallel over each expert's hidden width, split unevenly as 1280 + 1024).
- **Set Drafting to Off.** The DSpark drafter makes it slower here (13–21 tok/s): each drafted token roughly triples the CPU
  lane's expert work.
- Speed varies from run to run (about 20–30 tok/s) when the system swaps; close large programs before loading it. The first
  load reads about 130 GB and takes several minutes.
- Details: [DEEPSEEK41.md](DEEPSEEK41.md#gfx906-mac-pro-2019-radeon-pro-vega-ii-duo).

## Quality checks

Run on 12 November 2026 with the engines above, both dies, GPU power level `auto`:

| Model | Functional checks (`strata bench run`, 12 cases) | Logits |
| --- | --- | --- |
| Qwen3.8-Flash-Next IQ3_XXS | 12 / 12 | `tools/quant_quality.py` against the gated 0.1.59.15 run: 100 % top-1, KL 0 (bit-for-bit) |
| Qwen3.6-35B-A3B Q8_0 | 12 / 12 | against llama.cpp: KL 3.8e-3 |
| DeepSeek-V4.1-Flash 2.00 bpw | 12 / 12 | against `tools/dsv41/ref_gguf.py` (an independent PyTorch forward on the CPU, 428 tokens of prose and code): 96.7 % top-1, mean KL 0.008, perplexity 6.54 vs the reference's 6.58 |

The functional checks are instruction following (JSON and exact text), retrieval from 4K and 16K prompts, and tool
calls ([BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)). The benchmark's speeds at 4K / 16K prompts: Qwen3.6 Q8_0
reads 1,073 / 899 tok/s and decodes 84 / 47 tok/s; Qwen3.8 IQ3_XXS reads 475 / 661 and decodes 78 / 72; DeepSeek reads
only 19 / 36 tok/s (a 3,000-token prompt waits about 2.5 minutes for its first token) and decodes 28 at every length.

To repeat the DeepSeek check (the reference needs `torch` (CPU) and llama.cpp's `gguf-py`; one pass of 428 tokens took
72 minutes and ~35 GB of RAM):

```bash
strata-dsv41 --pack PACK --gpus 0,1 --ids @text.ids --n 1 --max-context 8192 --dspark none --dump-logits engine.f32
python tools/dsv41/ref_gguf.py --pack PACK --ids "$(cat text.ids)" --dump ref.npz --no-hs --dev cpu
python tools/dsv41/compare.py --ref ref.npz --engine engine.f32
```

## Where the time goes, and what's left

- **Qwen3.8:** at `auto` clocks each die is busy ~70–77 % of a window; the rest is the cross-die and host round trip and the
  drafter's chain, not kernel time.
- **DeepSeek:** about 1,700 small kernels per token on each die (5–50 µs each, mostly fixed latency) and ~9 ms of CPU lane.
  Getting past ~32–35 tok/s would need those fused into far fewer kernels.
- **Qwen3.6:** remaining ideas are the shared-expert gate/up, the router top-k and a recurrent-layer kernel that spills
  registers (roughly 10 % together).

Every gfx906 path keeps a switch back to the generic kernel (`STRATA_G906_*=0`, `STRATA_Q36_G906=0`); they are listed in the
linked documents.
