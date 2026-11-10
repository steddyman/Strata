# gfx906 decode rewrite: plan

Goal: decode on two gfx906 dies (Radeon Pro Vega II Duo; also 2x MI50/MI60) much closer to what HBM2 can deliver,
with gfx906-native kernels instead of CUDA-shaped kernels compiled through HIP.

Baseline (engine 0.1.59.1, Qwen3.8-Flash-Next IQ3_S, `--layer-split auto`, MTP, `--pipeline-windows 2`,
`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`): **33–38 tok/s**, one verify window = **~1.9 tokens**, and each
stage (one die, 24 layers) spends **25.4 ms per window** (stage 1 adds a 2.4 ms output head).

## Where a stage's 25 ms go, and the floor

Floor = the weight bytes the step reads per window at the **780 GB/s** a plain read kernel reaches on one die
(`tools/gfx906/mmvq_bench.cpp`). Bytes are per stage (18 GDN + 6 QSA layers) from the GGUF tensor sizes.

| Step (profile labels) | Now (ms) | Bytes / stage | Floor (ms) | Bound by |
| --- | ---: | ---: | ---: | --- |
| Dense GEMVs: GDN qkv, z, out-proj; QSA q, k/v, out | 6.4 | ~1.0 GB | 1.3 | vector-memory instruction rate, ALU per column |
| Hyper-connections: hc0/hc1 norm, down, up (BF16) | 5.4 | 0.63 GB | 0.8 | same, plus 6 launches/layer |
| Experts: VRAM hits + PCIe grp (IQ3_XXS/IQ2_S gate/up, IQ4_NL down) | 5.6 | ~0.85 GB | 1.1 | Mode 7 kernels, 4 launches/layer |
| Router + plan + doorbell (BF16 512x2560 + top-10) | 2.7 | 63 MB | 0.1 | latency: 3–4 small launches/layer |
| Shared expert + quantize (8 launches/layer) | 1.6 | 65 MB | 0.1 | latency |
| GDN conv, a/b, recurrence; QSA append, scores, attention, gate | 2.6 | small | — | latency, 32-lane warps on a 64-lane GPU |
| Waits, copies, combine, gaps | 1.5 | — | — | launches, flags |
| **Total** | **25.4** | **~2.6 GB** | **~3.3** | |

So the dies run at about 100 GB/s of useful weight traffic, an eighth of what they can read. No single kernel
explains it: the time is spread over about 35 launches per layer, most of them well below the roofline.

## What the first kernel taught (Q6_K, branch `gfx906-kernels`)

- The CUDA-shaped kernels are limited by the **count of small loads** (MemUnitBusy 60–85 % with DRAM at 20–35 %),
  not by bytes or the ALUs: 4-byte and 1-byte loads, activations re-read per row and column.
- gfx906 loads 16 bytes from the GGUF blocks' 2-byte alignment (clang emits `global_load_dwordx4` for an
  `aligned(2)` vector type), so wide loads need **no repacking and no extra VRAM**.
- Staging every column's activations in LDS once per block of rows takes them off the vector memory unit.
- **Register budget decides everything**: at 88+ VGPRs a SIMD holds two waves and HBM latency is not hidden;
  keep kernels at ≤64 VGPRs (4 waves/SIMD) or hide latency explicitly.
- Result: Q6_K 10240x2560 112 → 48 µs at T=1 (447 GB/s), 125 → 66 µs at T=2; end to end only ~2 %, because Q6_K is a
  sixth of the dense GEMV time. Every step above has to move for decode to move.

## Results so far

| Engine (test build) | ms per window | Decode | Quality vs 0.1.59.1 |
| --- | ---: | ---: | --- |
| 0.1.59.1 (stable) | 51.5 | 39 tok/s | reference |
| + DPP / swizzle for constant `__shfl_xor` (exact) | ~50.2 | | 100.00 %, KL 0 |
| + gfx906 defaults table (exact in the gate) | 48.4 | | 100.00 %, KL 0 |
| + Q6_K kernel and DPP row sums (rounding order) | 44.7 | 43–45 tok/s | 95.04 %, KL 0.0228, PPL 4.031 (control: 94.85 %, 0.0242, 4.103) |
| + two-kernel hyper-connection read (no norm launch) | 41.7 | ~46 tok/s | |
| + Q4_K and IQ4_XS kernels, IQ3_XXS sign unpack | 41.2 | 46.5 tok/s | 95.20 %, KL 0.0216, PPL 4.094 (0.1.59.8) |
| + expert mode 9, one-token router via the rows kernel, QSA merge v2 (all bitwise) | 39.7 | 48.9 tok/s | bitwise 0.1.59.8 (0.1.59.9) |
| + lane-group GEMV rows, Q5_K kernel | 37.3 | 50.9 tok/s | 94.87 %, KL 0.0239, PPL 4.107 (0.1.59.10) |
| 0.1.59.10 with the shader clock on `auto` (was pinned `low`) | 27.3 | 70.9 tok/s | (same engine) |
| + 6144-input rows on the gfx906 kernels, IQ4_NL expert down kernel (`auto` clock) | **26.2** | **74.5 tok/s** | 94.69 %, KL 0.0255, PPL 4.097 (0.1.59.11; with STRATA_G906_DOWN=0: 95.14 %, 0.0215, 4.135) |

The quality gate's noise, from the summation-order variants so far: top-1 94.7-95.2 %, KL 0.021-0.026, PPL
4.03-4.14, with KL and PPL often moving in opposite directions - a single run inside that envelope is a pass.

What did not pay: LDS staging alone on the CUDA-shaped kernels (the weight loads, not the activations, bound
them), more rows per wavefront, a software prefetch that doubled the registers (two waves a SIMD), the existing
`STRATA_GR_V3` read (it needs 80 KB of LDS; gfx906 has 64 KB), dword loads in `get_int_b2` (the expert gate/up is
ALU-bound: VALU 63 %, memory 35 %). Pitfalls met: indexing a register vector's bytes (`&hdr`) put it in scratch
(Q4_K 35 -> 54 us until fixed); a kernel chosen by column count breaks verify's per-column exactness.

Round 2 (0.1.59.9-10), from a fresh trace grouped by kind (dense GEMV 26 %, experts 18 %, hyper-connections 11.5 %,
router 8 %, sync 6.5 %, attention 5 %):

- Expert gate/up, mode 9 (`g906_gu_kernel`): each signed grid word built once, `(g ^ m) + b` with the sign masks from
  `v_mul_u32_u24` + `v_perm_b32` (mode 8 used a quarter-rate `v_mul_lo_u32` and a correction dot a token).  Bitwise
  mode 8.  What did not pay: a sign table in LDS (random 16-byte reads conflict), double-buffered passes (151 VGPRs).
  The expert kernels stay at ~260 GB/s: latency-bound at 3 waves a SIMD, VALU ~50 % (a plain read of the same
  98-byte blocks runs at 700 GB/s, so the access pattern is not the limit).
- The router's one-token BF16 GEMV launched 512 one-row blocks; on this card that shape hits a ~70 us stall (a bare
  read kernel shows it too, at some grid sizes).  The rows kernel (4 rows a block, bitwise) takes 14 us.
- The drafter's QSA merge walked all 512 chunks of `--mtp-window` (243 us); `STRATA_ATTN_MERGE_V2` (bitwise,
  `qsa_merge_daily_parity` 1172 cases on gfx906) stops at the live width.
- Dense kernels: LPR lanes a row chosen from n_in (2560 inputs = 40 lane tasks: 8 lanes x 5, not 40 of 64), used
  where >= 128 blocks remain; a Q5_K kernel.  The 640-row shared-expert matrices are launch/latency-bound (~8-15 us
  isolated); in the trace they read 40-80 us because they share the GPU with the experts' stream.

**Clock note**: `/etc/udev/rules.d/30-amdgpu-pm.rules` sets `power_dpm_force_performance_level=low` on both cards,
which pins the shader clock at 999 MHz (the card boosts to 1730 MHz).  All numbers here are at that setting.

At `auto` clocks each GPU is busy ~70-77 % of a steady decode: the rest is the window's cross-GPU and host round
trip, not kernel time.  The drafter's LM head (106K x 2560 Q6_K, 340 us a window) already reads at ~85 % of the
read roofline.

Round 3 (0.1.59.12, not installed): a Q2_0 expert down kernel (T 2: 60 -> 42 us), the gate/up kernel at 3 waves a
SIMD (IQ2_S -13 %, IQ3_S -6 %, IQ3_XXS -8 % vs mode 8, bitwise), a split LDS layout in the hc down read (bitwise, ~5 %).
**The window did not get faster** (25.8 vs 25.9 ms; tok/s moved only with tokens a window, 1.77-2.02, which follows
the text).  At `auto` clocks the kernels are no longer what bounds the window:

- moving layers between the GPUs (`--layer-split 26` / `28` vs auto's 24) changed nothing (25.4-26.3 ms);
- `STRATA_VERIFY_DEVICE_PLAN=1` (plan resident layers on the GPU, exact) changed nothing;
- the HIP API trace: `hipGraphLaunch` ~8 % of wall (~8.5 graphs a window, ~50 us each); the host thread otherwise
  polls; each GPU is busy ~70 %, CUDA0 idle ~12 ms a window waiting for the other side.

So the window is a dependency chain - the CPU expert plan a layer (doorbell -> host -> `wait_flag_ge`), the GPU hand-off,
the MTP draft steps (~0.95 ms each on CUDA1, 340 us of it the 106K-row LM head), speculative windows mostly rolled back
- rather than a sum of kernel time.

Scheduling, measured (0.1.59.11, `auto` clocks, 2 x 2 interleaved runs each):

| Setting | ms / window | Note |
| --- | ---: | --- |
| default (pipelined windows, drafter on CUDA1, theta 0.20) | 25.7-26.4 | fresh windows ~28-29 ms, on-path speculative ~18 ms |
| `STRATA_PIPELINE_AUTO=0` (serial) | 29.5-30.0 | 66-67 tok/s against 74: the pipeline pays |
| `STRATA_MTP_STAGE0=1` (drafter on CUDA0) | 28.2-28.7 | fresh windows 32.8 ms: worse, as noted before on gfx906 |
| theta 0.05 / 0.40 (`STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_THETA`) | 25.8-26.0 / 27.0-27.2 | lower is at most ~1.5 % better (within noise) |
| `--layer-split 26` / `28` | 25.4-26.3 | no change |
| `STRATA_VERIFY_DEVICE_PLAN=1` | 25.5-26.7 | no change |

A serial window (trace): CUDA0's 24 layers 16.3 ms (98 % of it kernels: main stream 12 ms, the side stream 5.9 ms
beside it; `wait_flag_ge` 0.6 ms), CUDA1's 24 layers + head 16.8 ms, the hand-off ~0.5 ms, the drafter's chain 3.3 ms
(traced times).  The pipeline turns ~25-35 % of windows into on-path speculative ones; the rest are rolled back, and
"chain late" (the drafter's next chain not ready in time) blocks ~20-25 % of the chances.  0.1.59.12's kernels save
~0.45 ms a serial window (30.0 -> 29.55), invisible in the pipelined one.

So what is left is either more kernel work spread over every layer (each 1 % of a half is ~0.15 ms), a faster drafter
chain (fewer late chains; e.g. a lower-bit copy of the 106K-row draft head, 340 us of each ~0.95 ms step), or better
drafts (more on-path windows).

Next, by the trace (rocprofv3 --kernel-trace on a live serve; the engine must get SIGINT itself to write it):
expert gate/up and down (~4 s of 26 s traced, ALU-bound grid lookups), QSA `attn_merge` (87 us on 24 blocks),
the router chain (BF16 MMVF + `route_multi`, ~1.7 s), the shared expert (8 launches a layer), the remaining wave
kernels (Q5_K, Q8_0, IQ4_NL with 640-wide rows).

ISA notes that paid (Vega 7nm ISA, GPUOpen 26-Nov-2019): DPP `quad_perm` / `row_half_mirror` / `row_ror:8` and
`ds_swizzle_b32` replace `ds_bpermute` for xor 1-16; DPP `row_bcast15/31` finish a 64-lane sum; `global_load_dwordx4`
works on the GGUF blocks' 2-byte alignment. Checked and not useful here: `buffer_load ... lds` (dword-sized only),
`v_dot8_i32_i4` (needs 4-bit activations), wider SDWA unpacking (the compiler already emits it).

## Phases

Each phase lands as its own PR with: the microbenchmark numbers, the profile before/after, decode tok/s on the
three-run benchmark, and the quality gate below. Every new path keeps a `STRATA_G906_*=0` switch back to the old one.

### Phase 0 — measurement and quality gate (do first)

- Extend `tools/gfx906/mmvq_bench.cpp` into a per-step bench: every dense format at the model's shapes, the hc
  kernels, the router, the expert kernels, T = 1..4, with the read roofline and a bitwise / tolerance comparison.
- A `tools/gfx906/decode_profile.sh`: start `strata.py serve` on a test port with the profiler on, run the fixed
  prompts, print the stage table and tok/s. The stage table above is its first output.
- Quality gate: `tools/quant_quality.py run --gpu-experts` on the current engine as the reference, then on each
  candidate, compared with `tools/quant_quality.py compare`. On this model **any** change of summation order moves
  top-1 by about 5 % (the routers' near-ties flip, and the GDN state carries the difference along the segment): the
  stable engine against itself is 100.00 % / KL 0, but against its own CUDA layout (`STRATA_MMVQ_WAVE=0`, a shipped
  rounding variant) it is 94.85 % / KL 0.0242 / PPL 4.103 vs 4.080. So the bar is: **no worse than that control**
  (top-1, mean KL) and **perplexity within it**; an exact change (data movement only) must stay at 100.00 % / KL 0.
  Columns must stay exact across T (`mmvq_multi_parity`'s contract) or verify breaks; the kernel bench checks it.
  The gate runs FP16 KV (fixed by the tool), so an int8-KV-only path needs its own check.
- Note: profiling turns off the shared-expert side stream and some fusions, so the final number is always the
  un-profiled tok/s, not the stage sum.

### Phase 1 — switch on what already exists (small, fast)

The gfx906 build does not define `STRATA_USE_HIP`, so every AMD tuning gated on it takes the CUDA branch:
`fuse_convab` (conv + a/b), `fuse_hits` (`combine_k10_vec4_hits`), the fused wait + copy, `norm_rope_kernel`,
the side mixer (`verify.cpp:433-500`, `native_rope.cu:175`). Opt-in fusions exist too: `STRATA_HIP_SWIGLU_FUSED`,
`STRATA_EXP_MODE=8` (expert gate/up + SwiGLU + q8_1 in one kernel), `STRATA_QFUSE`, `STRATA_LFUSE` +
`STRATA_MMVF_ROWS` (router + shared gate), `STRATA_HC_PERSIST`.

- Try each on gfx906 one at a time, keep what wins and passes the gate, then make those the gfx906 defaults
  (an `|| defined(STRATA_HIP_GFX906)` per gate, each measured).
- Expected: 1–3 ms per stage, mostly launches removed. Effort: small.

### Phase 2 — dense GEMVs, gfx906-native (largest bandwidth share)

One kernel family for the dense formats in this model: Q6_K (qkv, ssm_out, k/v, head), Q4_K (z, attn_output,
shexp), IQ4_XS (attn_q, shexp), IQ4_NL (shexp down), BF16 (router, indexer, alpha/beta) — built on the Q6_K design:

- 16-byte weight loads straight from the GGUF blocks; each lane unpacks its weights once for every column.
- Activations in LDS (q8_1 for the quantized formats, F32/BF16 for BF16) once per block of rows.
- ≤64 VGPRs at T ≤ 4 (unpack per 16-value run, not all at once), enough rows per block to amortize the staging,
  and two rows per wave where n_in is small (2560 = 40 lane-tasks leaves 24 lanes idle).
- Fold the q8_1 quantize into the producer (the norm or the previous kernel) where the graph allows.
- Target: ≥ 450 GB/s at T=1 and ≥ 300 GB/s at T=4 for every shape. Expected: 6.4 → ~2.5 ms per stage.
  Effort: medium (five formats, one design).

### Phase 3 — hyper-connections (biggest single step)

Six launches per layer (norm, down 10240x320, up 320x10240, twice) at ~136 GB/s, 5.4 ms per stage.

- Rewrite down and up with the Phase 2 load pattern (BF16 rows, 16-byte loads); fuse norm into down
  (the norm's output is down's only input) and the inject into up; one launch per hc read where LDS allows.
- Measure `STRATA_HC_REQ8=1` (int8 hc weights, half the bytes) against the quality gate before relying on it.
- Expected: 5.4 → ~1.5 ms per stage. Effort: medium.

### Phase 4 — experts

Mode 7 (`native_gu_lds_kernel`, `native_down_lds_kernel`, `iq_kernels.cu:5898/5993`) is already gfx906-specific
but runs at ~170 GB/s.

- Bench each at the window's real shape (T ≈ 2, ~15–20 distinct experts per layer), then apply the same rules:
  wide loads, the VGPR budget, activations staged once.
- Fuse gate/up + SwiGLU + q8_1 (Mode 8 exists: measure it first) and down + combine (+ the hit add).
- Expected: 5.6 → ~2.5 ms per stage. Effort: medium-large (IQ3_XXS / IQ2_S grids, two expert layouts).

### Phase 5 — latency-bound small steps

Router + plan (2.7 ms), shared expert (1.6 ms), the GDN steps and QSA steps (2.6 ms), waits (1.5 ms).

- Router: one kernel for the 512x2560 BF16 projection + top-10 + the resident plan (the wave top-10 already
  exists in `router_top10.cu` but is off the native path).
- Shared expert: gate/up + SwiGLU + down in one or two kernels (its weights are 2.7 MB a layer).
- GDN conv/a/b/recurrence and the QSA steps: rewrite the 32-lane-warp kernels for 64-lane wavefronts (today a
  "warp" is half a wavefront, so half of every shuffle reduction is wasted) and merge neighbours.
- Expected: 8.4 → ~4 ms per stage. Effort: large (many small kernels, verify bit rules for the recurrence).

### Phase 6 — stretch: a layer-persistent decode kernel

One persistent kernel per layer (or per stage) that walks the steps with grid-wide barriers, removing the
remaining launches and flag waits. Only worth starting once Phases 1–5 have shown what is left.

## Expected result

| After | ms per stage (of 25.4) | Decode (est.) |
| --- | ---: | ---: |
| today | 25.4 | 33–38 tok/s |
| Phase 1 | ~23 | ~40 |
| Phases 1–3 | ~15 | ~55 |
| Phases 1–5 | ~11 | ~70 |
| floor (bandwidth only) | ~3.3 | not reachable: launches, recurrence, attention, drafts |

The estimates assume the two stages keep overlapping as they do now (`--pipeline-windows 2`) and that the MTP
acceptance stays at ~1.9 tokens per window. They are targets to measure against, not promises: Phase 2's first
kernel came in at 2.3x on its own matrix but 2 % end to end, which is why every phase is measured end to end.

## Order and risk

1. Phase 0 and 1 first: cheap, and they set the baseline every later number is compared to.
2. Phases 2 and 3 next: the most bandwidth per unit of work, and they share one design.
3. Phase 4, then 5. Phase 6 only if the profile still shows launches as the largest cost.

Risks: verify windows need every column exact across T (or decode and verify disagree); the GDN recurrence has
its own bitwise rules; gfx906 has 64 KB LDS and no matrix cores; ROCm for gfx906 comes from TheRock's community
wheels, so a compiler change can move the numbers — pin `ROCM_VER` per measurement.
