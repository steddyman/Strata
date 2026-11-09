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
| + Q6_K kernel and DPP row sums (rounding order) | **44.7** | **43–45 tok/s** | 95.04 %, KL 0.0228, PPL 4.031 (control: 94.85 %, 0.0242, 4.103) |

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
