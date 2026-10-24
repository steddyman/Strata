# Strata vs llama.cpp v0.6.0 on a DGX Spark (2026-10-24)

Measurement only. The method and script of [2026-10-06-llamacpp-0.6.0](../2026-10-06-llamacpp-0.6.0/README.md)
(`bench/bench_vs_llama.py`), the same prompts (`prompt-4k.txt`, `prompt-32k.txt` there), on one NVIDIA DGX Spark.
Raw per-request rows: `raw/*.jsonl` (the command is in each config's first row), the machine config `machine-spark.json`,
the console `run.log`. Home folders and the host name are replaced.

## What was compared

- **Machine:** DGX Spark, GB10 (sm_121), 128 GB unified memory (121 GiB seen by Linux), 20 Arm cores (Cortex-X925 +
  Cortex-A725), DGX OS, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0.88.
- **Model:** Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** GGUF (ISTA-DASLab, the 2-shard files setup downloads). Both engines
  read the same files.
- **Strata:** engine 0.1.48.1 (`dgx-spark` branch, built on the Spark for sm_121), the config `./setup.sh --setup --family
  qwen --model IQ3_XXS --context 65536 --vision no --yes` wrote, unchanged: `--spec 4 --spec-min-p 0.5 --mtp rt --kv int8
  --kv-resident 32768 --expert-cache auto --prefill auto --mmap-experts --ple-io ram --max-context 65536`. Not tuned
  (`--calibrate` not run for this model).
- **llama.cpp:** **v0.6.0** (tag b11429, commit d8123504), built on the Spark with `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_NATIVE=ON`. The 2026-10-06 report names commit 8345f33 for the same tag; this is
  the commit the tag points to now.
  - `llama-gpu`: `-ngl 99 -b 2048 -ub 2048 -c 65536 -ctk q8_0 -ctv q8_0 -fa on`. Every layer and expert on the GPU (the
    whole model fits; no `-ncmoe`).
  - `llama-gpu-mtp3`: the same plus unsloth's MTP head `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (`-md ... --spec-type
    draft-mtp --spec-draft-n-max 3`), as in the 2026-10-06 report. Strata's own MTP head is another file of the same layer.
- **Protocol:** as 2026-10-06: `/v1/chat/completions`, no thinking, 256 new tokens, `top_k 20, top_p 0.95` at temp 0.7,
  seed 1, prompt cache off, a unique nonce per request. 3 rounds, the config order alternating; each round starts every
  engine fresh with two warm-up requests. **n = 6 per temp at 4K, n = 3 at 32K.** Medians. Prompt tok/s, decode tok/s and
  TTFT are the servers' own timings.

## Results (medians; decode = 256 tokens)

| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft accepted |
|---|---|---|---|---|---|---|---|
| **Strata setup** | 4K | 0 | 6 | **1265.6** | **51.1** | **3.20** | 0.68 |
| **Strata setup** | 4K | 0.7 | 6 | **1258.5** | **46.7** | **3.22** | 0.56 |
| **Strata setup** | 32K | 0 | 3 | **1469.4** | **52.8** | **21.8** | 0.66 |
| **Strata setup** | 32K | 0.7 | 3 | **1448.6** | **38.9** | **22.1** | 0.56 |
| llama.cpp, MTP off | 4K | 0 | 6 | 952.0 | 29.7 | 4.26 | - |
| llama.cpp, MTP off | 4K | 0.7 | 6 | 948.8 | 29.6 | 4.27 | - |
| llama.cpp, MTP off | 32K | 0 | 3 | 872.4 | 25.1 | 36.7 | - |
| llama.cpp, MTP off | 32K | 0.7 | 3 | 879.4 | 25.3 | 36.4 | - |
| llama.cpp, MTP n=3 | 4K | 0 | 6 | 880.7 | 36.2 | 4.60 | 0.47 |
| llama.cpp, MTP n=3 | 4K | 0.7 | 6 | 858.1 | 35.7 | 4.73 | 0.48 |
| llama.cpp, MTP n=3 | 32K | 0 | 3 | 778.4 | 34.7 | 41.2 | 0.59 |
| llama.cpp, MTP n=3 | 32K | 0.7 | 3 | 812.8 | 30.4 | 39.4 | 0.43 |

Strata against llama.cpp's faster setting in each cell:

| | 4K, temp 0 | 4K, temp 0.7 | 32K, temp 0 | 32K, temp 0.7 |
|---|---|---|---|---|
| prompt tok/s (vs MTP off) | **1.33x** | **1.33x** | **1.68x** | **1.65x** |
| decode tok/s (vs MTP n=3) | **1.41x** | **1.31x** | **1.52x** | **1.28x** |

- **MTP helps llama.cpp here**, unlike on the 12-16 GB cards of 2026-10-06: with every expert resident, decode goes from
  29.7 to 36.2 tok/s at 4K (+22%) and 25.1 to 34.7 at 32K (+38%), at a 7-11% cost in prompt speed.
- Strata's prompt speed rises with the context (1266 -> 1469 tok/s, 4K -> 32K); llama.cpp's falls (952 -> 872).
- Strata's decode drops at temp 0.7 (51.1 -> 46.7 at 4K, 52.8 -> 38.9 at 32K) as its draft acceptance does (0.66-0.68 ->
  0.56); llama.cpp without MTP is flat.
- **Answers:** both engines' summaries were read for round 1, temp 0, both cells: coherent and close to each other (`text`
  in the raw rows, first 300 characters).

## Caveats (be fair)

- One machine, 3 rounds. Spread was not analysed beyond the medians; the raw rows are included.
- llama.cpp's settings were not swept (`-ub`, `-b`, thread count, `--spec-draft-n-max`). Batch 2048 / ubatch 2048 is the
  2026-10-06 report's tuned value, not one tuned on the Spark.
- Strata's config is setup's, untuned for IQ3_XXS on this machine; it keeps the 29 GB n-gram table in RAM (`--ple-io ram`).
- The two MTP heads are different files (Strata's `rt` head vs unsloth's Q8_0 head) for the same layer.
- Memory columns are left out: `nvidia-smi` reports no memory use on the GB10 (unified memory), so the script's VRAM peak
  reads 0, and its RSS counts different things for the two engines (Strata's mapped experts are in its RSS; llama.cpp's
  CUDA allocations are not).
- No quality run (`bench_vs_llama.py quality`), no 128K prompts, no concurrency.
