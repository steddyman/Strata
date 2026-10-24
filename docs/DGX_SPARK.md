# NVIDIA DGX Spark (GB10)

Strata runs on an NVIDIA DGX Spark. **Status: experimental**, measured on one Spark (below). There is no ready-made
engine, app or Terminal for it: you build the engine yourself, on the Spark (about 1.5 minutes) or on an Apple Silicon
Mac, and start Strata with `./setup.sh`.

What is different about the Spark:

- **The CPU is ARM (arm64), not x86.** Strata's own CPU expert kernels are written for x86 (AVX2 and AVX-512). The arm64
  build leaves them out. A model's CPU experts, if any, run on ggml-cpu's rows, which use the ARM vector unit (NEON).
- **The GPU is a Blackwell GB10 with compute capability 12.1 (`sm_121`).** The RTX 50 cards are 12.0 (`sm_120`). The
  engine is built for `sm_121`. The settings tuned on RTX 50 cards (`STRATA_SM120_PERF`) stay off on 12.1.
- **The CPU and the GPU share one memory (128 GB).** `nvidia-smi` shows the GPU's memory as `[N/A]`. Setup counts all of
  the RAM, less 6 GB for the system, as GPU memory, as it does for Strix Halo ([STRIX_HALO.md](STRIX_HALO.md)). The engine
  sizes its expert cache from what Linux can give back (`MemAvailable` less 6 GiB, `STRATA_UMA_HEADROOM_GIB` changes it).
  Setup adds `--mmap-experts` there: the experts are read from the model files in place instead of a second copy in RAM,
  so the GPU's cache holds all of them (NVFP4: 24576 of 24576 experts, against 13702 with the RAM copy).
- **Use a model that setup prepares.** These are "native" packs and run on any CPU. The old canonical Q2_0 pack (a pack
  folder without `native_experts.txt`) needs an x86 CPU with AVX-512, and the engine stops with a message there.

## Build the engine on a Mac

You need an Apple Silicon Mac with [Docker Desktop](https://www.docker.com/products/docker-desktop/). The Mac runs an
arm64 Linux container with NVIDIA's CUDA 13.0 toolkit. This needs no emulation, and the compiler needs no GPU.

1. In Docker Desktop, open **Settings > Resources** and give Docker as many CPUs and as much memory as you can spare. Each
   compile job takes about 2-3 GB. With the default 2 CPUs and 8 GB the build is slow.
2. In the Strata folder, run:

   ```sh
   spark/build_on_mac.sh
   ```

   The first build takes a while (it downloads the CUDA image, about 4 GB, and compiles every kernel). A second build
   compiles only what changed. `BUILD_VISION=0 spark/build_on_mac.sh` builds without the picture reader, which is faster.

3. The engine is in `out-spark/engine/`, and the same folder packed as `out-spark/strata-spark-engine.tar.gz`.

## Build the engine on the Spark itself

DGX OS comes with the CUDA toolkit. Use CUDA 13.0 or 13.1 if you can: CUDA 13.2.0 and 13.2.1 made wrong answers on RTX 50
cards (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)); 13.2.2 is fixed.

```sh
git clone https://github.com/Niko1221/Strata.git && cd Strata
sudo apt install build-essential python3-venv
python3 -m venv .venv-build && .venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH=$PWD/.venv-build/bin:$PATH PYTHON=$PWD/.venv-build/bin/python spark/build_engine.sh
cp -r out-spark/engine ./engine
```

You can also let setup compile it: `./setup.sh --build` (10-20 minutes, once).

## Install and start on the Spark

1. Get this Strata folder onto the Spark (`git clone` it, or copy it).
2. Put the engine in the Strata folder as `engine/`. From the Mac:

   ```sh
   scp out-spark/strata-spark-engine.tar.gz spark:~/
   ssh spark 'tar -xzf strata-spark-engine.tar.gz -C ~/Strata'
   ```

   Replace `spark` with your Spark's host name, and `~/Strata` with where the Strata folder is. The engine must end up
   as `engine/` inside that folder: `ls ~/Strata/engine/strata` should list the file. If setup finds no engine there, it
   says which folder it looked in and compiles one on the Spark instead (this needs the CUDA toolkit, which DGX OS has).
3. On the Spark, in the Strata folder, run:

   ```sh
   ./setup.sh
   ```

   Setup finds the GB10, says "ready-made engine already installed" and installs NVIDIA's cuBLAS and CUDA runtime for
   arm64 from pip. Then it asks for the model, the context and pictures, downloads the model and starts it, as on any PC.
   The Strata page is at `http://127.0.0.1:8080`. To use it from another computer, see [INSTALL.md](INSTALL.md).

## Using it from another computer

`127.0.0.1:8080` is the Spark itself: from your Mac it does not answer. Either forward the port over SSH (with NVIDIA Sync,
`ssh -L 8080:127.0.0.1:8080 <user>@<spark>.local`, then open `http://127.0.0.1:8080` on the Mac), or let Strata listen on
the network with an API key ([INSTALL.md](INSTALL.md)).

## Measured

One DGX Spark (GB10, 128 GB, DGX OS, driver 580.178.04), engine 0.1.48.1 built on the Spark with CUDA 13.0.88 for
`sm_121`, NVFP4 (Maximilian228 Qwen3.8-Flash-Next-NVFP4-GPTQ), 128K context, 8-bit KV, images on. `tools/strata_bench.py`
(`./bench.sh`) medians, 2026-10-24:

| | experts copied to RAM (before) | `--mmap-experts` (setup's default here) |
| --- | --- | --- |
| experts in the GPU's cache | 13702 of 24576 | 24576 of 24576 |
| chat output | 40.6 tok/s | 44.4 tok/s |
| 4K prompt | 1222 tok/s | 1322 tok/s |
| 16K prompt | 1397 tok/s | 1471 tok/s |
| first token, 16K prompt | 11.55 s | 10.97 s |
| agent turns, output | 43.6 tok/s | 43.1 tok/s |

Setup's tuning then chose `--pool-workers 4` (43.9 tok/s in its own measurement). Its PCIe-share step measures nothing
here (44.0-44.2 tok/s at every value): every expert is already on the GPU.

Prompt reading is about 55% of an RTX card that holds the whole model (docs/NVFP4.md: 2,503 tok/s, 4-8K prompts): the GB10
has 48 SMs and 273 GB/s of LPDDR5X. The Blackwell FP4 prompt mode (`STRATA_PREFILL_NVFP4=w4a4`) runs on it (`sm_121a`)
but is not faster here: 870-1,159 tok/s against 1,012-1,113 with the default `w4a8` on three ~10K-token prompts of
random words, so the default stays.

**Against llama.cpp** (v0.6.0, every layer on the GPU, same IQ3_XXS GGUF, same prompts;
[bench/results/2026-10-24-dgx-spark-vs-llamacpp](../bench/results/2026-10-24-dgx-spark-vs-llamacpp/README.md)): Strata reads
prompts 1.33x (4K) to 1.68x (32K) faster and decodes 1.28x to 1.52x faster than llama.cpp's faster setting (with its MTP
head for decode, without for prompts). At 4K, temp 0: 1266 vs 952 tok/s prompt, 51.1 vs 36.2 tok/s decode.

Answers and pictures were checked by hand: a factual question with a code answer, and a picture of a red square and a blue
circle, described correctly.

## What was checked, and what was not

- Checked: the engine (`strata`, `libstrata_kq.so`) and the picture reader (`strata-vision`) compile and link for arm64
  with CUDA 13.0 for `sm_121`, in the container `spark/build_on_mac.sh` uses. Setup's handling of the GB10 and of the
  arm64 CPU has tests: `python -m unittest tools.test_setup_spark`.
- Checked on a DGX Spark: setup, the start, tuning, `./bench.sh`, chat and pictures with NVFP4 (above).
- Not checked: the other model families on a Spark, the Mac-built engine against the Spark-built one for speed, long
  sessions, and `strata bench run` (the standard benchmark).
- Not available: the Strata app, the `strata` Terminal and its `install.sh`, and the Docker images are for x86_64 only.

If you run it, please report what you see: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
