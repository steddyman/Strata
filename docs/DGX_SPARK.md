# Strata on the NVIDIA DGX Spark

Strata runs on an NVIDIA DGX Spark: one desktop box with a GB10 GPU, an Arm CPU (10 Cortex-X925 + 10 Cortex-A725) and
128 GB of memory shared by both. It runs Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on a single Spark.

**Status: experimental.** Measured on one Spark. There is no ready-made engine, app or Terminal for the Spark: you build
the engine (about 1.5 minutes on the Spark) and start Strata with `./setup.sh`.

## Speeds at a glance

| Model | Answers | Reading your prompt | Context | Download | Quality |
| --- | --- | --- | --- | --- | --- |
| **Qwen3.8-Flash-Next NVFP4** | **46.9 tokens/s** | 1,312 tokens/s (4K) · 1,453 tokens/s (16K) | 128K, reads pictures | 128 GB | The best Qwen: closest to the full model |
| **Qwen3.8-Flash-Next IQ3_XXS** | **52.7 tokens/s** | 1,258 tokens/s (4K) · 1,496 tokens/s (32K) | 64K as measured | 76 GB | 3-bit: smaller and faster, lower quality |
| **DeepSeek-V4.1-Flash 1.75-bit** | **37 tokens/s** once warm (24 just after starting) | 286 tokens/s (3.7K, warm) · 403 tokens/s (31K) | 8K by default, 128K tested* | 239 GB | Good for code and agents, weaker on plain prose |
| **DeepSeek-V4.1-Flash 2.00-bit** | **28 tokens/s** once warm (17 just after starting) | 252 tokens/s (3.7K, warm) | 8K by default* | 257 GB | Better than 1.75-bit, slower here |

On the same model file (IQ3_XXS) and the same Spark, Strata answers **1.4x faster** and reads a 32K prompt **1.7x
faster** than llama.cpp v0.6.0 ([the comparison](../bench/results/2026-10-24-dgx-spark-vs-llamacpp/README.md)).

\* Setup gives DeepSeek an 8K context. Coding agents need more: see
[Give DeepSeek a longer context](#give-deepseek-a-longer-context).

"Answers" is how fast the reply is written. "Reading your prompt" is how fast Strata reads what you send before it
starts to answer. All numbers are medians from one DGX Spark, engine 0.1.51 with the Spark speed settings, 26 October
2026 (from 24 October: the "just after starting" figures, with memory emptied first, and the 128K test, 403
tokens/s at 31K); the details are in
[How these were measured](#how-these-were-measured).

## Which model should I pick?

- **Qwen3.8-Flash-Next NVFP4** for most people. It is the closest to the full model, reads pictures, keeps every expert
  on the GPU and reads prompts fastest. Setup lists it as `--family nvfp4 --model NVFP4`.
- **Qwen3.8-Flash-Next IQ3_XXS** if you want the fastest answers or a smaller download, and can accept lower quality.
- **DeepSeek-V4.1-Flash 1.75-bit** for a second, different model for code and agent work. It is faster than 2.00-bit on
  a Spark because more of it fits on the GPU.
- **DeepSeek-V4.1-Flash 2.00-bit** for better DeepSeek quality, a little slower.

DeepSeek's experts (115 GB at 1.75-bit, 132 GB at 2.00-bit) are bigger than the Spark's memory, so part of them is read
from the SSD while it answers. That is why its first answers after a start are slower.

You can install several models. Only one runs at a time.

## Get started

### 1. Build the engine

**On the Spark itself** (DGX OS comes with the CUDA toolkit):

```sh
git clone https://github.com/steddyman/Strata.git ~/Strata && cd ~/Strata
sudo apt install build-essential python3-venv
python3 -m venv .venv-build && .venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH=$PWD/.venv-build/bin:$PATH PYTHON=$PWD/.venv-build/bin/python spark/build_engine.sh
cp -r out-spark/engine ./engine
```

This builds both engines: Strata's (for Qwen) and DeepSeek's. `BUILD_DSV41=0` leaves DeepSeek's out. Use CUDA 13.0 or
13.1 if you can: 13.2.0 and 13.2.1 made wrong answers on RTX 50 cards ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

**Or on an Apple Silicon Mac** with [Docker Desktop](https://www.docker.com/products/docker-desktop/): in the Strata
folder run `spark/build_on_mac.sh`. It builds in an arm64 Linux container (no emulation, no GPU needed) and leaves
`out-spark/strata-spark-engine.tar.gz`. Give Docker more CPUs and memory first (Settings > Resources) or it is slow.
Copy the archive to the Spark and unpack it inside the Strata folder, so that the engine ends up as `~/Strata/engine`:

```sh
scp out-spark/strata-spark-engine.tar.gz <you>@<spark>.local:~/
ssh <you>@<spark>.local 'tar -xzf ~/strata-spark-engine.tar.gz -C ~/Strata && ls ~/Strata/engine/strata'
```

### 2. Install a model

On the Spark, in the Strata folder:

```sh
./setup.sh
```

Setup finds the GB10 and the engine, then asks which model, how much context and whether it should read pictures,
downloads the model and starts it. To install DeepSeek, or another model later:

```sh
./setup.sh --setup                                                  # choose from the menu
./setup.sh --setup --family deepseek41 --model IST-DAS-1.75bpw      # or name it
```

### 3. Start it

Each installed model has its own start script in the Strata folder, for example:

```sh
./run-nvfp4-nvfp4.sh
./run-deepseek-v4.1-flash-ist-das-1.75bpw.sh
```

The model takes 1-3 minutes to load. The server says `ready:` when it is. Close the window or press Ctrl+C to stop it.

A server started in an SSH session stops when that session closes. To keep it running, start it inside `tmux`
(`tmux new -s strata`, then the script; detach with Ctrl+B then D, come back with `tmux attach -t strata`).

## Use it from your Mac or another computer

By default Strata only answers on the Spark itself (`127.0.0.1:8080`). To use it from other computers on your network,
set it up with an address and an API key (pick your own key):

```sh
./setup.sh --setup --family nvfp4 --model NVFP4 --host 0.0.0.0 --api-key <your-key>
```

The same `--host` and `--api-key` work with any model. They are saved in the model's `strata-*.json` file in the Strata
folder (`"host"` and `"api_key"`), which you can also edit by hand. Then, from your Mac:

- the chat page: `http://<spark>.local:8080/`
- OpenAI-compatible apps: base URL `http://<spark>.local:8080/v1` with your key
- Anthropic-compatible apps: `http://<spark>.local:8080/v1/messages` with your key

Quick check from the Mac: `curl http://<spark>.local:8080/v1/models -H "Authorization: Bearer <your-key>"`.

Without opening it to the network, you can also forward the port over SSH:
`ssh -N -L 8080:127.0.0.1:8080 <you>@<spark>.local`, then use `http://127.0.0.1:8080` on the Mac.

## Give DeepSeek a longer context

Setup gives DeepSeek 8K tokens of context, the size its engine was first checked at. Coding agents send far more than
that (their instructions and tools alone can be several thousand tokens). Install it with a larger context:

```sh
./setup.sh --setup --family deepseek41 --model IST-DAS-1.75bpw --context 131072
```

Setup warns that sizes above 64K were not tested by the DeepSeek release. On a Spark, 1.75-bit at 128K started with
0.12 GB of extra memory and read a 30,965-token prompt correctly (403 tokens/s, answering at 23 tokens/s). 2.00-bit at
128K was not tried.

## How Strata works on the Spark

- **The CPU is Arm (arm64), not x86.** Strata's own CPU expert kernels are written for x86 (AVX2 and AVX-512); the arm64
  build leaves them out. A model's CPU experts, if any, run on ggml-cpu's rows, which use the Arm vector unit (NEON).
- **The GPU is a Blackwell GB10 (compute capability 12.1, `sm_121`).** The RTX 50 cards are 12.0 (`sm_120`). The engine
  is built for `sm_121`. The settings tuned on RTX 50 cards (`STRATA_SM120_PERF`) stay off on 12.1.
- **The CPU and the GPU share one memory.** `nvidia-smi` shows the GPU's memory as `[N/A]`. Setup counts all of the RAM,
  less 6 GB for the system, as GPU memory, as it does for Strix Halo ([STRIX_HALO.md](STRIX_HALO.md)). Setup adds
  `--mmap-experts` there: the experts are read from the model files in place instead of copied into RAM first, so the
  GPU's cache holds all of them (NVFP4: 24,576 of 24,576 experts, against 13,702 with the RAM copy). DeepSeek's engine
  sizes its GPU share from what Linux can give back, less 6 GiB (`DS4_UMA_HEADROOM_GIB`): 107 GB of experts on the GPU.
- **Use a model that setup prepares.** These run on any CPU. The old canonical Q2_0 pack (a pack folder without
  `native_experts.txt`) needs an x86 CPU with AVX-512, and the engine stops with a message there.
- **Not available on the Spark:** the Strata app, the `strata` Terminal and its `install.sh`, and the Docker images,
  which are x86_64 only.

## Spark speed settings (Qwen3.8-Flash-Next NVFP4)

Answering is limited by memory bandwidth on the Spark. Each verify window reads a few GB of weights from the one
LPDDR5X pool for about 3 tokens, and the GPU is busy 97-98% of the time while answering (Nsight Systems). The dense
8-bit matrices and the NVFP4 experts already read at 87-95% of the ~230 GB/s the GB10 sustains. So the gains below come
from reading fewer bytes. Together they give **+10.5% faster answers**: 56.4 against 51.0 tokens/s, on 3 prompts of 512
tokens, 3 alternating rounds, 26 October 2026.

| Setting | On by default | What it does | Answers |
| --- | --- | --- | --- |
| 8-bit hyper-connection read | yes, on a GB10 (`STRATA_HC_Q8=0` turns it off) | The 1.2 GiB of bf16 hyper-connection weights read as Q8_0. About +7% | Can differ at near-tie tokens: 94.6% same top token, KL 0.030, perplexity 3.936 against 3.937 |
| `--mtp-q4 all` | yes, written by setup | The draft layer's 4-bit copies, so drafting reads fewer bytes. About +4.5% | Same bits: the verify window decides every token |
| Batched PLE projections | yes, on a GB10 (`STRATA_PLE_BATCH=0` turns it off) | The window's n-gram key and value rows in one pass. About +1.8% | Same bits |
| English draft vocabulary | no: setup's draft vocabulary choice `en` | A 105 MiB draft head instead of 276 MiB. About +4.5% | Same bits. Answers in Chinese, Japanese or Korean get fewer drafts |

**Tried and not kept.** All of these were measured on this Spark:

- *Requantizing the dense weights* (`tools/requant_dense_gguf.py`): Q6_K was +3% at KL 0.047, Q5_K +7% at KL 0.064 and
  Q4_K +15% at KL 0.129. Against the 8-bit dense weights, that is too much quality for the speed.
- *Dropping low-weight routed experts per window*: under +5%, within noise.
- *The DFlash2 block drafter*: 50 against 51.4 tokens/s for MTP, with the same tokens per window.
- *Strix Halo's decode switch set*: each switch alone was within ±2%. Together they were slower, and QFUSE changes the
  answers.
- *FP4×FP4 prompt GEMMs* (`STRATA_PREFILL_NVFP4=w4a4`): 1141/1318/1298 against 1151/1341/1350 tokens/s at 16K/32K/64K.
- *A deeper speculation window* (`--spec 6`) and other `--spec-min-p` values.

## Spark speed settings (DeepSeek-V4.1-Flash)

Setup writes `--dspark-conf 0.4` on a Spark, against 0.65 elsewhere. That is the draft confidence below which the DSpark
drafter's guesses are not checked. With the 1.75 bpw set the Spark answers **28.1 instead of 25.3 tokens/s (+11%)**; 0.5
gave 26.9 and 0.3 gave 27.4 (2 rounds of 3 prompts each, 26 October 2026). The 2.00 bpw set gains about 2%: more of it is
read from the SSD. Greedy answers are the same, because the full model checks every draft.

Where DeepSeek's time goes on a GB10 (Nsight Systems and the engine's `DS4_PROF`):

- The GPU is busy 92-94% of the time.
- Its dense Q5_K matrix products (5.9 GB per step) are about 40% of that, at 160-180 GB/s. Strata's own K-quant GEMV is
  only faster on some of these shapes (`q5k_gemv_bench`).
- The routed experts read at about 175 GB/s.
- About 10% is spent waiting on the CPU lane and between kernels. The engine reads each layer's routing back to the host,
  so it cannot use CUDA graphs.

Tried and not kept: less memory left to the OS (`DS4_UMA_HEADROOM_GIB` 3 or 2 instead of 6). It puts 650 more experts
on the GPU but measured within noise.

## How these were measured

One DGX Spark (GB10, 128 GB, DGX OS, kernel 7.0.0-1019-nvidia, driver 580.178.04), engines built on the Spark with CUDA
13.0.88 for `sm_121`, 24 October 2026. Every number is the server's own timing, medians.

### Qwen3.8-Flash-Next NVFP4

Maximilian228's GPTQ pack, 128K context, 8-bit KV cache, pictures on, `tools/strata_bench.py` (`./bench.sh`): 9 chat
replies, 3 prompts each of 4K and 16K tokens, 15 agent turns.

| | experts copied to RAM (before) | `--mmap-experts` (setup's default here) | the same on engine 0.1.49.2, tuned |
| --- | --- | --- | --- |
| experts in the GPU's cache | 13,702 of 24,576 | 24,576 of 24,576 | 24,576 of 24,576 |
| chat replies | 40.6 tokens/s | 44.4 tokens/s | 44.7 tokens/s |
| 4K prompt | 1,222 tokens/s | 1,322 tokens/s | 1,330 tokens/s |
| 16K prompt | 1,397 tokens/s | 1,471 tokens/s | 1,471 tokens/s |
| first token, 16K prompt | 11.55 s | 10.97 s | 10.96 s |
| agent turns, replies | 43.6 tokens/s | 43.1 tokens/s | 44.6 tokens/s |

On 26 October 2026, engine 0.1.51 with the Spark speed settings (8-bit hyper-connection read, `--mtp-q4 all`, batched
PLE; the default draft vocabulary), the same suite: chat replies **46.9 tokens/s** (43.1-62.2), 4K prompt 1,312
tokens/s, 16K prompt 1,453 tokens/s (first token 11.10 s), agent turns 46.7 tokens/s, 70% of the agent's prompt
reused. `--draft-vocab en` adds about 4.5% to answers in English.

The first two columns are engine 0.1.48.1. Setup's tuning chose `--pool-workers 4` (43.9 tokens/s in its own
measurement); its PCIe-share step changes nothing here (44.0-44.2 tokens/s at every value), because every expert is
already on the GPU.

Prompt reading is about 55% of an RTX card that holds the whole model ([NVFP4.md](NVFP4.md): 2,503 tokens/s at 4-8K):
the GB10 has 48 SMs and 273 GB/s of memory bandwidth. The Blackwell FP4 prompt mode (`STRATA_PREFILL_NVFP4=w4a4`) runs on
it but is not faster (870-1,159 against 1,012-1,113 tokens/s on three ~10K-token prompts), so the default stays.

### Qwen3.8-Flash-Next IQ3_XXS, and against llama.cpp

ISTA-DASLab's GSQ-RCO IQ3_XXS GGUF, read by both engines, 64K context, pictures off, engine 0.1.48.1. The method of
[the 2026-10-06 comparison](../bench/results/2026-10-06-llamacpp-0.6.0/README.md): 3 rounds, 256-token replies,
temperature 0 unless noted; full results in
[bench/results/2026-10-24-dgx-spark-vs-llamacpp](../bench/results/2026-10-24-dgx-spark-vs-llamacpp/README.md).

| | Strata | llama.cpp v0.6.0 | llama.cpp with its MTP head |
| --- | --- | --- | --- |
| replies after a 4K prompt | **51.1 tokens/s** (46.7 at temperature 0.7) | 29.7 tokens/s | 36.2 tokens/s |
| replies after a 32K prompt | **52.8 tokens/s** (38.9 at temperature 0.7) | 25.1 tokens/s | 34.7 tokens/s |
| reading a 4K prompt | **1,266 tokens/s** | 952 tokens/s | 881 tokens/s |
| reading a 32K prompt | **1,469 tokens/s** | 872 tokens/s | 778 tokens/s |

Re-measured on 26 October 2026 (Strata only, engine 0.1.51, the Spark speed settings, the same 3 rounds): replies
after a 4K prompt **52.7 tokens/s** (49.2 at temperature 0.7), after a 32K prompt 46.7 (39.1; the 32K replies ranged
42.6-53.4 over the rounds), reading a 4K prompt 1,258 tokens/s and a 32K prompt 1,496 tokens/s.

llama.cpp ran with every layer on the GPU (`-ngl 99 -b 2048 -ub 2048`, 8-bit KV, flash attention); its settings were not
tuned for the Spark.

### DeepSeek-V4.1-Flash

IST-DAS 1.75-bit and 2.00-bit sets, setup's configuration (8K context, DSpark drafter on), engine 0.1.49.2, temperature
0, no thinking. Each run: 6 short questions (256-token replies) and 2 prompts of 3,709 tokens.

| | replies, short questions | reading a 3,709-token prompt | replies after it |
| --- | ---: | ---: | ---: |
| **1.75-bit**, started with memory empty (107.3 GB of experts on the GPU, 15.2 GB from the SSD) | 23.5 tokens/s (19.4-36.3) | 207.4 tokens/s | 22.8 tokens/s |
| **1.75-bit**, the same server's second run | 33.0 tokens/s (22.7-43.5) | 298.5 tokens/s | 21.1 tokens/s |
| **1.75-bit**, the engine built on a Mac (`spark/build_on_mac.sh`) | 26.7 tokens/s | 193.9 tokens/s | 20.8 tokens/s |
| **2.00-bit**, started with memory empty (106.8 GB on the GPU, 33.2 GB from the SSD) | 17.4 tokens/s (12.2-26.3) | 175.1 tokens/s | 17.5 tokens/s |
| **2.00-bit**, started with the model files cached | 28.4 tokens/s (15.2-40.8) | 174.7 tokens/s | 17.1 tokens/s |

On 26 October 2026, engine 0.1.51 with `--dspark-conf 0.4` (setup's Spark setting), the same questions on a fresh
server whose model files were still in the page cache, then a second run: the **1.75-bit** set answered 31.8, then
**37.0 tokens/s** (prompt 185, then 286 tokens/s; after the prompt 22.6, then 25.4), with 108.2 GB of experts on the GPU;
the **2.00-bit** set 20.3, then 27.7 tokens/s (prompt 179, then 252 tokens/s).

Replies get faster as a model warms up, because the experts it reads from the SSD stay in memory. The DSpark drafter
accepted 80-95% of its tokens on these questions. For comparison, the DeepSeek release measured 4.7 tokens/s on an RTX
5070 with 64 GB of RAM.

The 1.75-bit set at 128K context (`--max-context 131072`): one 30,965-token prompt read in 77 s (403 tokens/s), reply at
23.2 tokens/s, the document's last line quoted correctly.

Before the arm64 build's change to DeepSeek's memory placement, the same engine put only 42 GB of experts on the GPU when
the page cache was full (10-12 tokens/s), and was killed by Linux's out-of-memory killer when memory was empty.

## What was checked, and what was not

- Checked on a DGX Spark: setup, starting, tuning, `./bench.sh`, chat, pictures and tool calls with Qwen NVFP4; Qwen
  IQ3_XXS in the llama.cpp comparison; DeepSeek 1.75-bit and 2.00-bit setup, chat, tool calls and speed; requests from
  another computer on the network with an API key.
- Checked: the engines build for arm64 with CUDA 13.0 on the Spark and on a Mac (`spark/build_on_mac.sh`). Setup's GB10
  and arm64 handling has tests: `python -m unittest tools.test_setup_spark`.
- Not checked: DeepSeek 2.25-bit and UD-Q8_K_XL, DeepSeek 2.00-bit at 128K context, the other Qwen families on a Spark,
  long sessions, and `strata bench run` (the standard benchmark).

If you run Strata on a Spark, please share what you see: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
