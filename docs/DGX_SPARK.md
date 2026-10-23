# NVIDIA DGX Spark (GB10)

Strata can run on an NVIDIA DGX Spark. **Status: experimental.** The engine builds for it, but nobody has measured it on a
Spark yet. There is no ready-made engine, app or Terminal for it: you build the engine yourself, on the Spark or on an
Apple Silicon Mac, and start Strata with `./setup.sh`.

What is different about the Spark:

- **The CPU is ARM (arm64), not x86.** Strata's own CPU expert kernels are written for x86 (AVX2 and AVX-512). The arm64
  build leaves them out. A model's CPU experts, if any, run on ggml-cpu's rows, which use the ARM vector unit (NEON).
- **The GPU is a Blackwell GB10 with compute capability 12.1 (`sm_121`).** The RTX 50 cards are 12.0 (`sm_120`). The
  engine is built for `sm_121`. The settings tuned on RTX 50 cards (`STRATA_SM120_PERF`) stay off on 12.1.
- **The CPU and the GPU share one memory (128 GB).** `nvidia-smi` shows the GPU's memory as `[N/A]`. Setup counts all of
  the RAM, less 6 GB for the system, as GPU memory, as it does for Strix Halo ([STRIX_HALO.md](STRIX_HALO.md)). The engine
  sizes its expert cache from what Linux can give back (`MemAvailable` less 6 GiB, `STRATA_UMA_HEADROOM_GIB` changes it).
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

   Replace `spark` with your Spark's host name, and `~/Strata` with where the Strata folder is.
3. On the Spark, in the Strata folder, run:

   ```sh
   ./setup.sh
   ```

   Setup finds the GB10, says "ready-made engine already installed" and installs NVIDIA's cuBLAS and CUDA runtime for
   arm64 from pip. Then it asks for the model, the context and pictures, downloads the model and starts it, as on any PC.
   The Strata page is at `http://127.0.0.1:8080`. To use it from another computer, see [INSTALL.md](INSTALL.md).

## What was checked, and what was not

- Checked: the engine (`strata`, `libstrata_kq.so`) and the picture reader (`strata-vision`) compile and link for arm64
  with CUDA 13.0 for `sm_121`, in the container `spark/build_on_mac.sh` uses. Setup's handling of the GB10 and of the
  arm64 CPU has tests: `python -m unittest tools.test_setup_spark`.
- Not checked: running on a real DGX Spark. No speed has been measured. The memory plan (how much of the 128 GB goes to
  the expert cache) uses the same rules as Strix Halo and may need tuning there.
- Not available: the Strata app, the `strata` Terminal and its `install.sh`, and the Docker images are for x86_64 only.

If you run it, please report what you see: [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md).
