#!/bin/sh
# Build the Strata engine for an NVIDIA DGX Spark on an Apple Silicon Mac: spark/build_engine.sh in an arm64 Linux
# container with the CUDA 13.0 toolkit. An Apple Silicon Mac runs that container natively (no emulation), and nvcc needs
# no GPU to compile.
#
#   spark/build_on_mac.sh                    # -> out-spark/engine (and out-spark/strata-spark-engine.tar.gz)
#   BUILD_VISION=0 spark/build_on_mac.sh     # text only (faster)
#
# Needs Docker Desktop. Give it CPUs and memory first (Settings > Resources): nvcc takes about 2 GB per job, so JOBS
# defaults to the container's memory / 3 GB. A first build is 30-90 minutes; a second one recompiles only what changed
# (the build folders build-spark/ and build-vision-spark/ stay in the checkout).
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
image=${CUDA_IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}
command -v docker >/dev/null 2>&1 || { echo "build_on_mac.sh: Docker is needed (Docker Desktop)" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "build_on_mac.sh: Docker is not running: start Docker Desktop" >&2; exit 2; }
mem=$(docker info --format '{{.MemTotal}}')
cpus=$(docker info --format '{{.NCPU}}')
jobs=${JOBS:-$(( mem / 3000000000 ))}
[ "$jobs" -lt 1 ] && jobs=1
[ "$jobs" -gt "$cpus" ] && jobs=$cpus
echo "Docker: $cpus CPUs, $(( mem / 1073741824 )) GiB: $jobs compile jobs"
[ "$mem" -lt 12000000000 ] && echo "  (more memory in Docker Desktop > Settings > Resources makes this much faster)"

# the build runs as you, so the files it leaves in the checkout are yours; pip and the venv live in the container
docker run --rm --platform linux/arm64 \
  -v "$repo:/src" -w /src \
  -e BUILD_VISION="${BUILD_VISION:-1}" -e JOBS="$jobs" -e ARCHS="${ARCHS:-121}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  "$image" bash -euc '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential ca-certificates git python3 python3-venv \
        libgomp1 libatomic1 >/dev/null
    python3 -m venv /venv
    /venv/bin/pip install -q -r requirements.txt -r requirements-build.txt
    export PATH=/venv/bin:$PATH PYTHON=/venv/bin/python
    spark/build_engine.sh
    chown -R "$HOST_UID:$HOST_GID" out-spark build-spark build-vision-spark third_party 2>/dev/null || true
  '
tar -C "$repo/out-spark" -czf "$repo/out-spark/strata-spark-engine.tar.gz" engine
echo
echo "Engine for the DGX Spark: $repo/out-spark/engine"
echo "Unpack it INTO the Strata folder on the Spark, so that it becomes <Strata folder>/engine (docs/DGX_SPARK.md), e.g."
echo "  scp $repo/out-spark/strata-spark-engine.tar.gz spark:~/"
echo "  ssh spark 'tar -xzf ~/strata-spark-engine.tar.gz -C <Strata folder> && ls <Strata folder>/engine/strata'"
