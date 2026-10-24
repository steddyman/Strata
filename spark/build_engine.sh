#!/bin/sh
# Build the Strata CUDA engine for an NVIDIA DGX Spark (GB10, sm_121, arm64 Linux) into a ready-made engine folder.
#
# Runs on arm64 Linux with a CUDA 13.0 toolkit: inside the container spark/build_on_mac.sh starts, or on the Spark itself.
#
#   spark/build_engine.sh                    # engine + picture reader -> ./out-spark/engine
#   BUILD_VISION=0 spark/build_engine.sh     # text only (faster)
#   BUILD_DSV41=0 spark/build_engine.sh      # without the DeepSeek-V4.1 engine (strata-dsv41)
#   ARCHS="121" OUT=/tmp/eng JOBS=8 spark/build_engine.sh
#
# The result is the folder setup.py keeps its ready-made engine in: copy it to Strata/engine on the Spark and run
# ./setup.sh there (docs/DGX_SPARK.md). The flags are packaging/build_cuda_linux.sh's (the release build), for arm64.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
archs=${ARCHS:-121}
out=${OUT:-$repo/out-spark/engine}
build=${BUILD:-$repo/build-spark}
vision_dir=${BUILD_VISION_DIR:-$repo/build-vision-spark}
jobs=${JOBS:-$(nproc)}
py=${PYTHON:-python3}
cmake=${CMAKE:-cmake}
llama=${STRATA_GGML_DIR:-$repo/third_party/llama.cpp}

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64|Linux-arm64) ;;
  *) echo "build_engine.sh: this builds on arm64 Linux (on a Mac: spark/build_on_mac.sh)" >&2; exit 2 ;;
esac
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$cudafull" in
  13.0|13.1) ;;
  13.2) echo "build_engine.sh: warning: CUDA 13.2.0 / 13.2.1 miscompiled the engine for sm_120 (13.2.2 is fixed;" \
             "docs/TROUBLESHOOTING.md); 13.0 is the release toolkit" >&2 ;;
  13.*) echo "build_engine.sh: note: CUDA $cudafull has not been tried for this build; 13.0 is the release toolkit" >&2 ;;
  *) echo "build_engine.sh: needs the CUDA 13 toolkit (nvcc here: ${cudafull:-missing})" >&2; exit 2 ;;
esac
if [ ! -f "$llama/ggml/CMakeLists.txt" ]; then
  # llama.cpp at the pinned commit (ggml for the engine, mtmd for pictures), exactly as setup.py fetches it
  (cd "$repo" && "$py" -c "import setup; print(setup.get_llama_cpp())")
fi

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
last=$(echo "$archs" | tr ';,' '  ' | awk '{print $NF}')
cd "$repo"
"$cmake" -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_BUILD_DSV41="$([ "${BUILD_DSV41:-1}" = 1 ] && echo ON || echo OFF)" \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR="$llama"
"$cmake" --build "$build" --target strata strata_kq $([ "${BUILD_DSV41:-1}" = 1 ] && echo strata-dsv41) -j "$jobs"

if [ "${BUILD_VISION:-1}" = 1 ]; then
  "$cmake" -G Ninja -S tools/vision -B "$vision_dir" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$llama" \
      -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm \
      "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
  "$cmake" --build "$vision_dir" --target strata-vision -j "$jobs"
fi

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
rm -rf "$out"
mkdir -p "$out"
cp "$build/strata" "$out/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$out/"
# DeepSeek-V4.1's engine beside it: setup.py takes engine/strata-dsv41 (dsv41_exe) on a computer the DeepSeek packages do not cover
[ "${BUILD_DSV41:-1}" = 1 ] && cp "$build/strata-dsv41" "$out/"
vision_json=""
if [ "${BUILD_VISION:-1}" = 1 ] && [ -f "$vision_dir/bin/strata-vision" ]; then
  cp "$vision_dir/bin/strata-vision" "$out/"
  vision_json=", \"vision_archs\": \"$archs\""
fi
# the CUDA libraries the engine loads (cuBLAS, cuBLASLt, cudart) go beside it: the Spark needs only its driver
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$out" --also 'libcublasLt.so*'
# no "lib_dirs": setup.py loads cuBLAS and cudart from NVIDIA's pip wheels (aarch64 ones exist), as for the release engine
echo "{\"version\": \"$ver\", \"source\": \"spark/build_engine.sh\", \"platform\": \"linux-aarch64\", \"archs\": \"$archs\", \"ptx\": true, \"cuda\": \"$cudafull\", \"portable\": true$vision_json}" > "$out/BUILD.json"
"$out/strata" --version 2>/dev/null || true
echo "built: $out"
