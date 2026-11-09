#!/bin/bash
# The Linux AppImage for AMD gfx906 cards (Radeon VII, Radeon Pro VII / Vega II / Vega II Duo, Instinct MI50 / MI60), with the
# gfx906 engine inside it (engine-bundle), so the app needs no download from the update feed to start an engine.
#
#   packaging/build_appimage_gfx906.sh --appimage <an official Strata-<ver>-linux-x86_64.AppImage>
#
# What it does, each step skipped when its output is already there (FORCE=1 redoes everything):
#   1. ROCm for gfx906: TheRock's wheels from https://rocm.nightlies.amd.com/v2/gfx906/ into $WORK/rocm-gfx906 (no sudo).
#   2. llama.cpp at setup.py's LLAMA_CPP_COMMIT into $WORK/llama.cpp.
#   3. The engine package: packaging/build_hip_linux.sh VARIANT=gfx906 (the wave64 STRATA_HIP_GFX906 build and both picture readers).
#   4. The app's payloads in apps/desktop/src-tauri/resources: node (fetch-node.sh), control (stage-control.py), harness (copied from
#      the official AppImage given with --appimage: the pinned DeepSeek Harness, already built), engine-bundle (step 3).
#   5. The UI (apps/host), the Tauri AppImage with tauri.gfx906.conf.json, then repack-appimage.sh puts the payloads back byte for byte.
#
# Needs: git, curl, gcc, squashfs-tools, and the Tauri libraries
#   sudo apt install pkg-config libwebkit2gtk-4.1-dev libssl-dev libxdo-dev libayatana-appindicator3-dev librsvg2-dev
# plus uv, cargo (rustup) and Node with npm on the PATH. Settings: WORK=~/strata-build  ROCM_VER=7.14.0a20260612  JOBS=$(nproc)
set -euo pipefail
OFFICIAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --appimage) OFFICIAL="$(realpath "$2")"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK=${WORK:-$HOME/strata-build}
ROCM_VER=${ROCM_VER:-7.14.0a20260612}
JOBS=${JOBS:-$(nproc)}
FORCE=${FORCE:-0}
RES="$SRC/apps/desktop/src-tauri/resources"
mkdir -p "$WORK"
say() { echo "== $* ($(date +%T))"; }

# 1. ROCm for gfx906
VENV="$WORK/rocm-gfx906"
if [ "$FORCE" = 1 ] || [ ! -x "$VENV/bin/rocm-sdk" ]; then
  say "ROCm $ROCM_VER for gfx906"
  uv venv -q --allow-existing --python 3.12 "$VENV"
  uv pip install -q --python "$VENV/bin/python" --index-url https://rocm.nightlies.amd.com/v2/gfx906/ "rocm[libraries,devel]==$ROCM_VER"
  uv pip install -q --python "$VENV/bin/python" cmake ninja patchelf
fi
SP="$VENV/lib/python3.12/site-packages"
export HP="$SP/_rocm_sdk_devel" CORE="$SP/_rocm_sdk_core" LIBW="$SP/_rocm_sdk_libraries_gfx906"
export PATH="$VENV/bin:$PATH"

# 2. llama.cpp at the pinned commit
LLAMA_COMMIT=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)".*/\1/p' "$SRC/setup.py")
LLAMA="$WORK/llama.cpp"
if [ "$(git -C "$LLAMA" rev-parse HEAD 2>/dev/null || true)" != "$LLAMA_COMMIT" ]; then
  say "llama.cpp $LLAMA_COMMIT"
  mkdir -p "$LLAMA"
  git -C "$LLAMA" init -q
  git -C "$LLAMA" remote get-url origin >/dev/null 2>&1 || git -C "$LLAMA" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$LLAMA" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$LLAMA" checkout -q FETCH_HEAD
fi

# 3. the engine package
PKGVER=$(sed -n 's/^PKGVER=\${PKGVER:-\([0-9.]*\)}.*/\1/p' "$SRC/packaging/build_hip_linux.sh")
ENGINE_OUT="$WORK/work-gfx906"
PKG="$ENGINE_OUT/components/engine/$PKGVER-linux-x86_64-hip-gfx906"
if [ "$FORCE" = 1 ] || [ ! -f "$PKG/engine-component.json" ]; then
  say "engine $PKGVER for gfx906"
  VARIANT=gfx906 LLAMA="$LLAMA" HP="$HP" CORE="$CORE" LIBW="$LIBW" OUT="$ENGINE_OUT" COMMIT="$(git -C "$SRC" rev-parse HEAD)" \
    JOBS="$JOBS" bash "$SRC/packaging/build_hip_linux.sh" || { tail -20 "$ENGINE_OUT/build.status"; exit 1; }
fi

# 4. the payloads
say "payloads"
"$SRC/apps/desktop/scripts/fetch-node.sh" --check >/dev/null 2>&1 || "$SRC/apps/desktop/scripts/fetch-node.sh"
"$VENV/bin/python" "$SRC/apps/desktop/scripts/stage-control.py"     # always: it copies this checkout's setup.py, strata_control and tools
if [ ! -d "$RES/harness" ]; then
  [ -n "$OFFICIAL" ] || { echo "no harness staged: give --appimage <official Strata AppImage> to copy its harness folder" >&2; exit 2; }
  X="$WORK/official-appimage"
  [ -d "$X" ] || unsquashfs -no-progress -q -o "$("$OFFICIAL" --appimage-offset)" -d "$X" "$OFFICIAL" >/dev/null
  cp -a "$(find "$X/usr/lib" -maxdepth 2 -type d -name harness | head -1)" "$RES/harness"
fi
find "$RES/engine-bundle" -mindepth 1 -maxdepth 1 -type d -name '*-linux-x86_64-*' | grep -qv -- "/$PKGVER-linux-x86_64-hip-gfx906\$" \
  && { echo "another engine is in $RES/engine-bundle: remove it first" >&2; exit 2; }
[ -d "$RES/engine-bundle/$(basename "$PKG")" ] || cp -a "$PKG" "$RES/engine-bundle/"

# 5. UI, AppImage, repack
say "UI"
npm --prefix "$SRC/apps/host" ci --no-audit --no-fund >/dev/null
npm --prefix "$SRC/apps/host" run build >/dev/null
say "AppImage"
cd "$SRC/apps/desktop"
npm ci --no-audit --no-fund >/dev/null
APPIMAGE_EXTRACT_AND_RUN=1 NO_STRIP=1 npx tauri build --bundles appimage --config src-tauri/tauri.gfx906.conf.json
VER=$(node -p 'require("./package.json").version')
BUILT=$(ls -t src-tauri/target/release/bundle/appimage/*.AppImage | head -1)
OUTFILE="$WORK/Strata-$VER-gfx906-linux-x86_64.AppImage"
say "repack (the staged payloads byte for byte)"
APPIMAGE_EXTRACT_AND_RUN=1 bash scripts/repack-appimage.sh "$BUILT" "$OUTFILE"
APPIMAGE_EXTRACT_AND_RUN=1 bash scripts/repack-appimage.sh "$OUTFILE" "" --check
say "done: $OUTFILE"
