#!/usr/bin/env bash
# Build a relocatable Linux bundle around a previously compiled runner.
# Usage:
#   package_linux.sh <yolomaster_edge> [output-dir] [onnx-model] [ncnn-dir]
# The model arguments are optional; when omitted the bundle contains only the
# executable and its shared-library closure, which is useful for CI smoke tests.
# Set ORT_PROVIDER_ROOT when ONNX Runtime provider .so files are kept outside
# the directory containing libonnxruntime.so (the script also auto-discovers
# the usual sibling/provider directories).
set -euo pipefail

BIN="${1:?usage: package_linux.sh <binary> [output-dir] [onnx-model] [ncnn-dir]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
EXAMPLE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DIST="${2:-$EXAMPLE_ROOT/dist/linux-x64}"
ONNX_MODEL="${3:-}"
NCNN_DIR="${4:-}"
ORT_PROVIDER_ROOT="${ORT_PROVIDER_ROOT:-}"

# The script recreates DIST, so reject obviously unsafe targets before the
# intentional cleanup below.  A caller should pass a dedicated bundle folder.
DIST_ABS="$(readlink -f -- "$DIST" 2>/dev/null || realpath -- "$DIST")"
case "$DIST_ABS" in
  /|""|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|"$EXAMPLE_ROOT"|"$EXAMPLE_ROOT/"|"$EXAMPLE_ROOT/cpp")
    echo "refusing to recreate unsafe bundle directory: $DIST_ABS" >&2
    exit 2
    ;;
esac

command -v ldd >/dev/null || { echo "ldd is required" >&2; exit 2; }
command -v patchelf >/dev/null || { echo "patchelf is required" >&2; exit 2; }
[[ -x "$BIN" ]] || { echo "runner is not executable: $BIN" >&2; exit 2; }

# Resolve the input before recreating DIST.  Without this guard, passing an
# output directory that contains the runner would delete the only input copy
# and make the subsequent `cp` fail (or, worse, package a partial bundle).
BIN_ABS="$(readlink -f -- "$BIN" 2>/dev/null || realpath -- "$BIN")"
case "$BIN_ABS" in
  "$DIST_ABS"|"$DIST_ABS"/*)
    echo "refusing to recreate a bundle directory containing the input runner: $DIST_ABS" >&2
    exit 2
    ;;
esac

# Resolve optional model inputs before DIST is recreated as well.  Keeping a
# model under the requested output directory would otherwise delete it before
# the late copy step below.
ONNX_MODEL_ABS=""
if [[ -n "$ONNX_MODEL" ]]; then
  [[ -f "$ONNX_MODEL" ]] || { echo "ONNX model not found: $ONNX_MODEL" >&2; exit 2; }
  ONNX_MODEL_ABS="$(readlink -f -- "$ONNX_MODEL" 2>/dev/null || realpath -- "$ONNX_MODEL")"
  case "$ONNX_MODEL_ABS" in
    "$DIST_ABS"|"$DIST_ABS"/*)
      echo "refusing to recreate a bundle directory containing the ONNX model: $DIST_ABS" >&2
      exit 2
      ;;
  esac
fi
NCNN_DIR_ABS=""
if [[ -n "$NCNN_DIR" ]]; then
  [[ -d "$NCNN_DIR" ]] || { echo "NCNN directory not found: $NCNN_DIR" >&2; exit 2; }
  NCNN_DIR_ABS="$(readlink -f -- "$NCNN_DIR" 2>/dev/null || realpath -- "$NCNN_DIR")"
  case "$NCNN_DIR_ABS" in
    "$DIST_ABS"|"$DIST_ABS"/*)
      echo "refusing to recreate a bundle directory containing the NCNN directory: $DIST_ABS" >&2
      exit 2
      ;;
  esac
fi
ORT_PROVIDER_ROOT_ABS=""
if [[ -n "$ORT_PROVIDER_ROOT" ]]; then
  [[ -d "$ORT_PROVIDER_ROOT" ]] || { echo "ORT provider directory not found: $ORT_PROVIDER_ROOT" >&2; exit 2; }
  ORT_PROVIDER_ROOT_ABS="$(readlink -f -- "$ORT_PROVIDER_ROOT" 2>/dev/null || realpath -- "$ORT_PROVIDER_ROOT")"
  case "$ORT_PROVIDER_ROOT_ABS" in
    "$DIST_ABS"|"$DIST_ABS"/*)
      echo "refusing to recreate a bundle directory containing the ORT provider directory: $DIST_ABS" >&2
      exit 2
      ;;
  esac
fi

# Inspect the original executable before deleting/recreating DIST.  A broken
# input should leave any previous bundle intact for diagnosis.
LDD_INITIAL="$(ldd "$BIN_ABS" 2>&1)" || {
  echo "ldd failed for $BIN" >&2
  echo "$LDD_INITIAL" >&2
  exit 2
}
if grep -q "not found" <<<"$LDD_INITIAL"; then
  while read -r missing; do
    [[ -z "$missing" ]] && continue
    # GPU providers may be either dlopen()'d later or listed as a direct
    # NEEDED entry, and are resolved by the provider-root pass below. The
    # kernel driver is always supplied by the target host.
    case "$missing" in
      libonnxruntime_providers_*.so*|libcuda.so*|libnvidia-*.so*|libnvoptix.so*) continue ;;
    esac
    echo "unresolved dependency in $BIN: $missing" >&2
    echo "$LDD_INITIAL" >&2
    exit 2
  done < <(awk '/not found/ {print $1}' <<<"$LDD_INITIAL")
fi

rm -rf -- "$DIST"
mkdir -p "$DIST/lib" "$DIST/models"
cp -L -- "$BIN_ABS" "$DIST/yolomaster_edge"

# Never bundle glibc/loader core; those must match the target kernel.
EXCLUDE='libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|ld-linux|libresolv\.so|linux-vdso'

is_core_library() {
  [[ "$(basename -- "$1")" =~ $EXCLUDE ]]
}

# NVIDIA's kernel driver is intentionally supplied by the target host. CUDA
# provider libraries may therefore show these names as "not found" on a CPU
# build machine; every other unresolved provider dependency remains fatal.
is_external_gpu_driver() {
  case "$(basename -- "$1")" in
    libcuda.so*|libnvidia-*.so*|libnvoptix.so*) return 0 ;;
    *) return 1 ;;
  esac
}

copy_library() {
  local source="$1" base
  [[ -e "$source" || -L "$source" ]] || return 0
  base="$(basename -- "$source")"
  is_core_library "$source" && return 0
  is_external_gpu_driver "$source" && return 0
  cp -L -- "$source" "$DIST/lib/$base"
}

ldd_paths() {
  awk '/=>/ && $3 ~ /^\// {print $3} !/=>/ && $1 ~ /^\// {print $1}'
}

copy_ldd_closure() {
  local target="$1" allow_driver="${2:-0}" output missing
  output="$(ldd "$target" 2>&1)" || {
    echo "ldd failed for $target" >&2
    echo "$output" >&2
    return 1
  }
  if grep -q "not found" <<<"$output"; then
    while read -r missing; do
      [[ -z "$missing" ]] && continue
      if [[ "$allow_driver" == 1 ]] && is_external_gpu_driver "$missing"; then
        continue
      fi
      echo "unresolved dependency for $target: $missing" >&2
      echo "$output" >&2
      return 1
    done < <(awk '/not found/ {print $1}' <<<"$output")
  fi
  mapfile -t deps < <(printf '%s\n' "$output" | ldd_paths | sort -u)
  for so in "${deps[@]}"; do
    copy_library "$so"
  done
}

copy_ldd_closure "$DIST/yolomaster_edge"

# ORT loads execution providers with dlopen(), so provider .so files do not
# appear in the runner's ldd output. Discover the common sibling layout and
# optionally an explicit provider root, then copy each provider and its own
# dependency closure. This keeps CPU bundles small while making CUDA bundles
# relocatable; the NVIDIA kernel driver remains host-provided.
declare -a PROVIDERS=()
declare -a PROVIDER_ROOTS=()
if [[ -n "$ORT_PROVIDER_ROOT_ABS" ]]; then PROVIDER_ROOTS+=("$ORT_PROVIDER_ROOT_ABS"); fi
mapfile -t ORT_CORES < <(awk '/libonnxruntime\.so/ {for (i = 1; i <= NF; ++i) if ($i ~ /^\//) print $i}' <<<"$LDD_INITIAL" | sort -u)
for core in "${ORT_CORES[@]}"; do
  core_dir="$(dirname -- "$core")"
  PROVIDER_ROOTS+=("$core_dir" "$core_dir/providers")
done
declare -A SEEN_PROVIDER_ROOTS=()
for provider_root in "${PROVIDER_ROOTS[@]}"; do
  [[ -d "$provider_root" ]] || continue
  provider_root="$(readlink -f -- "$provider_root" 2>/dev/null || realpath -- "$provider_root")"
  [[ -n "${SEEN_PROVIDER_ROOTS[$provider_root]+x}" ]] && continue
  SEEN_PROVIDER_ROOTS["$provider_root"]=1
  shopt -s nullglob
  provider_files=("$provider_root"/libonnxruntime_providers_*.so*)
  shopt -u nullglob
  for provider in "${provider_files[@]}"; do
    [[ -f "$provider" ]] || continue
    copy_library "$provider"
    PROVIDERS+=("$provider")
  done
done
for provider in "${PROVIDERS[@]}"; do
  # Include the provider root and already copied libraries while resolving its
  # dependencies; permit only the explicitly documented GPU driver exception.
  LD_LIBRARY_PATH="$(dirname -- "$provider"):$DIST/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    copy_ldd_closure "$provider" 1
done

patchelf --set-rpath '$ORIGIN/lib' "$DIST/yolomaster_edge"
for so in "$DIST"/lib/*.so*; do
  [[ -e "$so" ]] || continue
  patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
done

# Verify the finished relocatable layout with its bundled library directory in
# the search path.  Keep the glibc/core exclusions above intentional, but fail
# on any third-party library that still cannot be resolved.
LDD_BUNDLED="$(LD_LIBRARY_PATH="$DIST/lib" ldd "$DIST/yolomaster_edge" 2>&1)" || {
  echo "ldd failed for bundled runner" >&2
  echo "$LDD_BUNDLED" >&2
  exit 2
}
if grep -q "not found" <<<"$LDD_BUNDLED"; then
  echo "bundle has unresolved dependencies:" >&2
  echo "$LDD_BUNDLED" >&2
  exit 2
fi

if [[ -n "$ONNX_MODEL_ABS" ]]; then
  cp -L -- "$ONNX_MODEL_ABS" "$DIST/models/"
fi
if [[ -n "$NCNN_DIR_ABS" ]]; then
  cp -a -- "$NCNN_DIR_ABS" "$DIST/models/"
fi

cat > "$DIST/README.txt" <<'EOF'
YOLO-Master-EsMoE-N edge runner (Linux x86_64)
The bundle carries the runner, ONNX Runtime execution-provider libraries, and
their non-core shared-library closure. Models are included only when supplied
to package_linux.sh. NVIDIA kernel-driver libraries (libcuda/libnvidia*) are
provided by the target host and are intentionally not bundled.

  ./yolomaster_edge --model models/<model>.onnx --source <image|dir> --out out --acceptance
  ./yolomaster_edge --model models/<ncnn-dir> --source <image|dir> --out out
  (video input requires a non-portable build with OpenCV videoio support)
  ./yolomaster_edge --help
EOF

# Re-apply the relocatable rpath to provider libraries copied after the main
# closure and verify each provider independently. The runner-only ldd check
# cannot observe dlopen() dependencies.
for so in "$DIST"/lib/*.so*; do
  [[ -e "$so" ]] || continue
  patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
done
for provider in "$DIST"/lib/libonnxruntime_providers_*.so*; do
  [[ -e "$provider" ]] || continue
  provider_ldd="$(LD_LIBRARY_PATH="$DIST/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$provider" 2>&1)" || {
    echo "ldd failed for bundled provider $provider" >&2
    echo "$provider_ldd" >&2
    exit 2
  }
  while read -r missing; do
    [[ -z "$missing" ]] && continue
    is_external_gpu_driver "$missing" && continue
    echo "bundle has unresolved provider dependency: $missing" >&2
    echo "$provider_ldd" >&2
    exit 2
  done < <(awk '/not found/ {print $1}' <<<"$provider_ldd")
done

DIST_PARENT="$(dirname -- "$DIST")"
DIST_NAME="$(basename -- "$DIST")"
ARCHIVE="$DIST_PARENT/${DIST_NAME}.tar.gz"
tar czf "$ARCHIVE" -C "$DIST_PARENT" "$DIST_NAME"
echo "bundle: $DIST"
echo "tarball: $ARCHIVE"
echo "libraries: $(find "$DIST/lib" -maxdepth 1 -type f | wc -l)"
