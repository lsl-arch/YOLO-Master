#!/usr/bin/env bash
# Runtime robustness battery. Supply real artifacts explicitly:
#   BIN=./build/yolomaster_edge ONNX=... NCNN=... DIR=... YAML=... bash ./run_tests.sh
set -uo pipefail

BIN="${BIN:?set BIN to the compiled yolomaster_edge executable}"
ONNX="${ONNX:?set ONNX to an exported .onnx model}"
NCNN="${NCNN:?set NCNN to an exported param/bin directory}"
DIR="${DIR:?set DIR to a directory containing validation images}"
YAML="${YAML:-}"
OUT="$(mktemp -d)"
trap 'rm -rf -- "$OUT"' EXIT

first_image="$(find "$DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n 1)"
if [[ -z "$first_image" ]]; then
  echo "no images found under $DIR" >&2
  exit 2
fi

passed=0; failed=0; skipped=0
ok() { echo "  PASS  $1"; passed=$((passed + 1)); }
no() { echo "  FAIL  $1"; failed=$((failed + 1)); }
skip() { echo "  SKIP  $1"; skipped=$((skipped + 1)); }
run() { "$BIN" "$@" 2>&1; }

echo "== sources and auto-detection =="
run -m "$ONNX" -s "$first_image" --no-save | grep -q "backend=onnx" && ok "T1 ONNX auto backend" || no "T1 ONNX auto backend"
run -m "$NCNN" -s "$first_image" --no-save | grep -q "backend=ncnn" && ok "T2 NCNN auto backend" || no "T2 NCNN auto backend"
run -m "$ONNX" -s "$DIR" --limit 2 --quiet --no-save | grep -q "frames=2" && ok "T3 directory source" || no "T3 directory source"
if [[ -n "$YAML" && -f "$YAML" ]]; then
  run -m "$ONNX" -s "$YAML" --limit 2 --quiet --no-save | grep -q "frames=2" && ok "T4 dataset.yaml source" || no "T4 dataset.yaml source"
else
  skip "T4 dataset.yaml (set YAML to enable)"
fi

echo "== timing and output contracts =="
run -m "$ONNX" -s "$first_image" --warmup 1 --csv "$OUT/timing.csv" --no-save --quiet >/dev/null \
  && head -1 "$OUT/timing.csv" | grep -q "pre_ms.*infer_ms.*total_ms" \
  && ok "T5 timing CSV" || no "T5 timing CSV"
run -m "$ONNX" -s "$first_image" --out "$OUT/annotated" >/dev/null 2>&1 \
  && find "$OUT/annotated" -type f -name '*.jpg' | grep -q . \
  && ok "T6 annotated output" || no "T6 annotated output"

echo "== overrides and error handling =="
run -m "$ONNX" -s "$first_image" --classes sku --conf 0.5 --no-save | grep -q "nc=1" \
  && ok "T7 class/threshold override" || no "T7 class/threshold override"
run -m /no/such/model.onnx -s "$first_image" --no-save >/dev/null 2>&1; [[ $? -ne 0 ]] \
  && ok "T8 missing model is non-zero" || no "T8 missing model is non-zero"
run -m "$ONNX" -s /no/such/image.jpg --no-save >/dev/null 2>&1; [[ $? -ne 0 ]] \
  && ok "T9 missing source is non-zero" || no "T9 missing source is non-zero"
run -m model.foo -s "$first_image" --no-save 2>&1 | grep -qi "cannot infer backend" \
  && ok "T10 unknown extension asks for backend" || no "T10 unknown extension asks for backend"
"$BIN" -m "$ONNX" >/dev/null 2>&1; [[ $? -ne 0 ]] \
  && ok "T11 missing required source is non-zero" || no "T11 missing required source is non-zero"
run --help 2>&1 | grep -q "universal YOLO-Master" && ok "T12 help" || no "T12 help"

echo "== summary =="
echo "RESULT: $passed passed, $failed failed, $skipped skipped"
[[ $failed -eq 0 ]]
