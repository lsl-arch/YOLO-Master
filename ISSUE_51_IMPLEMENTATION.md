# Issue #51 Implementation

This checkout implements the requirements in Tencent/YOLO-Master Issue #51:
vertical-domain EsMoE-N edge inference, ONNX plus NCNN/MNN export, parity and
accuracy gates, INT8 calibration, and cross-platform C++ packaging.

## What is included

The production example is
`examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/`.

- A single C++17 runner with ONNX Runtime and NCNN backends (MNN is covered by
  the Python conversion/parity path, so the C++ binary does not carry an MNN
  ABI dependency).
- Aspect-ratio-preserving letterbox preprocessing, RGB/NCHW conversion, class-aware
  multi-label NMS, configurable confidence/IoU/max-detection thresholds, and
  VisDrone/SKU-110K metadata fallbacks.
- Automatic ONNX/NCNN model discovery, input/output blob discovery for NCNN,
  output-layout normalization for common ONNX/NCNN tensor layouts, and explicit
  failures for invalid model shapes or runtime return codes.
- `--warmup` and `--csv` timing output with per-frame pre/infer/post/total
  milliseconds and FPS summary.
- CMake options for Linux x86_64, Windows, and portable image-only builds;
  SDK/library checks fail at configure time instead of producing a broken binary
  (`ALLOW_NO_BACKENDS=ON` is an explicit diagnostic-only escape hatch).
- Reproducible ONNX/NCNN/MNN/INT8 scripts and an mAP evaluator that uses
  per-ground-truth image indices plus `im_name` (the current `DetMetrics`
  contract). ONNX simplification and NCNN graph loading are mandatory by
  default; NCNN exports also verify dense-routing discovery and reject stale
  param/bin pairs. MNN conversion reports serialization checks separately and
  requires the raw-tensor parity helper before it can be treated as an
  acceptance artifact. Diagnostic opt-outs are explicitly marked non-compliant.
- Dependency-light contract tests plus a validation runbook for the 500-image
  accuracy gate and 300-image INT8 calibration gate.

## Build and run

From the production example's `cpp/` directory:

```bash
cmake -S . -B build -DPORTABLE=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DNCNN_ROOT=/opt/ncnn
cmake --build build --config Release
./build/yolomaster_edge \
  --model /models/esmoe_n_visdrone_sim.onnx \
  --source /data/VisDrone/images/val \
  --warmup 5 --acceptance \
  --save-txt artifacts/preds --csv artifacts/onnx.csv --no-save
```

Use an NCNN directory containing its `.param` and `.bin` pair as `--model` to
run the NCNN backend. `cpp/run_tests.sh` exercises source resolution,
overrides, timing CSVs, output naming, and failure handling once real artifacts
are supplied through environment variables.

## Acceptance protocol

`examples/YOLO-Master-Edge-Deployment/VALIDATION.md` is the executable checklist.
It requires the same ordered validation image list for PyTorch and every export,
at least 500 images for mAP50-95, at least 300 *training* images for INT8
calibration, and recorded preprocessing/NMS/runtime metadata. The issue budgets
are `<0.5%` mAP50-95 error for FP32 exports and `<1.0%` for INT8.
`scripts/eval_map.py` can enforce a selected budget with
`--reference-json ... --max-abs-delta-pct 0.5` (or `1.0` for INT8); it records
the gate result and exits non-zero on an over-budget delta. Without the
optional budget flag, smoke/reference reporting remains backward-compatible.

## Local verification status

The dependency-light Python contract tests and all production Python scripts
pass compilation in the workspace. On the Ubuntu 22.04 x86_64 target, GCC
11.4.0/CMake 3.22.1 with OpenCV 4.5.4, ONNX Runtime 1.18.1, and a static NCNN
build produced a working `yolomaster_edge` binary with both ORT and NCNN
backends. The target contract suite reported `25 passed in 4.20s`; after the
legacy YOLOv5/YOLOv7 output compatibility extension was rebuilt, a public
YOLOv5s ONNX smoke run detected 6 objects and wrote annotated output, labels,
and timing CSV (`infer=859.44 ms`, `model-FPS=1.03` for one image).

These are runtime smoke results, not an EsMoE accuracy claim. The fine-tuned
EsMoE checkpoint and VisDrone/SKU-110K split were not supplied, so the 500-image
mAP gate, 300-image INT8 calibration gate, and target-device benchmark remain
explicit follow-up acceptance steps. Do not replace that distinction with
placeholder numbers when publishing.
