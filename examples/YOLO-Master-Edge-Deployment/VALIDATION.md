# Edge Validation Runbook

This runbook turns the issue #51 acceptance criteria into repeatable checks. It
is deliberately separate from the model-specific C++ example: the checks here
do not require a checkpoint, a dataset download, or an ONNX/NCNN/MNN SDK.

## Fast, dependency-light checks

From the repository root:

```bash
python -m pip install -r examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/requirements-edge.txt
python -m pytest tests/test_edge_deployment_utils.py tests/test_edge_deployment_contract.py tests/test_issue51_runtime_contract.py -q
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B /tmp/yolo-master-edge-build
cmake --build /tmp/yolo-master-edge-build
```

The CMake target is a CLI/CSV smoke target. It proves that the benchmark entry
point builds and that its argument and output contract is usable without a
runtime SDK; it does **not** run neural-network inference. Use the ONNX Runtime,
NCNN, or MNN build options in `README.md` for a real backend.

For the production C++ runner (the implementation for this issue), configure
`examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/cpp` with
`-DONNXRUNTIME_ROOT=... -DNCNN_ROOT=...` and `-DPORTABLE=ON`, then run
`cpp/run_tests.sh` with real model and validation paths. The smoke scaffold and
the production runner intentionally have separate targets so a missing SDK
cannot be mistaken for a successful neural-network build. For a dependency-only
CLI diagnostic, pass `-DALLOW_NO_BACKENDS=ON`; this option is not acceptance
evidence.

## Export and tensor parity

1. Export the same checkpoint at a fixed input size to ONNX and one mobile
   format (NCNN or MNN). Keep the ONNX opset and simplification settings in the
   report. For INT8, record the calibration image count and preprocessing; use
   at least 300 images and keep the validation set separate.
2. Run PyTorch and each exported backend on the **same, ordered image list**.
   Save the raw output tensor for at least one batch per backend as `.npy`.
3. Compare tensors before decoding/NMS:

```bash
python examples/YOLO-Master-Edge-Deployment/validate_edge_outputs.py \
  --reference artifacts/pytorch.npy \
  --candidate artifacts/onnx.npy \
  --tolerance 0.005
```

The command returns exit code `0` only when shapes match and the maximum
absolute error is within tolerance. A shape mismatch is a hard failure, not a
warning. If a parity check fails, retain the first failing tensor and image so
the difference can be localized to preprocessing, model output, or decoding.

## Accuracy gate (500-image minimum)

Use one fixed list of at least 500 validation images (the issue target is 500;
548 is a useful VisDrone reference). Evaluate PyTorch and every backend with the
same `conf`, `iou`, class mapping, multi-label policy, and maximum detections.
Report mAP50 and mAP50-95 for each format, plus the delta from PyTorch:

| Gate | Required evidence |
| --- | --- |
| FP32 export | ONNX + NCNN or MNN output and mAP50-95 delta |
| INT8 (optional) | Calibration count (>=300), quantization recipe, mAP50-95 delta |
| Preprocessing | Input shape, letterbox ratio/padding, RGB/NCHW convention |
| Postprocessing | Confidence/IoU thresholds, class-aware NMS, max detections |
| Debuggability | Raw output or per-image diff for any failed gate |

Use the issue budgets as the release decision: target less than 0.5% mAP
50-95 error for non-quantized exports and less than 1.0% for INT8. Do not mix
different image lists or NMS settings when calculating deltas. The evaluator
can enforce those budgets directly when a PyTorch reference JSON is available:

```bash
python examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/scripts/eval_map.py \
  --preds artifacts/onnx_txt \
  --images /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --reference-json artifacts/pytorch_map.json \
  --max-abs-delta-pct 0.5 \
  --json artifacts/onnx_map.json
```

`--max-abs-delta-pct` is optional and is expressed as a relative percentage
(for example, `0.5` means 0.5%). It requires `--reference-json`; the command
returns a non-zero status and records `mAP50-95_delta_gate_passed: false` when
the observed absolute value of the relative delta exceeds the budget. A smoke run without this
option keeps the dependency/data smoke behavior and does not claim acceptance.

## Latency and throughput

Warm up each backend, then benchmark the same image list and thread count. Keep
preprocess, inference, and postprocess timings separate. Report count, mean,
P50, P95, P99, and FPS (`1000 / mean_ms`) for each backend and platform. Include
the CPU/GPU model, runtime version, input size, precision, and thread count;
latency numbers without that metadata are not comparable.

The utility functions in `edge_utils.py` consume the scaffold's `latency_ms`
column or the full runner's end-to-end `total_ms` column and produce the summary
statistics used in reports. The full runner also writes a `#summary` CSV row with
mean/P50/P95/P99/FPS aggregate columns; its `total_ms` cell is intentionally
blank so per-frame readers do not count the aggregate twice:

```python
import sys
from pathlib import Path

sys.path.insert(0, "examples/YOLO-Master-Edge-Deployment")
from edge_utils import read_latency_csv, summarize_latency_ms

summary = summarize_latency_ms(read_latency_csv(Path("benchmark.csv")))
print(summary)
```

Finally, run the CMake build on at least two target platforms (for example,
Linux x86_64 and Windows x64, or Linux x86_64 and Linux ARM64/Jetson) and attach
the exact configure command and generated binary name to the technical report.
