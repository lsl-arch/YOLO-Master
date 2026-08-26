# Edge Deployment of YOLO-Master-EsMoE-N — A Technical Report

End-to-end deployment of **YOLO-Master-EsMoE-N** (VisDrone) to the edge: three export formats (ONNX / NCNN / MNN), mixed-precision INT8, a single universal C++ runtime, cross-platform builds, and accuracy/latency validation against the PyTorch original that isolates *format fidelity* from *pipeline noise*.

**Source:** this hand-off contains the runtime, scripts, and validation contract;
model weights, datasets, and vendor SDKs are supplied separately by the
deployer.

> **Reproduction note.** The numerical tables in this report are the upstream
> Issue #51 experiment attached to PR #97. This source checkout does not carry
> the checkpoint, VisDrone images, exported models, or vendor SDKs, so a local
> run must regenerate the tables with the commands in `README.md` and the
> validation runbook. The runtime and scripts below are the reproducible part of
> the deliverable; they do not silently fabricate benchmark results when an
> artifact is missing.

The interesting parts of this work were not the happy paths (`model.export()` mostly works); they were the failure modes — a quantized model that silently emits **zero detections**, an mAP that reads **1.3 points low for the wrong reason**, and a 129 MB "portable" bundle that ships a **PostgreSQL client**. This report documents those, and how each was diagnosed and closed.

---

## 1. Why the model's internals dictate the deployment strategy

EsMoE-N is not a vanilla CNN. Three structural facts drove every downstream decision:

1. **Mixture-of-Experts (`ES_MOE`).** At training/inference the router sparsely selects experts. That path is export-hostile (data-dependent control flow) — but `ES_MOE.forward` switches to a **dense** unroll under `torch.onnx.is_in_onnx_export()`: a static loop over the full expert list, `Conv/Pool/Softmax/Mul/Add`, no dynamic dispatch. Crucially, the dense path is also the **numerically faithful** one — it sidesteps the sparse-inference collapse that the sparse path exhibits — so exporting *improves* determinism rather than approximating it.
2. **Area-attention (`A2C2f`).** The backbone carries transformer-style attention blocks. These reshape activations to `[1, 1600, 192]` internally, which — as we'll see — is exactly where the static-shape assumptions of downstream quantizers break.
3. **A stride-4/8/16/32 detection head** whose classification branch produces raw logits fed through a terminal sigmoid. This branch is the single most quantization-sensitive component in the network, for a concrete reason developed in §3.

The takeaway: the model exports cleanly precisely *because* the dense MoE path is standard ops — but the attention and the head are landmines for INT8 and for third-party converters.

## 2. Export pipeline

### 2.1 ONNX (opset 12, onnxsim)

Exported to a fully **static** graph — input `images [1,3,640,640]`, output `output0 [1,14,8400]` (4 box + 10 class over 8400 anchors), 628 nodes, IR 7 — and simplified with **onnxsim**. Opset 12 was chosen deliberately as the compatibility floor: it loads unchanged under ONNXRuntime 1.18 / 1.20 / 1.27 and converts cleanly to *both* NCNN and MNN, which is the operational definition of "opset-compatible" that matters here.

The export emits shape-inference warnings on the attention transposes (`.../attn/Transpose_output_0 source:{1,1600,192} target:{}`, resolved by ONNXRuntime's lenient merge). These are benign at inference — ORT resolves the shapes at runtime — but they are a **leading indicator**: any tool that requires fully static shape propagation (offline quantizers) will choke here. That prediction is borne out in §3.5. The bundled exporter still runs `onnx.checker` and an explicit `onnxsim` pass; if either fails, it refuses to label the artifact acceptance-ready.

Ultralytics metadata (class names, `imgsz`, `stride`, `task`) is embedded in the ONNX `metadata_props`, which the C++ runtime later reads to auto-configure itself — no hardcoded class tables.

### 2.2 NCNN via pnnx

Converted through **pnnx** (PyTorch/ONNX → pnnx IR → ncnn), not the legacy `onnx2ncnn`. pnnx preserves higher-level operator semantics and emits a cleaner graph. The param file was validated structurally: magic `7767517`, **561 layers / 665 blobs**, input blob `in0`, sigmoid-terminated head. A `metadata.yaml` sidecar carries the same names/imgsz so the ncnn path is self-describing like the ONNX one.

### 2.3 MNN

Converted with `mnnconvert` (ONNX → MNN, 10.8 MB) — the *same* graph as ONNX/ncnn, which lets us later prove MNN correctness by direct tensor comparison against ONNX rather than a separate mAP run (§5.3).

## 3. INT8 quantization — the substantive part

The requirement was ≤ 1.0% mAP error under INT8 with ≥ 300 calibration images. The naive route fails silently and instructively.

### 3.1 The collapse: full INT8 emits zero detections

Static per-channel INT8 over the whole graph produces a model that runs, returns the correct output tensor shape, contains **no NaNs** — and detects **nothing**. mAP = 0.0000.

Isolating the output tensor shows why. The box-regression channels are intact (`min 0, max 644, mean 210`, matching FP32 within noise); the **classification channels are uniformly zero** (`max = 0.0000`, zero scores above 0.001). The failure is entirely in the class head.

The mechanism: the classification branch emits wide-dynamic-range **logits** consumed by a sigmoid. Per-tensor/per-channel MinMax calibration maps that wide range onto 256 INT8 levels; the small positive logits that correspond to real detections fall *below one quantization step* and round to a value whose sigmoid is ~0. The nonlinearity turns a modest quantization error on the logits into a total loss of signal. Box regression, by contrast, is a smooth linear readout with no saturating nonlinearity downstream, so it tolerates INT8 comfortably. This asymmetry — **regression robust, classification catastrophic** — is the key diagnostic.

### 3.2 Localizing the sensitivity

Keeping the detection head (`/model.25/`, 85 nodes) in FP32 and quantizing everything else recovers the model to **mAP50-95 = 0.1924, −1.12%** vs PyTorch — functional, but over budget. The residual loss is not uniform; it concentrates in two more structures:

- **The MoE router.** Expert mixing is a softmax over routing logits. INT8 perturbs the routing weights, which re-weights the expert combination — a first-order change to the features, not a rounding error on them.
- **Area-attention.** Attention scores pass through a softmax whose output is sensitive to input scale; INT8 on the QK path shifts the attention distribution.

Both are the same failure class as the head: **a softmax/sigmoid amplifying a quantization perturbation.**

### 3.3 The mixed-precision recipe

The fix is node-level precision assignment: keep the three softmax/sigmoid-bearing blocks — head (`/model.25/`), attention (`/attn/`), router (`routing`), **289 nodes** — in FP32, INT8 everything else (the bulk of the convolutional compute). `scripts/quantize_int8.py` now uses this mixed-precision exclusion list and `QOperator` by default; passing `--no-default-exclude` is explicitly diagnostic and should not be used as acceptance evidence. The progression is monotonic and diagnostic:

| Configuration | mAP50-95 | Δ vs PyTorch |
|---|---|---|
| Full INT8 | 0.0000 | collapse |
| head FP32 | 0.1924 | −1.12% |
| head + attention + router FP32 | **0.1952** | **−0.84% ✅** |

Final model: **10.9 → 5.4 MB (2.0×)**, mAP50-95 error **−0.84%**, inside the 1.0% budget. This is not a lucky threshold — it's the direct consequence of removing quantization from exactly the operators that violate the "smooth, non-saturating" assumption PTQ relies on.

### 3.4 Calibration engineering

Three non-obvious details mattered:

- **Letterbox-matched calibration.** Calibrators default to a plain resize; the model is trained on **letterboxed** input. Calibrating on the wrong preprocessing biases every activation range. We pre-letterbox 300+ VisDrone *train* images (no val leakage) to 640×640 and calibrate on those, so the calibration distribution matches inference exactly.
- **QOperator, and the opset floor.** Per-channel INT8 emits `DequantizeLinear` with an `axis` attribute, which is **only valid at opset ≥ 13**; the opset-12 export must be lifted (we upgrade to 17 in-line) or the quantized model is an invalid graph. QOperator (`QLinearConv`/`QLinearMatMul`) is chosen over QDQ for CPU execution.
- **MinMax over Percentile.** Percentile/entropy calibration builds a histogram per activation tensor; on a graph with hundreds of attention/MoE intermediates × hundreds of images, that is pathologically slow for no accuracy gain here — the exclusions already remove the outlier-heavy layers, so MinMax on the remaining well-behaved convolutions is both faster and sufficient.

### 3.5 Third-party INT8 toolchains hit the attention wall

MNN's offline quantizer (`mnnquant`) aborts immediately on this model — `std::length_error: cannot create std::vector larger than max_size()` — before any calibration runs. The cause is precisely the `[1,1600,192]` attention reshapes flagged in §2.1: the quantizer allocates buffers from statically-inferred tensor dimensions, and the dynamically-shaped attention intermediate reads back as a garbage size. MNN executes this graph fine at *inference* (it resolves shapes lazily); its *quantizer* assumes static shapes. This is a limitation of the tool's static-shape contract, not of the model, and it is not configurable. The ONNXRuntime quantizer, which tolerates dynamic intermediates, is the correct vehicle for this architecture.

### 3.6 Where INT8 actually pays off

On x86 CPU, INT8 is *slower* than FP32 — measured at **137 ms/frame vs 49 ms for FP32 on the same host, ~2.8× slower** (7.2 vs 19.5 FPS). The QDQ/QOperator kernels don't engage INT8 SIMD paths that beat the well-tuned FP32 convolutions, and the FP32↔INT8 boundaries around the excluded blocks add conversion overhead. This is expected, not a defect: INT8's throughput win is a property of **INT8 tensor-core hardware** (TensorRT on Orin, NPUs), not of desktop CPUs. We therefore treat the ONNX INT8 result as the **accuracy proof** (−0.84%, in budget) and locate the **performance** validation on the TensorRT path (§8), where the same mixed-precision assignment maps onto tensor-core execution.

## 4. The inference runtime

### 4.1 Universal binary

One executable (`yolomaster_edge`) with **ONNXRuntime** and **NCNN** backends behind a common interface. MNN is validated through the Python conversion/parity helpers rather than linked into this C++ binary, keeping the portable runtime dependency surface small. Backend, class names, and input size are **auto-detected** from the model (ONNX metadata / ncnn `metadata.yaml`), so the same binary serves any exported YOLO-Master variant with no recompilation. Source can be an image, a directory, a video, or a `dataset.yaml`. The runtime smoke battery covers corrupt images, missing files, imgsz mismatch, and backend errors; run it on each target SDK before release.

### 4.2 Preprocessing

Aspect-ratio-preserving **letterbox** (min-side scale, 114 padding) → RGB `/255` NCHW, matching training. The letterbox metadata (scale, pad) is threaded through decode so boxes map back to original-image pixel coordinates in float, with no intermediate integer rounding. The ORT boundary accepts both FP32 and FP16 ONNX tensors: FP16 inputs are packed with an explicit binary16 conversion and FP16 detection outputs are converted to FP32 before decode, avoiding an accidental UINT16 tensor tag.

### 4.3 Decode, NMS, and the mAP-parity subtlety

An early version of the C++ pipeline read **1.3 mAP points low** despite bit-accurate inference. The cause was in the decode, not the model: ultralytics `val` uses **`multi_label=True`** — one detection per class scoring above threshold per anchor, not a single argmax. Reproducing that (the runtime default; `--single-label` is an explicit override) recovered the gap exactly (0.3375 → 0.3494 mAP50). NMS is **per-class** (`agnostic=False`), implemented with a class-offset trick (shift each box by `class_id × 8192` so cross-class boxes never suppress each other), and capped at 300 detections. The runtime defaults to `conf=0.001` and `iou=0.7` for VisDrone's small/dense objects; `--conf`/`--iou` remain tunable per deployment.

### 4.4 Dependency surgery for a real portable bundle

The first self-contained Linux bundle was **231 shared libraries, 129 MB** — because Ubuntu's `libopencv_imgcodecs` links **GDAL**, which transitively pulls in PostgreSQL (`libpq`), MySQL, `libpoppler` (PDF), HDF5, and the GIS stack, and `libopencv_dnn` pulls protobuf. An object detector does not need a Postgres client. We removed both by replacing `cv::imread`/`imwrite` with **stb_image** (single-header) and `cv::dnn::NMSBoxes`/`blobFromImage` with a hand-written NMS and a manual NCHW pack. That drops the OpenCV surface to **core + imgproc only**: **231 → 10 libraries, 129 → 35 MB**, at a cost of a **0.087%** detection-count difference (stb vs OpenCV JPEG decoders diverge by sub-LSB pixel values on a handful of borderline boxes) — well inside tolerance. On Linux the binary is `$ORIGIN`-rpath'd and verified to run with no `LD_LIBRARY_PATH`; on Windows the entry point starts at `wmain`, converts UTF-16 arguments to UTF-8, and stb's `STBI_WINDOWS_UTF8`/`STBIW_WINDOWS_UTF8` wrappers keep non-ASCII image/output paths lossless. The MSVC runtime is bundled so targets need no VC++ Redistributable.

## 5. Accuracy validation

### 5.1 Methodology — one metric harness for everything

Every model — PyTorch, ONNX, NCNN, MNN, INT8 — is scored through a single path: predictions at **conf 0.001, NMS iou 0.7, multi-label, cap 300** (ultralytics `val` settings), fed to ultralytics' own `DetMetrics` + `box_iou` + `match_predictions` (`eval_map.py`). This guarantees the numbers are comparable across formats and directly comparable to the ultralytics reference, rather than four subtly different mAP implementations. ONNX/ncnn predictions are produced by the C++ runtime (`--save-txt`); MNN by a Python runner replicating the identical decode. A release check can pass `--reference-json artifacts/pytorch_map.json --max-abs-delta-pct 0.5` (or `1.0` for INT8); the evaluator records the gate result and exits non-zero when the absolute relative delta exceeds the budget. Smoke runs omit the optional budget.

### 5.2 Results — 548 val images (> 500 requirement)

| Model | mAP50 | mAP50-95 | Δ mAP50-95 vs PyTorch |
|---|---|---|---|
| **PyTorch (reference)** | 0.3504 | 0.2036 | — |
| ONNX | 0.3495 | 0.2034 | **−0.02%** |
| NCNN | 0.3495 | 0.2034 | **−0.02%** |
| MNN  | 0.3495 | 0.2034 | **−0.02%** |
| INT8 (mixed) | 0.3377 | 0.1952 | **−0.84%** |

All three FP32 export formats land on **identical** mAP (0.2034) — as they should, being the same graph — at **−0.02%** from PyTorch, 25× inside the 0.5% target. INT8 is **−0.84%**, inside the 1.0% target. (The INT8 mAP50 drop is larger, −1.27%, reflecting slightly softer classification confidences at INT8; the budget is defined on mAP50-95, which passes.)

### 5.3 Numerical parity — isolating format from pipeline

Because the FP32 formats share a graph, we verify fidelity directly rather than only through mAP. Feeding **identical letterboxed inputs** to MNN and the source ONNX across 100 val images yields **max|Δ| = 0.096, mean|Δ| = 9.7e-05** on the raw `[1,14,8400]` output. The max is a single box-coordinate least-significant bit (coordinates run to ~640; 0.096 px is nothing); the mean is negligible. Detection counts over the full set are effectively equal (ONNX 157,464 vs ncnn 157,465 at conf 0.001). This distinguishes *format equivalence* from *coincidentally similar mAP*.

The same discipline caught a **false alarm** on the CUDA path: a raw `max|Δ| = 2.31` looked alarming until it was traced to FP32 box-coordinate variance in a single anchor — functional mAP was identical. A naive "max-abs-diff < ε" gate would have failed a correct model; the box-vs-class decomposition is what makes the comparison meaningful.

## 6. Latency and throughput

> **Evidence status:** The values in this section are the upstream Issue #51 /
> PR #97 reference run. They are included to document the target reporting
> format, not as measurements made by this source checkout. Attach fresh CSVs
> from the target platform before treating them as release evidence.

Per-frame inference, VisDrone val:

| Platform | Backend | infer (ms) | FPS |
|---|---|---|---|
| Windows 11 CPU | ONNX (ORT) | 37.6 | **25.4** |
| Windows 11 CPU | NCNN | 80.1 | 12.2 |
| Linux CPU (4-thread) | ONNX (ORT) | 40.0 | 25.0 |
| Linux CPU (4-thread) | **MNN** | 74.0 | 13.5 |
| Linux CPU (4-thread) | NCNN | ~80 | ~12.5 |
| Linux CPU (4-thread) | ONNX INT8 (mixed) | 137 | 7.2 |
| Linux H200 | **ONNX CUDA (C++)** | 7.8 | **~128** |

The ordering is consistent and explicable: **ORT is ~2× faster than MNN and NCNN on x86** because it is heavily x86/AVX-tuned, while MNN and NCNN are mobile/ARM-first runtimes — which is exactly why both are carried forward for the Orin, where that ranking is expected to invert. CUDA delivers a ~5× step over CPU. **INT8 is the slowest CPU row (2.8× slower than FP32 ONNX), for the reasons in §3.6** — a reminder that INT8 is a *hardware*-dependent optimization, not a free win. On x86 CPU no format beats ORT, so the "best export format" is platform-dependent, not absolute — the reason we ship three.

## 7. Cross-platform builds and distribution

> **Evidence status:** The Linux/Windows results and bundle sizes below are
> upstream reference evidence. This checkout contains the CMake and packaging
> implementation, but the current workstation has no OpenCV, ONNX Runtime, or
> NCNN SDK, so a native production build is intentionally not claimed here.

A single cross-platform **CMake** is designed to build and run on two target platforms: **Linux x86_64** (with the ONNXRuntime CUDA EP) and **Windows 11 x64** (VS 2026 / MSVC 19.5x). The upstream reference run surfaced three concrete portability issues, each fixed in the build system rather than worked around: `Ort::Session` takes `const wchar_t*` on Windows (a platform `ORTCHAR_T` shim); the prebuilt OpenCV config doesn't recognize the VS 2026 toolset and reports an empty runtime (point `OpenCV_DIR` at the concrete `vc16/lib` config); and the exe needs the MSVC runtime on clean targets (bundled via `InstallRequiredSystemLibraries`). The packaging script produces **self-contained, relocatable bundles** when run with the target SDKs; attach the generated Linux/Windows logs and archives to promote the reference numbers to release evidence.

## 8. Future work

- **NVIDIA Jetson Orin (aarch64) + TensorRT.** The same CMake builds natively on aarch64 unchanged; the next step is a TensorRT FP16/INT8 engine built on-device, where the mixed-precision assignment from §3 maps onto tensor-core/DLA execution and INT8 finally buys throughput rather than costing it. The NCNN/MNN latency ranking is expected to invert in ARM's favor here.
- **Production drone platform — DJI Manifold 3.** VisDrone is aerial/drone imagery, so the natural production target is an onboard drone computer. [DJI Manifold 3](https://enterprise.dji.com/manifold-3) is an **NVIDIA Orin NX-based** enterprise edge computer purpose-built for drones — the exact aarch64 + TensorRT path above deploys onto it directly. Validating this pipeline on the Manifold 3 exercises **real-time on-drone inference in operational conditions** (aerial surveillance, infrastructure inspection, search-and-rescue), closing the loop from VisDrone training to production drone edge deployment.

---

*Reproducibility:* the C++ runtime and all scripts (`quantize_int8.py`, `eval_map.py`, `mnn_val.py`, `mnn_parity.py`, `package_linux.sh`) are in this source package. Exported models and deployment bundles are generated locally or obtained from a project-controlled release; no external release URL is required by the implementation.
