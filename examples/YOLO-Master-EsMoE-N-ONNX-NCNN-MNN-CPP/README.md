# YOLO-Master-EsMoE-N Edge Inference C++ Runtime

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"> <img alt="NCNN" src="https://img.shields.io/badge/NCNN-Tencent-blue.svg"> <img alt="MNN" src="https://img.shields.io/badge/MNN-Alibaba-orange.svg"> <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg">

This project provides a universal C++ inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) **EsMoE-N** object-detection models, leveraging both the [ONNX Runtime](https://onnxruntime.ai/) and [NCNN](https://github.com/Tencent/ncnn) backends together with the [OpenCV](https://opencv.org/) library. A single binary runs either backend on CPU and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), auto-detecting the model format, class names, and input size. It is designed for edge deployment of vertical-domain detectors (VisDrone aerial imagery, SKU-110K, etc.).

## ✨ Benefits

- **One Universal Binary:** A single executable integrates **ONNX Runtime** and **NCNN** backends; the backend, class names, and input size are auto-detected from the model — no recompilation or dataset YAML needed at runtime. MNN is covered by the Python conversion/parity helpers.
- **Acceptance-ready validation:** The supplied scripts enforce the Issue #51 preprocessing, 500-image accuracy gate, 300-image INT8 calibration gate, and explicit tensor/latency evidence. The numerical table below is an upstream reference and must be regenerated for a new checkpoint/device.
- **Deployment-Friendly:** Cross-platform [CMake](https://cmake.org/) build producing **self-contained and relocatable bundles** for Linux x86_64 and Windows 10/11 — installable by unzip, no dependencies on the target.
- **GPU Acceleration:** Supports FP32 CPU inference and optional [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) acceleration when built with `-DUSE_CUDA=ON` against a CUDA-enabled ONNX Runtime SDK.

## ☕ Note

The exported models embed their class names and input size as ONNX/NCNN metadata, so the runtime configures itself from the model file. Post-processing follows the Issue #51 validation contract — aspect-ratio-preserving letterbox, per-class **multi-label** NMS, `conf=0.001`, `iou=0.70`, and float-coordinate box restoration.

## 📦 Exporting Models

If your project provides a pre-built model release, use that release artifact;
otherwise export your own trained [YOLO-Master](https://github.com/Tencent/YOLO-Master)
checkpoint with `scripts/export_models.py`. The helper verifies the ONNX graph,
runs an actual NCNN load/extractor smoke test, and converts the canonical
simplified graph for MNN. MNN conversion is marked as converter-checked only;
run `scripts/mnn_parity.py` with representative images before treating it as an
acceptance artifact.

```bash
python scripts/export_models.py --model runs/train/esmoe_n_visdrone/weights/best.pt \
  --out-dir exports --formats onnx ncnn --imgsz 640 --opset 12
python scripts/quantize_int8.py --fp32 exports/best.onnx \
  --train /data/VisDrone/images/train --n-calib 500 --out exports/best_int8.onnx
```

The INT8 helper defaults to the mixed-precision Issue #51 recipe (`QOperator`,
head/attention/router kept FP32). `--no-default-exclude` is available only for
diagnostic experiments and should not be used for the accuracy gate.

### Fine-tuning a vertical-domain checkpoint

The deployment package does not ship weights or datasets. Fine-tune the
repository's EsMoE-N YAML/checkpoint with the target dataset first, then pass
the resulting `best.pt` to the exporter:

```bash
yolo train model=ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml \
  data=/data/VisDrone/visdrone.yaml imgsz=640 epochs=100 batch=16 \
  project=runs/issue51 name=esmoe_n_visdrone
```

The checked-in `v0/det/yolo-master-n.yaml` is the repository's EsMoE-N
configuration (`ES_MOE` blocks). If you are starting from a published EsMoE-N
checkpoint, skip training and pass that checkpoint directly to
`scripts/export_models.py`.

For SKU-110K, substitute its dataset YAML and use the same fixed `imgsz` for
training, export, calibration, and evaluation. Keep the validation split
separate from the 300+ training images used by `quantize_int8.py`.

### ONNX

```python
from ultralytics import YOLO

# Load a trained YOLO-Master-EsMoE-N checkpoint
model = YOLO("EsMoE-N_VisDrone.pt")

# opset=12 for broad compatibility (ORT + NCNN + MNN)
# simplify=True runs onnxsim; dynamic=False fixes the input shape for C++ deployment
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

### NCNN (via pnnx) and MNN

```bash
# NCNN + MNN — the helper forces dense routing because pnnx cannot lower topk/one_hot;
# MNN conversion consumes the same canonical ONNX artifact.
python scripts/export_models.py --model EsMoE-N_VisDrone.pt \
  --formats onnx ncnn mnn --out-dir exports --imgsz 640
```

For more details on exporting, refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/). The bundled helper is preferred because it applies the NCNN routing workaround and records machine-readable checks in `export_summary.json`.

## ⚙️ Dependencies

Ensure you have the following dependencies installed (not required if you only want to smoke-test the pre-built bundles):

| Dependency                                                          | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/)                | >=1.18        | Download pre-built binaries or build from source. Use the GPU build for the CUDA Execution Provider.           |
| [NCNN](https://github.com/Tencent/ncnn/releases)                    | recent        | Tencent NCNN; on Windows use the `windows-vs2022` prebuilt.                                                     |
| [OpenCV](https://opencv.org/releases/)                              | >=4.5.0       | `core` + `imgproc`; add `videoio` when building without `-DPORTABLE=ON`.                                        |
| C++ Compiler                                                        | C++17 Support | Needed for `<filesystem>`. ([GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), MSVC 2022/2026)      |
| [CMake](https://cmake.org/download/)                                | >=3.16        | Cross-platform build system generator.                                                                         |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (Optional)| 12.x          | Required only with `-DUSE_CUDA=ON`; match the ONNX Runtime GPU build.   |
| [MNN](https://github.com/alibaba/MNN) (Optional)                    | >=3.0         | Only for the third export format / benchmarking.                                                               |

**Important Notes:**

1.  **C++17:** The requirement stems from using the `<filesystem>` library for path handling.
2.  **CUDA/ONNX Runtime pairing:** The CUDA Execution Provider is ABI-coupled to a specific CUDA major version. Use the ONNX Runtime GPU build that matches your installed CUDA Toolkit (e.g. the CUDA-12 build with CUDA 12.x). Mismatched versions lead to runtime loader errors.
3.  **FP16 ONNX:** The C++ ONNX Runtime backend accepts exports whose input and detection output tensors are either FP32 or FP16. FP16 tensors are packed/unpacked explicitly at the ORT boundary; no implicit UINT16 cast is used. The accuracy gate should still compare each precision against its matching reference artifact.
4.  **Windows UTF-8 paths:** The Windows entry point receives wide command-line arguments and converts them to UTF-8. stb image I/O and the filesystem path helpers then use native UTF-16 APIs, so non-ASCII image, dataset, output, and CSV paths are supported. Keep the terminal/font configured to display the resulting UTF-8 log text.

## 🛠️ Build Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Tencent/YOLO-Master.git
    cd YOLO-Master/examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/cpp
    ```

2.  **Create Build Directory:**

    ```bash
    mkdir build && cd build
    ```

3.  **Configure with CMake:**
    Run CMake to generate the build files. You **must** point it at your ONNX Runtime and NCNN installations via `ONNXRUNTIME_ROOT` and `NCNN_ROOT`. Adjust the paths to where you extracted the SDKs.

    ```bash
    # Example for Linux (adjust paths as needed)
    cmake .. -DPORTABLE=ON -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DNCNN_ROOT=/path/to/ncnn
    ```

    ```bat
    :: Example for Windows, from the "x64 Native Tools Command Prompt"
    cmake .. -DCMAKE_BUILD_TYPE=Release ^
      -DOpenCV_DIR=C:/dev/opencv/build/x64/vc16/lib ^
      -DONNXRUNTIME_ROOT=C:/dev/onnxruntime-win-x64 ^
      -DNCNN_ROOT=C:/dev/ncnn-windows-vs2022/x64
    ```

    ```bash
    # Linux ARM64/Jetson (use an ARM sysroot and matching ARM64 SDK builds)
    cmake .. -DPORTABLE=ON -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake \
      -DCMAKE_SYSROOT=/opt/jetson-sysroot \
      -DONNXRUNTIME_ROOT=/opt/jetson-sysroot/opt/onnxruntime \
      -DNCNN_ROOT=/opt/jetson-sysroot/opt/ncnn
    ```

    **CMake Options:**
    - `-DONNXRUNTIME_ROOT=<path>`: **(Required)** Path to the extracted ONNX Runtime library.
    - `-DNCNN_ROOT=<path>`: **(Required)** Path to the extracted NCNN library.
    - `-DCMAKE_BUILD_TYPE=Release`: (Optional) Build with optimizations.
    - `-DPORTABLE=ON`: (Recommended for image/dir inference) Slim build with no OpenCV videoio dependency.
    - `-DUSE_CUDA=ON`: enable CUDA only with a CUDA-enabled ONNX Runtime SDK; CPU builds leave this off.
    - `-DALLOW_NO_BACKENDS=ON`: diagnostic-only CLI build when vendor SDKs are unavailable; normal
      submission builds fail at configure time instead of silently producing a runner with no backend.
    - If CMake struggles to find OpenCV, set `-DOpenCV_DIR=/path/to/opencv/build`.

4.  **Build the Project:**
    Use the build tool generated by CMake (Make, Ninja, or Visual Studio).

    ```bash
    # Using CMake's generic build command (works with Make, Ninja, MSBuild)
    cmake --build . --config Release
    ```

5.  **Locate Executable:**
    The compiled executable (`yolomaster_edge`, or `yolomaster_edge.exe` on Windows) is located in the `build` directory. On Windows the required backend and OpenCV DLLs are auto-copied next to it.

For a relocatable Linux bundle, run the packager after building. Set
`ORT_PROVIDER_ROOT` when the selected ONNX Runtime distribution has provider
libraries that must be shipped; NVIDIA driver libraries remain supplied by the
target host. The script verifies the copied runtime closure:

```bash
ORT_PROVIDER_ROOT=/opt/onnxruntime/lib \
  bash ../scripts/package_linux.sh ./yolomaster_edge ../dist/yolomaster_edge
```

## 🚀 Usage

Run the executable, pointing it at a model and a source (image, directory, or
`dataset.yaml`; video is available in a non-portable build with OpenCV
`videoio`):

```bash
./yolomaster_edge --model ../../models/esmoe_n_visdrone_sim.onnx \
                  --source path/to/image_or_dir \
                  --out out --warmup 5 --csv out/timing.csv --acceptance
```

The portable image path accepts JPEG, PNG, and BMP inputs (the bundled stb
decoder is intentionally dependency-light). TIFF/WebP files should be converted
before running the portable binary; they are not classified as image sources.
Use the same JPG/JPEG/PNG/BMP-only list for calibration, parity, and mAP
evaluation so every backend sees identical inputs.

The backend is inferred from the model (`.onnx` → ONNX Runtime, an NCNN directory or `.param`/`.bin` pair → NCNN), and class names and input size are read from the model metadata. Common options:

```text
--backend      auto | onnx | ncnn        (default: auto-detect)
--device       cpu | cuda                (ONNX backend; falls back to CPU)
--conf         confidence threshold      (default 0.001; Issue #51 validation)
--iou          NMS IoU threshold         (default 0.70)
--multi-label  one detection per class > conf per anchor (default; matches Ultralytics val mAP)
--single-label deployment override: argmax class per anchor
--small-conf    optional lower threshold for boxes below --small-area (-1 off)
--small-area    original-image area used by --small-conf (default 1024 px²)
--acceptance   require >=500 successful inputs and fail on any input error
--warmup       number of warmup images excluded from timing
--csv          per-frame pre/infer/post/total CSV plus a #summary row with mean/P50/P95/P99/FPS
--save-txt     dir to write predictions  ('class conf x1 y1 x2 y2')
--out          dir for annotated outputs  --no-save / --quiet
```

On Windows, pass paths normally (for example `--source "D:\数据\验证集"`); the executable uses the Unicode command line and does not require renaming the files to ASCII. ONNX FP16 exports generated with `scripts/export_models.py --half` can be run with the same command line as FP32 models.

See `cpp/run_tests.sh` for the runtime robustness battery.
The export/quantization/parity helpers write JSON summaries and fail on missing
artifacts or incompatible tensor shapes; they are under `scripts/`.

For the mandatory accuracy gate, first dump predictions with
`--save-txt`, then run the evaluator on the complete validation directory:

```bash
python scripts/eval_map.py --preds runs/onnx_txt --images /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val --classes visdrone --label-format visdrone \
  --reference-json artifacts/pytorch_map.json --max-abs-delta-pct 0.5 \
  --json runs/onnx_map.json
```

The evaluator refuses fewer than 500 images unless `--smoke` is supplied. A
smoke run is useful for wiring checks but is not evidence for the issue gate.
Pass `--reference-json artifacts/pytorch_map.json` to emit the relative
mAP50-95 delta in the same JSON report. For an explicit acceptance budget, add
`--max-abs-delta-pct PERCENT` (for example, `0.5` for the FP32 `<0.5%` target
or `1.0` for the INT8 target). The option requires `--reference-json`, writes
`mAP50-95_delta_gate_passed` to the JSON report, and exits non-zero when the
absolute value of the relative delta is above the requested budget. Omitting the option
preserves the previous smoke/reference-report behavior.

## 📊 Results

The reference measurements below come from the upstream Issue #51 experiment
(548 VisDrone validation images, conf 0.001, NMS IoU 0.7, multi-label). Re-run
`cpp/run_tests.sh`, `scripts/eval_map.py`, and the timing CSV workflow on the
target device before treating them as production numbers; model and dataset
artifacts are intentionally not committed to this repository.

| Model                     | mAP50-95 | Δ vs PyTorch | Latency | FPS   |
| :------------------------ | :------- | :----------- | :------ | :---- |
| ONNX (ONNX Runtime, CPU)  | 0.2034   | −0.02%       | 40 ms   | 25.0  |
| NCNN (CPU)                | 0.2034   | −0.02%       | ~80 ms  | ~12.5 |
| MNN (CPU)                 | 0.2034   | −0.02%       | 74 ms   | 13.5  |
| INT8 mixed (CPU) ¹        | 0.1952   | −0.84%       | 137 ms  | 7.2   |
| ONNX CUDA (H200 GPU)      | 0.2033   | −0.03%       | 7.8 ms  | ~128  |

CPU latencies are x86 @ 4 threads on one host; mAP is identical across FP32 formats because they are of the same graph.

> ¹ **INT8 is *slower* than FP32 on CPU** (137 ms vs 49 ms on the same host). This is expected: the QDQ/QOperator kernels do not engage INT8 SIMD paths that beat the well-tuned FP32 convolutions, and the FP32↔INT8 boundaries around the mixed-precision blocks add overhead. INT8's throughput payoff requires INT8 tensor cores (TensorRT on NVIDIA Orin); the INT8 result here is an **accuracy** proof (−0.84%, within budget), with performance validation reserved for the on-device TensorRT path.

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full methodology, INT8 quantization deep-dive, and numerical parity analysis.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [YOLO-Master repository](https://github.com/Tencent/YOLO-Master).
