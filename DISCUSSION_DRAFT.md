# Issue #51 技术总结（Discussion 草稿）

这份草稿可在完成目标设备实测后直接发布到 YOLO-Master GitHub Discussion。
它把 Issue #51 的验收证据和部署入口集中在一起；其中表格里的数值必须替换为
目标机器重新运行 `scripts/eval_map.py` 和 `--csv` 后的结果。

## 实现

- 目录：`examples/YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP/`
- 格式：ONNX（opset 12 + onnxsim）和 NCNN（pnnx param/bin），MNN 为可选第三格式。
- 推理：C++17 + CMake，ONNX Runtime/NCNN 后端自动发现，支持 Linux x86_64、Windows
  x64，以及 `cpp/aarch64-toolchain.cmake` 的 ARM64 交叉编译；MNN 通过 Python
  转换与 parity 脚本验收，避免给 C++ 包引入额外 ABI 依赖。
- 预处理：保持长宽比的 114 letterbox、BGR→RGB、NCHW `/255`；解码后按原图坐标还原。
- 后处理：VisDrone 小目标使用可调低 `conf`，`--multi-label` 与 Ultralytics 验证一致，
  按类别 NMS，支持 `--max-det`。
- 兼容性：除 YOLO-Master/YOLOv8 的无 objectness `4+nc` 输出外，ONNX 后端也接受
  YOLOv5/YOLOv7 的 `5+nc` 输出，并在解码前执行 `objectness × class_score`。

## 验证协议

使用同一有序的至少 500 张验证图像比较 PyTorch、ONNX、NCNN/MNN；INT8 只使用训练集
至少 300 张校准图像。记录输入尺寸、letterbox 参数、opset、线程数、NMS 参数和硬件。
FP32 的 mAP50-95 误差目标是 `<0.5%`，INT8 目标是 `<1.0%`。任何张量形状或有限值检查
失败都视为失败，并保留首张差异样本。

## 部署命令

```bash
cmake -S cpp -B cpp/build -DPORTABLE=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime -DNCNN_ROOT=/opt/ncnn
cmake --build cpp/build --config Release
./cpp/build/yolomaster_edge --model models/esmoe_n_visdrone_sim.onnx \
  --source /data/VisDrone/images/val --warmup 5 --acceptance \
  --csv artifacts/onnx.csv --save-txt artifacts/preds --no-save
python scripts/eval_map.py --preds artifacts/preds \
  --images /data/VisDrone/images/val --labels /data/VisDrone/labels/val \
  --reference-json artifacts/pytorch_map.json --max-abs-delta-pct 0.5 \
  --json artifacts/onnx_map.json
```

发布时附上 `export_summary.json`、mAP JSON、timing CSV、CMake configure 命令和
可复现的模型/部署仓库链接。当前仓库不包含权重、数据集或第三方 SDK，因此本地
源码验证不会冒充目标设备 benchmark。`--max-abs-delta-pct` 是可选的相对 mAP
差值门槛（FP32 用 `0.5`，INT8 用 `1.0`）；超出门槛时评测命令返回非零，并在
JSON 中写入 `mAP50-95_delta_gate_passed: false`。省略该参数不会改变 smoke 流程。

已完成的 Ubuntu 22.04 x86_64 smoke 证据：GCC 11.4/CMake 3.22 原生构建通过，
契约测试 `25 passed`；公开 YOLOv5s ONNX 单图运行检测到 6 个目标，并生成标注图、
TXT 预测和 timing CSV。该 smoke 结果只证明运行时链路，不替代 EsMoE/VisDrone
的 500 图 mAP 验收。
