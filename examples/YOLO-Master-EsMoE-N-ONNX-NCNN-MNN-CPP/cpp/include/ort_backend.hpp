// ONNXRuntime backend for YOLO-Master-EsMoE-N (CPU, optional CUDA provider).
#pragma once
#include "yolomaster.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace yolomaster {

class OrtBackend : public Backend {
public:
    // device: "cpu" | "cuda" (falls back to CPU if the CUDA EP can't load)
    OrtBackend(const std::string& model_path, int threads = 4, const std::string& device = "cpu");
    std::vector<Detection> infer(const cv::Mat& bgr, const Config& cfg) override;

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::vector<std::string> in_names_s_, out_names_s_;
    std::vector<const char*> in_names_, out_names_;
    // ONNX exports made with ``half=True`` commonly use FLOAT16 for both the
    // input and detection output.  Keep the element type so infer() can pack
    // the host tensor correctly instead of relying on an implicit cast.
    ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

} // namespace yolomaster
