#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "ort_backend.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
#include <utility>
#include <cstdint>
#include <cstring>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#endif

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// ORT exposes FLOAT16 tensors as their 16-bit storage, but not every SDK
// version ships the same convenience conversion helpers.  Keep the runner
// self-contained and convert IEEE-754 binary16 explicitly at the boundary.
static float fp16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    std::uint32_t frac = bits & 0x03ffu;
    std::uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            // Normalize binary16 subnormals before rebiasing the exponent.
            int e = -14;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                --e;
            }
            frac &= 0x03ffu;
            out = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13); // infinities and NaNs
    } else {
        out = sign | ((exp - 15u + 127u) << 23) | (frac << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

static std::uint16_t float_to_fp16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t mantissa = bits & 0x007fffffu;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu);
    if (exponent == 0xff) {
        // Preserve NaN payloads; a zero mantissa must remain an infinity.
        std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13);
        if (mantissa != 0 && payload == 0) payload = 1; // keep tiny NaNs NaN
        return static_cast<std::uint16_t>(sign | 0x7c00u | payload);
    }
    const int unbiased = exponent - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (unbiased < -14) {
        if (unbiased < -24) return sign;
        std::uint32_t m = mantissa | 0x00800000u;
        const int shift = -unbiased - 14;
        std::uint32_t half_m = m >> (shift + 13);
        const std::uint32_t round_bit = (m >> (shift + 12)) & 1u;
        const std::uint32_t sticky = m & ((1u << (shift + 12)) - 1u);
        if (round_bit && (sticky || (half_m & 1u))) ++half_m;
        return static_cast<std::uint16_t>(sign | half_m);
    }
    int half_exp = unbiased + 15;
    std::uint32_t half_m = mantissa >> 13;
    const std::uint32_t round_bits = mantissa & 0x1fffu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_m & 1u))) {
        ++half_m;
        if (half_m == 0x400u) {
            half_m = 0;
            ++half_exp;
            if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exp) << 10) |
                                       static_cast<std::uint16_t>(half_m));
}

static size_t tensor_element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                           std::numeric_limits<size_t>::max() / count)
            return 0;
        count *= static_cast<size_t>(dim);
    }
    return count;
}

// ORT takes the model path as wchar_t* on Windows, char* elsewhere (ORTCHAR_T).
#ifdef _WIN32
static std::wstring ort_path(const std::string& s) {
    if (s.empty()) return {};
    const int length = static_cast<int>(s.size());
    auto convert = [&](UINT code_page, DWORD flags) {
        const int count = MultiByteToWideChar(code_page, flags, s.data(), length, nullptr, 0);
        if (count <= 0) return std::wstring();
        std::wstring result(static_cast<size_t>(count), L'\0');
        if (MultiByteToWideChar(code_page, flags, s.data(), length, result.data(), count) <= 0)
            return std::wstring();
        return result;
    };
    // CLI paths are normally UTF-8 in modern terminals; retain an ACP
    // fallback for legacy Windows shells so existing native paths continue to
    // work.  Byte-wise widening corrupts any non-ASCII model path.
    if (auto utf8 = convert(CP_UTF8, MB_ERR_INVALID_CHARS); !utf8.empty()) return utf8;
    if (auto acp = convert(CP_ACP, 0); !acp.empty()) return acp;
    throw std::runtime_error("onnxruntime: unable to convert model path to UTF-16");
}
#else
static const std::string& ort_path(const std::string& s) { return s; }
#endif

OrtBackend::OrtBackend(const std::string& model_path, int threads, const std::string& device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolomaster") {
    if (threads < 1) threads = 1;
    opts_.SetIntraOpNumThreads(threads);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == "cuda") {
#ifdef USE_CUDA
        try {                                    // graceful fallback if CUDA EP can't load
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda);
            active_ep = "CUDA";
        } catch (const std::exception& e) {
            std::cerr << "[ort] CUDA EP unavailable (" << e.what() << "); using CPU\n";
            active_ep = "CPU";
        }
#else
        // CPU-only ONNX Runtime distributions intentionally do not expose the
        // CUDA provider types.  Keep the binary buildable with those headers and
        // make an explicit --device cuda request degrade to CPU at runtime.
        std::cerr << "[ort] this binary was built without USE_CUDA; using CPU\n";
        active_ep = "CPU";
#endif
    }
    try {
        session_ = std::make_unique<Ort::Session>(env_, ort_path(model_path).c_str(), opts_);
    } catch (const std::exception& e) {
        // Some ORT builds accept the CUDA provider registration but fail only
        // when the session is created (missing CUDA DLL, incompatible driver,
        // or an unsupported graph). Retry once with a clean CPU session.
        if (device != "cuda" || active_ep != "CUDA") throw;
        std::cerr << "[ort] CUDA session unavailable (" << e.what() << "); using CPU\n";
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetIntraOpNumThreads(threads);
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        active_ep = "CPU";
        session_ = std::make_unique<Ort::Session>(env_, ort_path(model_path).c_str(), cpu_opts);
    }

    const size_t n_in = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    if (n_in == 0 || n_out == 0)
        throw std::runtime_error("onnxruntime: model must expose at least one input and output");
    input_type_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    if (input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        throw std::runtime_error("onnxruntime: input tensor must be float32 or float16");
    }
    for (size_t i = 0; i < n_in; ++i)
        in_names_s_.push_back(session_->GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < n_out; ++i)
        out_names_s_.push_back(session_->GetOutputNameAllocated(i, alloc_).get());
    for (auto& s : in_names_s_) in_names_.push_back(s.c_str());
    for (auto& s : out_names_s_) out_names_.push_back(s.c_str());

    // detect a static input size (H==W>0) -> hard constraint
    {
        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4 && shape[2] > 0 && shape[2] == shape[3]) {
            fixed_imgsz = static_cast<int>(shape[2]);
            meta_imgsz = fixed_imgsz;   // authoritative over the metadata string
        }
    }

    // auto-read ultralytics-embedded metadata (class names + imgsz)
    Ort::ModelMetadata md = session_->GetModelMetadata();
    if (auto v = md.LookupCustomMetadataMapAllocated("names", alloc_))
        meta_names = meta::parse_names_dict(v.get());
    if (auto v = md.LookupCustomMetadataMapAllocated("imgsz", alloc_)) {
        const std::string s = v.get();
        const size_t p = s.find_first_of("0123456789");
        if (p != std::string::npos) meta_imgsz = std::atoi(s.c_str() + p);
    }
}

std::vector<Detection> OrtBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    if (bgr.empty()) throw std::invalid_argument("onnxruntime: empty input image");
    if (cfg.imgsz <= 0) throw std::invalid_argument("onnxruntime: imgsz must be positive");
    // ---- preprocess: letterbox -> NCHW float RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = letterbox(bgr, cfg.imgsz, lb);   // imgsz x imgsz, CV_8UC3 BGR
    if (padded.empty()) throw std::invalid_argument("onnxruntime: empty input image or invalid imgsz");
    // NCHW float RGB /255 (replaces cv::dnn::blobFromImage with swapRB=true)
    const int sz = cfg.imgsz, hw = sz * sz;
    std::vector<float> blob(3 * hw);
    for (int y = 0; y < sz; ++y) {
        const uint8_t* row = padded.ptr<uint8_t>(y);
        for (int x = 0; x < sz; ++x) {
            const uint8_t* px = row + x * 3;          // BGR
            const int idx = y * sz + x;
            blob[idx]          = px[2] * (1.0f / 255); // R
            blob[hw + idx]     = px[1] * (1.0f / 255); // G
            blob[2 * hw + idx] = px[0] * (1.0f / 255); // B
        }
    }
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    std::array<int64_t, 4> in_shape{1, 3, cfg.imgsz, cfg.imgsz};
    // The input buffer is owned by the caller; use ORT's ordinary CPU memory
    // descriptor rather than the device allocator, which is not portable
    // across ORT builds and can reject user-provided host memory.
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // A half export generally changes the graph input type as well as the
    // output type.  Use the explicit element-type overload so uint16 storage
    // is tagged FLOAT16 (the templated uint16_t overload would mean UINT16).
    std::vector<std::uint16_t> blob16;
    Ort::Value in_tensor = [&]() {
        if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            blob16.resize(blob.size());
            for (size_t i = 0; i < blob.size(); ++i) blob16[i] = float_to_fp16(blob[i]);
            return Ort::Value::CreateTensor(
                mem, static_cast<void*>(blob16.data()), blob16.size() * sizeof(std::uint16_t),
                in_shape.data(), in_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        }
        return Ort::Value::CreateTensor<float>(
            mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());
    }();
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names_.data(), &in_tensor, 1,
                              out_names_.data(), out_names_.size());
    infer_ms = ms_since(t1);

    // ---- postprocess: decode (1, feat_dim, num_anchors) ----
    auto t2 = clk::now();
    const int expected_feat = 4 + cfg.num_classes();
    // Legacy YOLOv5/YOLOv7 exports include an objectness column between the
    // box coordinates and class scores: [cx, cy, w, h, obj, cls...].  Keep
    // the canonical decoder objectness-free by folding obj into each class
    // score after reading the tensor.
    const int legacy_feat = 5 + cfg.num_classes();
    const Ort::Value* selected = nullptr;
    std::vector<int64_t> selected_shape;
    bool has_objectness = false;
    for (const auto& value : outs) {
        if (!value.IsTensor()) continue;
        auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
        // A few converters retain a trailing singleton dimension
        // ([1,features,anchors,1]); it is layout-equivalent to the canonical
        // rank-3 tensor and can be normalized without copying the buffer.
        if (shape.size() == 4 && shape[0] == 1 && shape[3] == 1)
            shape = {1, shape[1], shape[2]};
        if (shape.size() != 3 || shape[0] != 1) continue;
        const bool canonical = (shape[1] == expected_feat && shape[2] > 0) ||
                               (shape[2] == expected_feat && shape[1] > 0);
        const bool legacy = (shape[1] == legacy_feat && shape[2] > 0) ||
                            (shape[2] == legacy_feat && shape[1] > 0);
        if (canonical || legacy) {
            selected = &value;
            selected_shape = std::move(shape);
            has_objectness = legacy && !canonical;
            break;
        }
    }
    if (!selected)
        throw std::runtime_error("onnxruntime: no detection tensor with shape [1, 4+nc, anchors], [1, anchors, 4+nc], [1, 5+nc, anchors], or [1, anchors, 5+nc]");
    const auto type = selected->GetTensorTypeAndShapeInfo().GetElementType();
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        throw std::runtime_error("onnxruntime: detection output must be float32 or float16");
    const int d1 = static_cast<int>(selected_shape[1]);
    const int d2 = static_cast<int>(selected_shape[2]);
    const int raw_feat_dim = has_objectness ? legacy_feat : expected_feat;
    const int num_anchors = (d1 == raw_feat_dim) ? d2 : d1;
    const size_t raw_elements = static_cast<size_t>(raw_feat_dim) *
                                     static_cast<size_t>(num_anchors);
    if (tensor_element_count(selected_shape) != raw_elements)
        throw std::runtime_error("onnxruntime: detection tensor has an invalid element count");
    std::vector<float> raw_major;
    raw_major.resize(raw_elements);
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* raw = selected->GetTensorData<float>();
        if (d1 == raw_feat_dim) {
            std::copy(raw, raw + raw_elements, raw_major.begin());
        } else {
            for (int a = 0; a < num_anchors; ++a)
                for (int f = 0; f < raw_feat_dim; ++f)
                    raw_major[static_cast<size_t>(f) * num_anchors + a] =
                        raw[static_cast<size_t>(a) * raw_feat_dim + f];
        }
    } else {
        const std::uint16_t* raw = selected->GetTensorData<std::uint16_t>();
        if (d1 == raw_feat_dim) {
            for (size_t i = 0; i < raw_elements; ++i) raw_major[i] = fp16_to_float(raw[i]);
        } else {
            for (int a = 0; a < num_anchors; ++a)
                for (int f = 0; f < raw_feat_dim; ++f)
                    raw_major[static_cast<size_t>(f) * num_anchors + a] =
                        fp16_to_float(raw[static_cast<size_t>(a) * raw_feat_dim + f]);
        }
    }
    const float* decode_data = raw_major.data();
    std::vector<float> canonical_major;
    if (has_objectness) {
        canonical_major.resize(static_cast<size_t>(expected_feat) * static_cast<size_t>(num_anchors));
        for (int a = 0; a < num_anchors; ++a) {
            for (int f = 0; f < 4; ++f)
                canonical_major[static_cast<size_t>(f) * num_anchors + a] =
                    raw_major[static_cast<size_t>(f) * num_anchors + a];
            const float objectness = raw_major[static_cast<size_t>(4) * num_anchors + a];
            for (int c = 0; c < cfg.num_classes(); ++c)
                canonical_major[static_cast<size_t>(4 + c) * num_anchors + a] =
                    objectness * raw_major[static_cast<size_t>(5 + c) * num_anchors + a];
        }
        decode_data = canonical_major.data();
    }
    auto dets = decode(decode_data, expected_feat, num_anchors, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
