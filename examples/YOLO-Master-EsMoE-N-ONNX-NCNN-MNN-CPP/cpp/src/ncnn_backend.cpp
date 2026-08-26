#include "ncnn_backend.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <algorithm>

namespace yolomaster {

using clk = std::chrono::high_resolution_clock;
static double ms_since(const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// pnnx writes a plain-text param graph.  Discover the actual input/output
// blobs instead of assuming the common in0/out0 names; custom exports often
// use images/output0 (and silently fail with the old hard-coded names).
static void discover_io(const std::string& path, std::string& input, std::string& output) {
    std::ifstream f(path_from_utf8(path));
    std::vector<std::string> tops;
    std::set<std::string> bottoms;
    std::vector<std::string> inputs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string type, layer;
        int bottom_count = 0, top_count = 0;
        if (!(ss >> type >> layer >> bottom_count >> top_count)) continue;
        if (bottom_count < 0 || top_count < 0) continue;
        std::vector<std::string> bottoms_line(static_cast<size_t>(bottom_count));
        std::vector<std::string> tops_line(static_cast<size_t>(top_count));
        for (auto& name : bottoms_line) ss >> name;
        for (auto& name : tops_line) ss >> name;
        for (const auto& name : bottoms_line) if (!name.empty()) bottoms.insert(name);
        for (const auto& name : tops_line) if (!name.empty()) tops.push_back(name);
        if (type == "Input" && !tops_line.empty()) inputs.push_back(tops_line.front());
    }
    if (!inputs.empty()) input = inputs.front();
    const std::vector<std::string> preferred = {"out0", "output0", "output", "out"};
    for (const auto& candidate : preferred) {
        if (std::find(tops.begin(), tops.end(), candidate) != tops.end()) {
            output = candidate;
            break;
        }
    }
    if (output.empty()) {
        for (auto it = tops.rbegin(); it != tops.rend(); ++it) {
            if (bottoms.find(*it) == bottoms.end()) { output = *it; break; }
        }
    }
    if (input.empty()) input = "in0";
    if (output.empty()) output = "out0";
}

static bool copy_matrix_channel_major(const ncnn::Mat& out, int expected_feat,
                                      std::vector<float>& buf, int& feat_dim, int& anchors) {
    if (out.empty() || out.elemsize != sizeof(float)) return false;
    // ncnn represents a [1, C, N] tensor as dims=3, w=N, h=C, c=1.
    if (out.dims == 1) {
        if (out.w % expected_feat != 0) return false;
        feat_dim = expected_feat; anchors = out.w / expected_feat;
        buf.resize(static_cast<size_t>(out.w));
        std::memcpy(buf.data(), out.data, static_cast<size_t>(out.w) * sizeof(float));
        return true;
    }
    if (out.dims == 2 || (out.dims == 3 && out.c == 1)) {
        const int rows = out.h;
        const int cols = out.w;
        if (rows == expected_feat) {
            feat_dim = rows; anchors = cols;
            buf.resize(static_cast<size_t>(rows) * cols);
            for (int f = 0; f < rows; ++f)
                std::memcpy(buf.data() + static_cast<size_t>(f) * cols,
                            out.row(f), static_cast<size_t>(cols) * sizeof(float));
            return anchors > 0;
        }
        if (cols == expected_feat) {
            feat_dim = cols; anchors = rows;
            buf.resize(static_cast<size_t>(feat_dim) * anchors);
            for (int a = 0; a < anchors; ++a)
                for (int f = 0; f < feat_dim; ++f)
                    buf[static_cast<size_t>(f) * anchors + a] = out.row(a)[f];
            return anchors > 0;
        }
        return false;
    }
    if (out.dims == 3 && out.c == expected_feat) {
        feat_dim = expected_feat;
        anchors = out.w * out.h;
        if (anchors <= 0) return false;
        buf.resize(static_cast<size_t>(feat_dim) * anchors);
        for (int f = 0; f < feat_dim; ++f)
            std::memcpy(buf.data() + static_cast<size_t>(f) * anchors,
                        out.channel(f), static_cast<size_t>(anchors) * sizeof(float));
        return true;
    }
    return false;
}

NcnnBackend::NcnnBackend(const std::string& param_path, const std::string& bin_path, int threads)
    : threads_(threads) {
    if (threads_ < 1) threads_ = 1;
    discover_io(param_path, in_blob_, out_blob_);
    net_.opt.num_threads = threads_;
    // Keep the terminal detection tensor in scalar FP32 storage.  Some ncnn
    // builds enable FP16/packed storage globally (especially on ARM); the
    // decoder deliberately consumes a plain feature-major float buffer.
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_bf16_storage = false;
    if (net_.load_param(param_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load param " + param_path);
    if (net_.load_model(bin_path.c_str()) != 0)
        throw std::runtime_error("ncnn: failed to load bin " + bin_path);

    // auto-read ultralytics metadata sidecar (class names + imgsz)
    const auto parent = path_from_utf8(param_path).parent_path();
    const std::string dir = parent.empty() ? std::string(".") : path_to_utf8(parent);
    std::vector<std::string> nm; int mi = 0;
    if (meta::read_ncnn_yaml(path_to_utf8(path_from_utf8(dir) / "metadata.yaml"), nm, mi)) {
        meta_names = nm; meta_imgsz = mi;
    }
    // YOLO-Master ncnn graphs bake the attention token counts at the training size,
    // so the input size is effectively fixed.
    fixed_imgsz = meta_imgsz;
}

std::vector<Detection> NcnnBackend::infer(const cv::Mat& bgr, const Config& cfg) {
    // ---- preprocess: letterbox -> ncnn RGB /255 ----
    auto t0 = clk::now();
    LetterboxInfo lb;
    cv::Mat padded = letterbox(bgr, cfg.imgsz, lb);
    if (padded.empty()) throw std::invalid_argument("ncnn: empty input image or invalid imgsz");
    ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          padded.cols, padded.rows);
    const float mean[3] = {0.f, 0.f, 0.f};
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean, norm);
    pre_ms = ms_since(t0);

    // ---- inference ----
    auto t1 = clk::now();
    ncnn::Extractor ex = net_.create_extractor();  // uses net_.opt.num_threads set in ctor
    if (ex.input(in_blob_.c_str(), in) != 0)
        throw std::runtime_error("ncnn: failed to set input blob '" + in_blob_ + "'");
    ncnn::Mat out;
    if (ex.extract(out_blob_.c_str(), out) != 0 || out.empty())
        throw std::runtime_error("ncnn: failed to extract output blob '" + out_blob_ + "'");
    infer_ms = ms_since(t1);

    // ---- reshape to channel-major [feat_dim x num_anchors] then decode ----
    auto t2 = clk::now();
    const int feat = 4 + cfg.num_classes();
    int feat_dim = 0, num_anchors = 0;
    std::vector<float> buf;
    if (!copy_matrix_channel_major(out, feat, buf, feat_dim, num_anchors))
        throw std::runtime_error("ncnn: unsupported output shape (expected 4+nc by anchors)");
    auto dets = decode(buf.data(), feat_dim, num_anchors, cfg, lb);
    post_ms = ms_since(t2);
    return dets;
}

} // namespace yolomaster
