// yolomaster_edge - universal, adaptive YOLO-Master edge runner.
// Runtime model loading (no baked-in weights), backend/classes/imgsz auto-detected
// from the model, versatile --source (image / dir / video / dataset.yaml).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "yolomaster.hpp"
#ifdef USE_ORT
#include "ort_backend.hpp"
#endif
#ifdef USE_NCNN
#include "ncnn_backend.hpp"
#endif
#include "CLI11.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cwchar>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace yolomaster;
namespace fs = std::filesystem;

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

static std::string csv_field(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char c : value) {
        if (c == '"') escaped.push_back('"');
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

static double percentile_ms(const std::vector<double>& values, double percentile) {
    if (values.empty()) return 0.0;
    std::vector<double> ordered = values;
    std::sort(ordered.begin(), ordered.end());
    const double rank = std::ceil(percentile / 100.0 * static_cast<double>(ordered.size()));
    const size_t index = std::min(ordered.size() - 1, static_cast<size_t>(std::max(1.0, rank)) - 1);
    return ordered[index];
}

// image I/O via stb (avoids OpenCV imgcodecs -> GDAL/DB/poppler dependency closure)
static cv::Mat imread_bgr(const std::string& path) {
    int w, h, n;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &n, 3);   // force 3-channel RGB
    if (!d) return cv::Mat();
    cv::Mat bgr;
    cv::cvtColor(cv::Mat(h, w, CV_8UC3, d), bgr, cv::COLOR_RGB2BGR);
    stbi_image_free(d);
    return bgr;
}
static bool imwrite_jpg(const std::string& path, const cv::Mat& bgr) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();
    return stbi_write_jpg(path.c_str(), rgb.cols, rgb.rows, 3, rgb.data, 90) != 0;
}

static int main_impl(int argc, char** argv) {
    CLI::App app{"yolomaster_edge - universal YOLO-Master edge runner (ONNX / ncnn)"};
    std::string model, source, backend = "auto", classes_opt = "auto", outdir = "runs_edge";
    std::string device = "cpu", savetxt, csv_path;
    int imgsz = 0, threads = 4, limit = 0, max_det = 300, warmup = 0, min_images = 500;
    float small_conf = -1.f, small_area = 32.f * 32.f;
    float conf = 0.001f, iou = 0.70f;
    bool no_save = false, quiet = false, multilabel = true, singlelabel = false, acceptance = false;

    app.add_option("-m,--model", model, "model: .onnx file, or ncnn dir / .param")->required();
    app.add_option("-s,--source", source, "image / directory / video / dataset.yaml")->required();
    app.add_option("-b,--backend", backend, "auto|onnx|ncnn")->default_str("auto");
    app.add_option("-d,--device", device, "cpu|cuda (onnx backend; falls back to cpu)")->default_str("cpu");
    app.add_option("--classes", classes_opt, "auto|visdrone|sku (auto = from model metadata)")->default_str("auto");
    app.add_option("--imgsz", imgsz, "inference size (0 = from model / 640)");
    app.add_option("--conf", conf, "confidence threshold (Issue #51 default: 0.001)")->capture_default_str();
    app.add_option("--iou", iou, "NMS IoU threshold (Issue #51 default: 0.70)")->capture_default_str();
    app.add_option("--max-det", max_det, "max detections per image after NMS")->capture_default_str();
    app.add_option("--small-conf", small_conf, "optional confidence for boxes below --small-area (-1 disables)")->capture_default_str();
    app.add_option("--small-area", small_area, "box area threshold for --small-conf (original-image pixels)")->capture_default_str();
    app.add_option("--threads", threads, "CPU threads")->capture_default_str();
    app.add_option("--warmup", warmup, "warmup frames before measuring (default 0)")->capture_default_str();
    app.add_option("--limit", limit, "cap #inputs (0 = all)");
    app.add_option("--min-images", min_images, "minimum images for --acceptance (default 500)")->capture_default_str();
    app.add_option("--out", outdir, "output dir for annotated results")->capture_default_str();
    app.add_option("--save-txt", savetxt, "dir to write per-image predictions ('class conf x1 y1 x2 y2')");
    app.add_option("--csv", csv_path, "write per-frame timing CSV plus a #summary row with P50/P95/P99/FPS");
    app.add_flag("--multi-label", multilabel, "one detection per class>conf per anchor (matches ultralytics val mAP)");
    app.add_flag("--single-label", singlelabel, "use only the highest-scoring class per anchor");
    app.add_flag("--acceptance", acceptance, "require at least --min-images successful inputs and fail on any input error");
    app.add_flag("--no-save", no_save, "do not write annotated outputs");
    app.add_flag("--quiet", quiet, "suppress per-image logs");
    CLI11_PARSE(app, argc, argv);

    backend = lower_copy(backend);
    device = lower_copy(device);
    classes_opt = lower_copy(classes_opt);
    if (device != "cpu" && device != "cuda") {
        std::cerr << "unknown --device value '" << device << "' (use cpu|cuda)\n";
        return 2;
    }
    // Acceptance defaults to multi-label decoding; --single-label is an
    // explicit deployment override and therefore takes precedence.
    if (singlelabel) multilabel = false;
    if (threads < 1 || warmup < 0 || limit < 0 || max_det < 1 || imgsz < 0 || min_images < 1 ||
        conf < 0.f || conf > 1.f || iou < 0.f || iou > 1.f ||
        small_conf < -1.f || small_conf > 1.f || small_area < 0.f) {
        std::cerr << "invalid numeric options (threads >= 1, max-det/min-images >= 1, thresholds in [0,1])\n";
        return 2;
    }
    if (acceptance && min_images < 500) {
        std::cerr << "--acceptance requires --min-images >= 500; use a non-acceptance run for diagnostics\n";
        return 2;
    }
    // ---- backend auto-detect from the model path ----
    if (backend == "auto") {
        std::error_code ec;
        const std::string model_lower = lower_copy(model);
        if (fs::is_directory(path_from_utf8(model), ec) || ends_with(model_lower, ".param") || ends_with(model_lower, ".bin")) backend = "ncnn";
        else if (ends_with(model_lower, ".onnx")) backend = "onnx";
        else { std::cerr << "cannot infer backend from '" << model << "'; pass --backend\n"; return 2; }
    }
    if (!fs::exists(path_from_utf8(model))) {
        std::cerr << "model path does not exist: " << model << "\n";
        return 2;
    }

    // ---- construct backend ----
    std::unique_ptr<Backend> be;
    try {
        if (backend == "onnx") {
#ifdef USE_ORT
            be = std::make_unique<OrtBackend>(model, threads, device);
#else
            std::cerr << "built without ONNXRuntime backend\n"; return 2;
#endif
        } else if (backend == "ncnn") {
#ifdef USE_NCNN
            std::string param = model, bin;
            std::error_code ec;
            if (fs::is_directory(path_from_utf8(model), ec)) {
                const fs::path preferred = path_from_utf8(model) / "model.ncnn.param";
                if (fs::is_regular_file(preferred, ec)) param = path_to_utf8(preferred);
                else {
                    for (const auto& entry : fs::directory_iterator(path_from_utf8(model), ec)) {
                        if (entry.is_regular_file() && lower_copy(entry.path().extension().string()) == ".param") {
                            param = path_to_utf8(entry.path()); break;
                        }
                    }
                }
                bin = path_to_utf8(path_from_utf8(param).replace_extension(".bin"));
            } else if (ends_with(lower_copy(param), ".bin")) {
                bin = param; param = path_to_utf8(path_from_utf8(param).replace_extension(".param"));
            } else bin = path_to_utf8(path_from_utf8(param).replace_extension(".bin"));
            if (!fs::is_regular_file(path_from_utf8(param), ec) || !fs::is_regular_file(path_from_utf8(bin), ec))
                throw std::runtime_error("ncnn param/bin pair not found: " + param + " / " + bin);
            be = std::make_unique<NcnnBackend>(param, bin, threads);
#else
            std::cerr << "built without ncnn backend\n"; return 2;
#endif
        } else { std::cerr << "unknown backend: " << backend << "\n"; return 2; }
    } catch (const std::exception& e) {
        std::cerr << "backend init failed: " << e.what() << "\n"; return 3;
    }

    // ---- resolve config: --flag > model metadata > default ----
    Config cfg;
    cfg.conf_thresh = conf;
    cfg.iou_thresh = iou;
    cfg.max_det = max_det;
    cfg.multi_label = multilabel;
    cfg.small_conf_thresh = small_conf;
    cfg.small_area = small_area;
    int want = imgsz > 0 ? imgsz : (be->meta_imgsz > 0 ? be->meta_imgsz : 640);
    if (be->fixed_imgsz > 0 && want != be->fixed_imgsz) {
        std::cerr << "[warn] model requires fixed imgsz=" << be->fixed_imgsz
                  << "; overriding requested imgsz=" << want << "\n";
        want = be->fixed_imgsz;
    }
    cfg.imgsz = want;
    std::string classes_src;
    if (classes_opt == "visdrone") { cfg.class_names = visdrone_classes(); classes_src = "flag:visdrone"; }
    else if (classes_opt == "sku" || classes_opt == "sku110k") { cfg.class_names = sku110k_classes(); classes_src = "flag:sku"; }
    else if (!be->meta_names.empty()) { cfg.class_names = be->meta_names; classes_src = "model-metadata"; }
    else if (classes_opt == "auto" && lower_copy(source).find("sku") != std::string::npos) {
        cfg.class_names = sku110k_classes(); classes_src = "fallback:path-sku";
    }
    else if (classes_opt == "auto") { cfg.class_names = visdrone_classes(); classes_src = "fallback:visdrone"; }
    else {
        std::cerr << "unknown --classes value '" << classes_opt << "' (use auto|visdrone|sku)\n";
        return 2;
    }

    std::cout << "[model] " << model << "  backend=" << backend << "  ep=" << be->active_ep
              << "  imgsz=" << cfg.imgsz << "  nc=" << cfg.num_classes() << " (" << classes_src << ")"
              << "  conf=" << cfg.conf_thresh << "  iou=" << cfg.iou_thresh << "  max_det=" << cfg.max_det
              << "  multi_label=" << (cfg.multi_label ? "true" : "false")
              << "  small_conf=" << cfg.small_conf_thresh << "\n";

    if (!no_save) { std::error_code ec; fs::create_directories(path_from_utf8(outdir), ec); }
    if (!savetxt.empty()) { std::error_code ec; fs::create_directories(path_from_utf8(savetxt), ec); }
    std::ofstream csv;
    if (!csv_path.empty()) {
        if (path_from_utf8(csv_path).has_parent_path()) {
            std::error_code ec;
            fs::create_directories(path_from_utf8(csv_path).parent_path(), ec);
        }
        csv.open(path_from_utf8(csv_path));
        if (!csv) { std::cerr << "cannot open timing CSV: " << csv_path << "\n"; return 2; }
        csv << "tag,pre_ms,infer_ms,post_ms,total_ms,detections,mean_ms,p50_ms,p95_ms,p99_ms,fps\n";
    }

    // ---- run over the source ----
    const SourceKind kind = classify_source(source);
    if (kind == SourceKind::Unknown) {
        std::cerr << "unsupported source (expected image, directory, video, or dataset yaml): " << source << "\n";
        return 4;
    }
    // Warmup is intentionally excluded from all reported timings.  For a video
    // source we leave warmup disabled because consuming frames would alter the
    // requested frame limit; image/directory/dataset sources can safely warm up.
    if (warmup > 0 && kind != SourceKind::Video) {
        auto warm_images = gather_images(source, warmup);
        for (const auto& path : warm_images) {
            cv::Mat warm = imread_bgr(path);
            if (!warm.empty()) {
                try { be->infer(warm, cfg); } catch (...) { /* measured pass reports the error */ }
            }
        }
    }
    auto t_start = std::chrono::high_resolution_clock::now();
    long frames = 0, total_dets = 0, failures = 0;
    double sum_pre = 0, sum_inf = 0, sum_post = 0;
    std::vector<double> total_times;

    std::unordered_map<std::string, int> output_counts;
    auto run_one = [&](const cv::Mat& img, const std::string& tag) {
        if (img.empty()) { std::cerr << "  [fail] unreadable: " << tag << "\n"; ++failures; return; }
        std::vector<Detection> dets;
        try {
            dets = be->infer(img, cfg);
        } catch (const std::exception& e) {
            std::cerr << "  [fail] inference error on " << tag << ": " << e.what() << "\n";
            ++failures;
            return;
        }
        frames++; total_dets += static_cast<long>(dets.size());
        sum_pre += be->pre_ms; sum_inf += be->infer_ms; sum_post += be->post_ms;
        const double total_ms = be->pre_ms + be->infer_ms + be->post_ms;
        total_times.push_back(total_ms);
        if (!quiet)
            std::cout << "  " << tag << "  dets=" << dets.size()
                      << "  infer=" << be->infer_ms << "ms\n";
        if (csv) {
            csv << csv_field(tag) << ',' << be->pre_ms << ',' << be->infer_ms << ','
                 << be->post_ms << ',' << total_ms << ',' << dets.size() << ",,,,,\n";
            if (!csv) {
                std::cerr << "  [fail] timing CSV write failed for " << tag << "\n";
                ++failures;
            }
        }
        if (!no_save) {
            cv::Mat vis = img.clone();
            draw(vis, dets, cfg);
            const std::string stem = path_to_utf8(path_from_utf8(tag).stem());
            const int ordinal = output_counts[stem]++;
            const std::string suffix = ordinal == 0 ? "" : "_" + std::to_string(ordinal);
            const auto output_path = path_to_utf8(path_from_utf8(outdir) / (stem + suffix + ".jpg"));
            if (!imwrite_jpg(output_path, vis)) {
                std::cerr << "  [warn] failed to write annotated output: " << output_path << "\n";
                ++failures;
            }
        }
        if (!savetxt.empty()) {                       // 'class conf x1 y1 x2 y2' (pixel xyxy)
            const std::string stem = path_to_utf8(path_from_utf8(tag).stem());
            const int ordinal = output_counts[stem + "#txt"]++;
            const std::string suffix = ordinal == 0 ? "" : "_" + std::to_string(ordinal);
            std::ofstream f(path_from_utf8(savetxt) / (stem + suffix + ".txt"));
            if (!f) {
                std::cerr << "  [fail] cannot write predictions for " << tag << "\n";
                ++failures;
                return;
            }
            f << std::setprecision(9);
            for (const auto& d : dets)
                f << d.class_id << ' ' << d.conf << ' ' << d.box.x << ' ' << d.box.y << ' '
                  << (d.box.x + d.box.width) << ' ' << (d.box.y + d.box.height) << '\n';
        }
    };

    if (kind == SourceKind::Video) {
#ifdef HAVE_VIDEOIO
        cv::VideoCapture cap(source);
        if (!cap.isOpened()) { std::cerr << "cannot open video: " << source << "\n"; return 4; }
        cv::Mat frame; long idx = 0;
        while (cap.read(frame)) {
            if (limit > 0 && idx >= limit) break;
            run_one(frame, source + "#" + std::to_string(idx));
            ++idx;
        }
#else
        std::cerr << "video source not supported in this portable build; use image/dir/dataset\n";
        return 4;
#endif
    } else {
        auto imgs = gather_images(source, limit);
        if (imgs.empty()) { std::cerr << "no inputs resolved from source: " << source << "\n"; return 4; }
        if (acceptance && static_cast<int>(imgs.size()) < min_images) {
            std::cerr << "acceptance requires at least " << min_images << " images, resolved " << imgs.size() << "\n";
            return 4;
        }
        if (acceptance) {
            // Prediction/label files are keyed by stem. Reject ambiguous
            // recursive-directory inputs instead of silently writing foo_1.txt
            // and making the later mAP join non-deterministic.
            std::unordered_set<std::string> stems;
            for (const auto& path : imgs) {
                const std::string stem = path_to_utf8(path_from_utf8(path).stem());
                if (!stems.insert(stem).second) {
                    std::cerr << "acceptance requires unique image stems; duplicate: " << stem << "\n";
                    return 4;
                }
            }
        }
        for (const auto& p : imgs) run_one(imread_bgr(p), p);
    }

    if (frames == 0) { std::cerr << "no frames processed\n"; return 5; }
    if (acceptance && frames < min_images) {
        std::cerr << "acceptance failed: only " << frames << " successful frames (need " << min_images << ")\n";
        return 6;
    }
    if (failures > 0) {
        std::cerr << "processing failed for " << failures << " input(s); refusing a partial-success exit\n";
        return 6;
    }
    const double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
    const double avg = (sum_pre + sum_inf + sum_post) / frames;
    const double p50 = percentile_ms(total_times, 50.0);
    const double p95 = percentile_ms(total_times, 95.0);
    const double p99 = percentile_ms(total_times, 99.0);
    const double fps = avg > 0.0 ? 1000.0 / avg : 0.0;
    if (csv) {
        // Keep aggregate evidence in the same artifact while leaving total_ms
        // blank on this row so generic per-frame CSV readers do not count it.
        csv << "#summary,,,,,," << avg << ',' << p50 << ',' << p95 << ',' << p99 << ',' << fps << '\n';
        if (!csv) {
            std::cerr << "timing CSV summary write failed\n";
            return 6;
        }
    }
    std::cout << "\n[summary] frames=" << frames << "  total_dets=" << total_dets
              << "  avg/frame: pre=" << sum_pre / frames << " infer=" << sum_inf / frames
              << " post=" << sum_post / frames << " total=" << avg << "ms"
              << "  p50=" << p50 << "ms"
              << " p95=" << p95 << "ms"
              << " p99=" << p99 << "ms"
              << "  model-FPS=" << fps << "  wall=" << wall << "s\n";
    if (!no_save) std::cout << "[saved] annotated -> " << outdir << "/\n";
    return 0;
}

#ifdef _WIN32
// Windows does not guarantee that a narrow ``main`` receives UTF-8 argv
// strings (the bytes normally use the active code page).  Start at wmain and
// convert once, so stb and std::filesystem can consume a lossless UTF-8 path.
static std::string wide_arg_to_utf8(const wchar_t* value) {
    if (!value) return {};
    const int length = static_cast<int>(std::wcslen(value));
    if (length == 0) return {};
    auto convert = [&](DWORD flags) {
        const int count = WideCharToMultiByte(CP_UTF8, flags, value, length, nullptr, 0, nullptr, nullptr);
        if (count <= 0) return std::string();
        std::string result(static_cast<size_t>(count), '\0');
        if (WideCharToMultiByte(CP_UTF8, flags, value, length, result.data(), count, nullptr, nullptr) <= 0)
            return std::string();
        return result;
    };
    if (auto utf8 = convert(WC_ERR_INVALID_CHARS); !utf8.empty()) return utf8;
    if (auto utf8 = convert(0); !utf8.empty()) return utf8;
    throw std::runtime_error("unable to convert Windows command-line argument to UTF-8");
}

int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));
    std::vector<char*> narrow_argv;
    narrow_argv.reserve(static_cast<size_t>(argc));
    try {
        for (int i = 0; i < argc; ++i) utf8_args.push_back(wide_arg_to_utf8(argv[i]));
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    for (auto& arg : utf8_args) narrow_argv.push_back(arg.data());
    return main_impl(argc, narrow_argv.data());
}
#else
int main(int argc, char** argv) {
    return main_impl(argc, argv);
}
#endif
