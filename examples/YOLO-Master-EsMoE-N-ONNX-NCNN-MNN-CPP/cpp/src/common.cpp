// Shared, backend/model-agnostic ops: class tables, letterbox, decode+NMS,
// drawing, model-metadata parsing, and versatile source resolution.
#include "yolomaster.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <sstream>

namespace fs = std::filesystem;

namespace yolomaster {

// Python's round() uses ties-to-even.  Resize dimensions therefore use the
// same rule, while Ultralytics' LetterBox applies a -0.1 tie-break to the
// left/top half-pad so an odd remainder is placed on the right/bottom edge.
static int round_even(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5) return static_cast<int>(lower);
    if (fraction > 0.5) return static_cast<int>(lower + 1.0);
    const int candidate = static_cast<int>(lower);
    return (candidate % 2 == 0) ? candidate : candidate + 1;
}

static int letterbox_half_pad(int residual) {
    return round_even(static_cast<double>(residual) / 2.0 - 0.1);
}

const std::vector<std::string>& visdrone_classes() {
    static const std::vector<std::string> c = {
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"};
    return c;
}
const std::vector<std::string>& sku110k_classes() {
    static const std::vector<std::string> c = {"object"};
    return c;
}

cv::Mat letterbox(const cv::Mat& img, int imgsz, LetterboxInfo& info) {
    if (img.empty() || imgsz <= 0)
        return cv::Mat();
    info.orig_w = img.cols;
    info.orig_h = img.rows;
    const double r = std::min(imgsz / static_cast<double>(img.cols),
                              imgsz / static_cast<double>(img.rows));
    // Python's round() is ties-to-even; use the same rule for resized
    // dimensions as for the half-padding below so a 0.5 pixel cannot move a
    // decoded box by one pixel between calibration and the native runner.
    const int nw = std::max(1, round_even(img.cols * r));
    const int nh = std::max(1, round_even(img.rows * r));
    info.scale = r;
    // Match ultralytics.data.augment.LetterBox: round the left/top half-pad
    // with the -0.1 tie-break, then let the right/bottom remainder absorb an
    // odd pixel.  This is important for exact mAP/calibration parity.
    info.pad_x = letterbox_half_pad(imgsz - nw);
    info.pad_y = letterbox_half_pad(imgsz - nh);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat out(imgsz, imgsz, img.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, nw, nh)));
    return out;
}

// greedy per-box NMS (score-descending, IoU suppression) - replaces
// cv::dnn::NMSBoxes; identical semantics (keep is returned score-descending).
static void nms_greedy(const std::vector<cv::Rect2d>& boxes, const std::vector<float>& scores,
                       float conf, float iou_thr, int max_keep, std::vector<int>& keep) {
    if (boxes.size() != scores.size() || max_keep <= 0) return;
    std::vector<int> order;
    order.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i]) && scores[i] > conf)
            order.push_back(static_cast<int>(i));
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    std::vector<char> dead(boxes.size(), 0);
    for (size_t m = 0; m < order.size(); ++m) {
        const int i = order[m];
        if (dead[i]) continue;
        keep.push_back(i);
        if (static_cast<int>(keep.size()) >= max_keep) break;
        for (size_t n = m + 1; n < order.size(); ++n) {
            const int j = order[n];
            if (dead[j]) continue;
            const double xx1 = std::max(boxes[i].x, boxes[j].x);
            const double yy1 = std::max(boxes[i].y, boxes[j].y);
            const double xx2 = std::min(boxes[i].x + boxes[i].width,  boxes[j].x + boxes[j].width);
            const double yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);
            const double inter = std::max(0.0, xx2 - xx1) * std::max(0.0, yy2 - yy1);
            const double uni = boxes[i].area() + boxes[j].area() - inter;
            if (uni > 0 && inter / uni > iou_thr) dead[j] = 1;
        }
    }
}

std::vector<Detection> decode(const float* out, int feat_dim, int num_anchors,
                              const Config& cfg, const LetterboxInfo& lb) {
    if (!out || feat_dim < 5 || num_anchors <= 0 || lb.scale <= 0.0 ||
        lb.orig_w <= 0 || lb.orig_h <= 0)
        return {};
    if (cfg.num_classes() > 0 && feat_dim != 4 + cfg.num_classes())
        return {};
    const int nc = feat_dim - 4;
    std::vector<cv::Rect2d> boxes;   // float boxes -> no int rounding (mAP-precise)
    std::vector<float> scores;
    std::vector<int> ids;
    for (int a = 0; a < num_anchors; ++a) {
        // qualifying classes: all > conf (multi_label, matches ultralytics val) or just argmax
        int best = -1; float bestv = 0.f;
        bool any = false;
        for (int c = 0; c < nc; ++c) {
            const float v = out[(4 + c) * num_anchors + a];
            if (!std::isfinite(v)) continue;
            if (best < 0 || v > bestv) { bestv = v; best = c; }
        }
        const float cx = out[0 * num_anchors + a];
        const float cy = out[1 * num_anchors + a];
        const float w  = out[2 * num_anchors + a];
        const float h  = out[3 * num_anchors + a];
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || w <= 0.f || h <= 0.f)
            continue;
        const double x0 = (cx - 0.5f * w - lb.pad_x) / lb.scale;
        const double y0 = (cy - 0.5f * h - lb.pad_y) / lb.scale;
        const double bw = static_cast<double>(w) / lb.scale, bh = static_cast<double>(h) / lb.scale;
        const double area = bw * bh;
        if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(bw) ||
            !std::isfinite(bh) || !std::isfinite(area) || bw <= 0.0 || bh <= 0.0)
            continue;
        const float threshold = (cfg.small_conf_thresh >= 0.f && area < cfg.small_area)
            ? std::min(cfg.conf_thresh, cfg.small_conf_thresh)
            : cfg.conf_thresh;
        if (cfg.multi_label) {
            for (int c = 0; c < nc; ++c) {
                const float v = out[(4 + c) * num_anchors + a];
                if (std::isfinite(v) && v > threshold) { any = true; break; }
            }
        }
        if (!(cfg.multi_label ? any : (best >= 0 && bestv > threshold))) continue;

        if (cfg.multi_label) {                       // one detection per class > conf
            for (int c = 0; c < nc; ++c) {
                const float v = out[(4 + c) * num_anchors + a];
                if (!std::isfinite(v) || v <= threshold) continue;
                boxes.emplace_back(x0, y0, bw, bh); scores.push_back(v); ids.push_back(c);
            }
        } else {                                     // single best class
            boxes.emplace_back(x0, y0, bw, bh); scores.push_back(bestv); ids.push_back(best);
        }
    }
    // per-class NMS (match ultralytics agnostic=False): offset boxes by class id
    // so detections of different classes never cross-suppress each other.
    // Ultralytics applies a max_nms=30000 guard before its quadratic NMS. Keep
    // the same bounded candidate set and preserve the highest-scoring boxes;
    // without it, conf=0.001 can produce tens of thousands of candidates for
    // a dense 8400-anchor model.
    constexpr size_t max_nms = 30000;
    if (boxes.size() > max_nms) {
        std::vector<int> order(boxes.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return scores[a] > scores[b];
        });
        order.resize(max_nms);
        std::vector<cv::Rect2d> limited_boxes;
        std::vector<float> limited_scores;
        std::vector<int> limited_ids;
        limited_boxes.reserve(max_nms);
        limited_scores.reserve(max_nms);
        limited_ids.reserve(max_nms);
        for (const int index : order) {
            limited_boxes.push_back(boxes[index]);
            limited_scores.push_back(scores[index]);
            limited_ids.push_back(ids[index]);
        }
        boxes.swap(limited_boxes);
        scores.swap(limited_scores);
        ids.swap(limited_ids);
    }
    std::vector<int> keep;
    {
        const double OFF = static_cast<double>(std::max(lb.orig_w, lb.orig_h)) + 1.0;
        std::vector<cv::Rect2d> off = boxes;
        for (size_t k = 0; k < off.size(); ++k) { off[k].x += ids[k] * OFF; off[k].y += ids[k] * OFF; }
        // Candidates have already passed their (possibly area-adaptive)
        // confidence threshold; use zero here so tiny-object candidates are
        // not filtered a second time by the global threshold.
        nms_greedy(off, scores, 0.f, cfg.iou_thresh, cfg.max_det, keep);
    }
    std::vector<Detection> dets;
    const cv::Rect2d frame(0, 0, lb.orig_w, lb.orig_h);
    for (int i : keep) {                             // keep is score-descending
        if (static_cast<int>(dets.size()) >= cfg.max_det) break;
        cv::Rect2d b = boxes[i] & frame;             // clip in float
        if (b.width > 0 && b.height > 0)
            dets.push_back({ids[i], scores[i],
                            cv::Rect2f(static_cast<float>(b.x), static_cast<float>(b.y),
                                       static_cast<float>(b.width), static_cast<float>(b.height))});
    }
    return dets;
}

void draw(cv::Mat& img, const std::vector<Detection>& dets, const Config& cfg) {
    for (const auto& d : dets) {
        const cv::Rect r(cvRound(d.box.x), cvRound(d.box.y), cvRound(d.box.width), cvRound(d.box.height));
        const cv::Scalar color(37 * (d.class_id + 1) % 255, 17 * (d.class_id + 3) % 255,
                               29 * (d.class_id + 5) % 255);
        cv::rectangle(img, r, color, 2);
        const std::string name = (d.class_id < cfg.num_classes()) ? cfg.class_names[d.class_id]
                                                                  : std::to_string(d.class_id);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), d.conf);
        int base = 0;
        cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(img, cv::Rect(r.x, std::max(0, r.y - ts.height - 4),
                                    ts.width + 2, ts.height + 4), color, cv::FILLED);
        cv::putText(img, buf, cv::Point(r.x, std::max(ts.height, r.y - 3)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ---------------- metadata ----------------
namespace meta {

std::vector<std::string> parse_names_dict(const std::string& s) {
    // Ultralytics has emitted both Python dictionaries ({0: 'car'}) and JSON
    // dictionaries ({"0": "car"}).  Extract values after ':' rather than all
    // quoted tokens; the latter incorrectly treats JSON keys as class names.
    std::map<int, std::string> indexed;
    std::vector<std::string> sequential;
    auto trim_local = [](std::string value) {
        const size_t first = value.find_first_not_of(" \t\r\n{}");
        const size_t last = value.find_last_not_of(" \t\r\n{},");
        return first == std::string::npos ? std::string() : value.substr(first, last - first + 1);
    };
    size_t pos = 0;
    while (true) {
        while (pos < s.size() && (s[pos] == ',' || std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == '{')) ++pos;
        const size_t key_start = pos;
        const size_t colon = s.find(':', pos);
        if (colon == std::string::npos) break;
        size_t value_start = colon + 1;
        while (value_start < s.size() && std::isspace(static_cast<unsigned char>(s[value_start]))) ++value_start;
        if (value_start >= s.size()) break;
        std::string value;
        if (s[value_start] == '\'' || s[value_start] == '"') {
            const char quote = s[value_start++];
            const size_t end = s.find(quote, value_start);
            if (end == std::string::npos) break;
            value = s.substr(value_start, end - value_start);
            pos = end + 1;
        } else {
            const size_t end = s.find(',', value_start);
            value = s.substr(value_start, end == std::string::npos ? std::string::npos : end - value_start);
            pos = end == std::string::npos ? s.size() : end + 1;
        }
        value = trim_local(value);
        if (value.empty()) continue;
        std::string key = trim_local(s.substr(key_start, colon - key_start));
        if (key.size() >= 2 && ((key.front() == '\'' && key.back() == '\'') ||
                                (key.front() == '"' && key.back() == '"'))) {
            key = key.substr(1, key.size() - 2);
        }
        char* end_ptr = nullptr;
        const long parsed = std::strtol(key.c_str(), &end_ptr, 10);
        if (end_ptr && *end_ptr == '\0' && parsed >= 0 && parsed < 1000000) {
            indexed[static_cast<int>(parsed)] = value;
        } else {
            sequential.push_back(value);
        }
    }
    if (!indexed.empty()) {
        std::vector<std::string> names;
        for (const auto& item : indexed) names.push_back(item.second);
        names.insert(names.end(), sequential.begin(), sequential.end());
        return names;
    }
    // YAML/metadata may also be a bare list: ['pedestrian', 'people'].
    std::vector<std::string> names;
    for (size_t i = 0; i < s.size();) {
        const char quote = s[i];
        if (quote != '\'' && quote != '"') { ++i; continue; }
        const size_t end = s.find(quote, i + 1);
        if (end == std::string::npos) break;
        names.push_back(s.substr(i + 1, end - i - 1));
        i = end + 1;
    }
    return names;
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static std::string yaml_scalar(std::string value) {
    value = trim(value);
    const size_t comment = value.find('#');
    if (comment != std::string::npos) value = trim(value.substr(0, comment));
    if (!value.empty() && value.front() == '[' && value.back() == ']')
        value = trim(value.substr(1, value.size() - 2));
    if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                              (value.front() == '"' && value.back() == '"')))
        value = value.substr(1, value.size() - 2);
    return trim(value);
}

bool read_ncnn_yaml(const std::string& path, std::vector<std::string>& names, int& imgsz) {
    std::ifstream f(path_from_utf8(path));
    if (!f) return false;
    std::map<int, std::string> nm;
    imgsz = 0;
    std::string line;
    enum { NONE, NAMES, IMGSZ } sec = NONE;
    while (std::getline(f, line)) {
        const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t' || line[0] == '-');
        if (!indented) {                                   // top-level key -> switch/close section
            if (line.rfind("names:", 0) == 0) {
                sec = NAMES;
                const auto rest = trim(line.substr(6));
                if (!rest.empty() && rest.front() == '[' && rest.back() == ']') {
                    std::string values = rest.substr(1, rest.size() - 2);
                    std::stringstream ss(values);
                    std::string value; int index = 0;
                    while (std::getline(ss, value, ',')) nm[index++] = yaml_scalar(value);
                    sec = NONE;
                }
                continue;
            }
            if (line.rfind("imgsz:", 0) == 0) {
                sec = IMGSZ;
                auto p = line.find('[');                   // inline "imgsz: [640, 640]"
                if (p != std::string::npos) imgsz = std::atoi(line.c_str() + p + 1);
                else {
                    auto colon = line.find(':');
                    if (colon != std::string::npos) imgsz = std::atoi(line.c_str() + colon + 1);
                }
                continue;
            }
            sec = NONE; continue;
        }
        if (sec == NAMES) {                                // "  0: pedestrian"
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                const int idx = std::atoi(trim(line.substr(0, colon)).c_str());
                nm[idx] = yaml_scalar(line.substr(colon + 1));
            }
        } else if (sec == IMGSZ && imgsz == 0) {           // "- 640"
            auto d = line.find_first_of("0123456789");
            if (d != std::string::npos) imgsz = std::atoi(line.c_str() + d);
        }
    }
    names.clear();
    for (auto& kv : nm) names.push_back(kv.second);
    // A model can legitimately omit names while still carrying imgsz.  The
    // caller can use either piece of metadata independently.
    return !names.empty() || imgsz > 0;
}

} // namespace meta

// ---------------- source ----------------
static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
// The portable runner uses stb_image. Keep the resolver limited to formats
// that stb_image decodes in this bundle; advertising TIFF/WebP here would
// collect those files and then fail later in stbi_load().
static const std::set<std::string> kImageExt = {".jpg", ".jpeg", ".png", ".bmp"};
static const std::set<std::string> kVideoExt = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

SourceKind classify_source(const std::string& src) {
    std::error_code ec;
    if (fs::is_directory(path_from_utf8(src), ec)) return SourceKind::Dir;
    const std::string ext = lower(path_from_utf8(src).extension().string());
    if (ext == ".yaml" || ext == ".yml") return SourceKind::Dataset;
    if (kVideoExt.count(ext)) return SourceKind::Video;
    if (kImageExt.count(ext)) return SourceKind::Image;
    return SourceKind::Unknown;
}

static void collect_dir(const std::string& dir, std::vector<std::string>& out) {
    std::error_code ec;
    for (auto& e : fs::recursive_directory_iterator(path_from_utf8(dir), ec)) {
        if (!e.is_regular_file()) continue;
        if (kImageExt.count(lower(e.path().extension().string())))
            out.push_back(path_to_utf8(e.path()));
    }
    std::sort(out.begin(), out.end());
}

// best-effort dataset.yaml -> val image dir
static std::vector<std::string> resolve_dataset(const std::string& yaml) {
    std::ifstream f(path_from_utf8(yaml));
    std::string path, val, line;
    while (std::getline(f, line)) {
        auto kv = [&](const char* k, std::string& dst) {
            if (line.rfind(k, 0) == 0) {
                std::string v = line.substr(std::strlen(k));
                auto h = v.find('#'); if (h != std::string::npos) v = v.substr(0, h);
                size_t a = v.find_first_not_of(" \t"); size_t b = v.find_last_not_of(" \t\r");
                dst = (a == std::string::npos) ? "" : meta::yaml_scalar(v.substr(a, b - a + 1));
            }
        };
        kv("path:", path);
        kv("val:", val);
    }
    if (val.empty()) return {};
    const fs::path ydir = fs::absolute(path_from_utf8(yaml)).parent_path();
    fs::path root = path_from_utf8(path);
    if (root.empty()) root = ydir;
    else if (root.is_relative()) root = ydir / root;
    std::vector<fs::path> cands = {
        path_from_utf8(val),                 // absolute val
        root / path_from_utf8(val),          // dataset root + val
        ydir / path_from_utf8(val),          // yaml_dir/val
        path_from_utf8("/data/datasets") / path_from_utf8(path) / path_from_utf8(val),
    };
    std::error_code ec;
    for (auto& c : cands) {
        if (fs::is_directory(c, ec)) { std::vector<std::string> v; collect_dir(path_to_utf8(c), v); if (!v.empty()) return v; }
        if (fs::is_regular_file(c, ec) && lower(c.extension().string()) == ".txt") {
            std::vector<std::string> v; std::ifstream tf(c); std::string l;
            while (std::getline(tf, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                l = meta::yaml_scalar(l);
                fs::path listed = path_from_utf8(l);
                if (listed.is_relative()) listed = c.parent_path() / listed;
                if (!l.empty()) v.push_back(path_to_utf8(listed));
            }
            if (!v.empty()) return v;
        }
    }
    return {};
}

std::vector<std::string> gather_images(const std::string& src, int limit) {
    std::vector<std::string> out;
    switch (classify_source(src)) {
        case SourceKind::Image:   out = {src}; break;
        case SourceKind::Dir:     collect_dir(src, out); break;
        case SourceKind::Dataset: out = resolve_dataset(src); break;
        default: break;
    }
    if (limit > 0 && static_cast<int>(out.size()) > limit) out.resize(limit);
    return out;
}

} // namespace yolomaster
