"""Dependency-light contracts for the full Issue #51 edge runtime."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "examples" / "YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_selection_enforces_issue_gate(tmp_path):
    quant = load_module("issue51_quant", EDGE / "scripts" / "quantize_int8.py")
    with pytest.raises(ValueError, match="300"):
        quant.select_calibration_images(tmp_path, 299)
    for index in range(300):
        (tmp_path / f"{index:04d}.jpg").touch()
    selected = quant.select_calibration_images(tmp_path, 300)
    assert len(selected) == 300
    assert selected[0].name == "0000.jpg"


def test_export_layout_normalizers_accept_common_shapes():
    parity = load_module("issue51_parity", EDGE / "scripts" / "mnn_parity.py")
    values = np.arange(14 * 5, dtype=np.float32).reshape(14, 5)
    assert parity.normalize_output(values, 14).shape == (14, 5)
    assert parity.normalize_output(values.T[None], 14).shape == (14, 5)
    assert np.array_equal(parity.normalize_output(values.T, 14), values)
    assert parity.select_detection_output([np.zeros((1, 2, 2)), values[None]], 14).shape == (1, 14, 5)

    mnn_val = load_module("issue51_mnn_val", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.normalize_output(values[None, :, :], 14).shape == (14, 5)


def test_nms_is_class_offset_friendly_and_handles_empty():
    mnn_val = load_module("issue51_mnn_val_nms", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [30, 30, 40, 40]], dtype=np.float32)
    keep = mnn_val.nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.5)
    assert keep == [0, 2]


def test_runtime_sources_contain_failure_guards():
    ort = (EDGE / "cpp" / "src" / "ort_backend.cpp").read_text(encoding="utf-8")
    ncnn = (EDGE / "cpp" / "src" / "ncnn_backend.cpp").read_text(encoding="utf-8")
    assert "no detection tensor" in ort
    assert "#ifdef USE_CUDA" in ort
    assert "legacy_feat = 5 + cfg.num_classes()" in ort
    assert "objectness * raw_major" in ort
    assert "ex.input" in ncnn and "failed to set input blob" in ncnn
    assert "discover_io" in ncnn
    common = (EDGE / "cpp" / "src" / "common.cpp").read_text(encoding="utf-8")
    assert "!std::isfinite(v) || v <= threshold" in common
    assert "- 0.1" in common
    assert "max_nms=30000" in common and "nms_greedy(off, scores, 0.f, cfg.iou_thresh, cfg.max_det" in common
    assert '".tif"' not in common and '".webp"' not in common
    for script_name in ("mnn_val.py", "mnn_parity.py", "quantize_int8.py", "eval_map.py"):
        script = (EDGE / "scripts" / script_name).read_text(encoding="utf-8")
        assert '".webp"' not in script
    ort_source = (EDGE / "cpp" / "src" / "ort_backend.cpp").read_text(encoding="utf-8")
    assert "MultiByteToWideChar" in ort_source
    cmake = (EDGE / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "opencv_*.dll" in cmake and "REMOVE_DUPLICATES OCV_DLLS" in cmake
    assert "target_link_options" in cmake and "-municode" in cmake
    assert "ALLOW_NO_BACKENDS" in cmake and "diagnostic build" in cmake
    main = (EDGE / "cpp" / "src" / "main.cpp").read_text(encoding="utf-8")
    stb = (EDGE / "cpp" / "src" / "stb_impl.cpp").read_text(encoding="utf-8")
    assert "wmain" in main and "STBI_WINDOWS_UTF8" in stb and "STBIW_WINDOWS_UTF8" in stb
    assert "--acceptance requires --min-images >= 500" in main
    assert "#summary,,,,,," in main and "p50_ms,p95_ms,p99_ms,fps" in main
    assert "acceptance requires unique image stems" in main


def test_submission_gates_are_explicit():
    export = (EDGE / "scripts" / "export_models.py").read_text(encoding="utf-8")
    quant = (EDGE / "scripts" / "quantize_int8.py").read_text(encoding="utf-8")
    evaluator = (EDGE / "scripts" / "eval_map.py").read_text(encoding="utf-8")
    mnn = (EDGE / "scripts" / "mnn_val.py").read_text(encoding="utf-8")
    packager = (EDGE / "scripts" / "package_linux.sh").read_text(encoding="utf-8")
    assert "onnxsim" in export and "load_param" in export and "load_model" in export
    assert "_ncnn_pair_fingerprint" in export and "stale export" in export
    assert "routers_dense" in export and "did not find an ES_MOE" in export
    assert "DEFAULT_EXCLUDE" in quant and "QOperator" in quant
    assert "missing_patterns" in quant and "mixed-precision acceptance recipe" in quant
    assert '"im_name"' in evaluator and "min-images" in evaluator
    assert "one label and prediction file per image" in evaluator
    assert "num_classes" in evaluator and "len(values) >= 8" in evaluator
    assert "max-abs-delta-pct" in evaluator and "apply_delta_gate" in evaluator
    assert "validate_acceptance_image_floor" in evaluator
    assert "getNumpyData" in mnn and "getData" in mnn
    assert "valid_geometry" in mnn and ".9g" in mnn and "scores.size > 30000" in mnn
    common = (EDGE / "cpp" / "src" / "common.cpp").read_text(encoding="utf-8")
    assert "v > threshold" in common and "classes > candidate_threshold" in mnn
    assert "v >= threshold" not in common
    assert "classes >= candidate_threshold" not in mnn
    assert 'grep -q "not found"' in packager
    assert "libonnxruntime_providers_*.so" in packager and "ORT_PROVIDER_ROOT" in packager
    assert "is_external_gpu_driver \"$source\" && return 0" in packager
    assert "libonnxruntime_providers_*.so*|libcuda.so*" in packager


def test_mnn_export_does_not_overclaim_runtime_acceptance():
    export = (EDGE / "scripts" / "export_models.py").read_text(encoding="utf-8")
    mnn_block = export[export.index('"format": "mnn"'):]
    assert '"checked_scope": "converter_output"' in mnn_block
    assert '"runtime_smoke_checked": False' in mnn_block
    assert '"acceptance_ready": False' in mnn_block
    assert '"parity_required": True' in mnn_block


def test_timing_summary_row_matches_the_11_column_header():
    import csv
    import io

    main = (EDGE / "cpp" / "src" / "main.cpp").read_text(encoding="utf-8")
    assert "tag,pre_ms,infer_ms,post_ms,total_ms,detections,mean_ms,p50_ms,p95_ms,p99_ms,fps" in main
    literal = "#summary,,,,,,1,2,3,4,5"
    row = next(csv.reader(io.StringIO(literal)))
    assert len(row) == 11
    assert row[0] == "#summary" and row[6:] == ["1", "2", "3", "4", "5"]


def test_map_delta_budget_is_optional_and_strict():
    evaluator = load_module("issue51_eval_map_gate", EDGE / "scripts" / "eval_map.py")

    # Existing smoke/reference-free calls have no gate and remain passing.
    assert evaluator.delta_gate_passes(25.0, None) is True
    unchanged = {"abs_delta_mAP50-95_pct": 25.0}
    assert evaluator.apply_delta_gate(unchanged, None) == 0
    assert "max_abs_delta_mAP50-95_pct" not in unchanged
    # The explicit budget is inclusive at the boundary and fails above it.
    assert evaluator.delta_gate_passes(0.5, 0.5) is True
    assert evaluator.delta_gate_passes(0.500001, 0.5) is False
    passing = {"abs_delta_mAP50-95_pct": 0.5}
    assert evaluator.apply_delta_gate(passing, 0.5) == 0
    assert passing["mAP50-95_delta_gate_passed"] is True
    failing = {"abs_delta_mAP50-95_pct": 0.500001}
    assert evaluator.apply_delta_gate(failing, 0.5) != 0
    assert failing["mAP50-95_delta_gate_passed"] is False

    assert evaluator.nonnegative_finite_float("0.5") == pytest.approx(0.5)
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("nan")
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("-0.1")


def test_map_delta_budget_requires_reference_json(monkeypatch):
    evaluator = load_module("issue51_eval_map_args", EDGE / "scripts" / "eval_map.py")
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_map.py",
            "--preds",
            "preds",
            "--max-abs-delta-pct",
            "0.5",
        ],
    )
    args = evaluator.parse_args()
    assert args.max_abs_delta_pct == pytest.approx(0.5)
    with pytest.raises(ValueError, match="reference-json"):
        evaluator.validate_delta_budget(args.max_abs_delta_pct, args.reference_json)


def test_map_acceptance_cannot_lower_the_500_image_floor(monkeypatch):
    evaluator = load_module("issue51_eval_min_images", EDGE / "scripts" / "eval_map.py")
    monkeypatch.setattr(
        "sys.argv",
        ["eval_map.py", "--preds", "preds", "--min-images", "1"],
    )
    args = evaluator.parse_args()
    assert args.smoke is False and args.min_images == 1
    with pytest.raises(ValueError, match="500"):
        evaluator.validate_acceptance_image_floor(args)


def test_map_delta_gate_cannot_be_claimed_on_smoke_run(monkeypatch):
    evaluator = load_module("issue51_eval_smoke_gate", EDGE / "scripts" / "eval_map.py")
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_map.py", "--preds", "preds", "--smoke",
            "--reference-json", "reference.json", "--max-abs-delta-pct", "0.5",
        ],
    )
    args = evaluator.parse_args()
    evaluator.validate_delta_budget(args.max_abs_delta_pct, args.reference_json)
    with pytest.raises(ValueError, match="smoke"):
        evaluator.validate_smoke_gate(args.smoke, args.max_abs_delta_pct)


def test_mnn_cli_rejects_nonfinite_thresholds_and_negative_limits(tmp_path, monkeypatch):
    """Bad CLI numerics must fail before importing optional MNN/ORT wheels."""
    mnn_val = load_module("issue51_mnn_val_args", EDGE / "scripts" / "mnn_val.py")
    model = tmp_path / "model.mnn"
    image = tmp_path / "image.jpg"
    model.touch()
    image.touch()

    monkeypatch.setattr(
        "sys.argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--conf", "nan"],
    )
    with pytest.raises(ValueError, match="finite"):
        mnn_val.main()

    monkeypatch.setattr(
        "sys.argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--limit", "-1"],
    )
    with pytest.raises(ValueError, match="non-negative"):
        mnn_val.main()

    mnn_parity = load_module("issue51_mnn_parity_args", EDGE / "scripts" / "mnn_parity.py")
    onnx = tmp_path / "model.onnx"
    onnx.touch()
    monkeypatch.setattr(
        "sys.argv",
        [
            "mnn_parity.py", "--mnn", str(model), "--onnx", str(onnx),
            "--images", str(image), "--tolerance", "nan",
        ],
    )
    with pytest.raises(ValueError, match="finite"):
        mnn_parity.main()
