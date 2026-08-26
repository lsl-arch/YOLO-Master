#!/usr/bin/env python3
"""Export YOLO-Master checkpoints for edge backend validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edge_utils import add_profile_args

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="YOLO-Master checkpoint path")
    parser.add_argument("--formats", nargs="+", default=["onnx", "ncnn"], choices=("onnx", "ncnn", "mnn"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    simplify = parser.add_mutually_exclusive_group()
    simplify.add_argument("--simplify", dest="simplify", action="store_true", help="Simplify ONNX with onnxsim (default)")
    simplify.add_argument("--no-simplify", dest="simplify", action="store_false", help="debug-only raw ONNX export")
    parser.set_defaults(simplify=True)
    add_profile_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    for fmt in args.formats:
        export_args = {
            "format": fmt,
            "imgsz": args.imgsz,
            "half": args.half,
            "int8": args.int8,
        }
        if fmt == "onnx":
            export_args.update({"opset": args.opset, "simplify": args.simplify})
        print(f"[export] {args.model} -> {fmt} with {export_args}")
        model.export(**export_args)
    print(
        f"[profile] {args.profile.name}: conf={args.conf if args.conf is not None else args.profile.conf_threshold}, "
        f"iou={args.iou if args.iou is not None else args.profile.iou_threshold}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
