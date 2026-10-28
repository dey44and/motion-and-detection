"""Launch detection review; help remains usable without a desktop."""

from __future__ import annotations

import argparse
from pathlib import Path

from threedviz.dataloader import available_datasets

from . import __version__
from .predictions import available_prediction_formats


def _nonnegative_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="3ddetviz", description="Review 3D detections alongside dataset ground truth.")
    parser.add_argument("--dataset", choices=available_datasets(), default="nuscenes")
    parser.add_argument("--dataroot", "--root", dest="root", type=Path)
    parser.add_argument("--version", help="dataset release (default: v1.0-mini for nuScenes)")
    parser.add_argument("--dataset-config", type=Path, help="shared DatasetConfig JSON; paths are relative to the file")
    parser.add_argument("--split", help="dataset-defined split (default: val for nuScenes/demo, all otherwise)")
    parser.add_argument("--predictions", type=Path, help="nuScenes results JSON or OpenPCDet result.pkl")
    parser.add_argument("--prediction-format", default="auto", choices=("auto", *available_prediction_formats()),
                        help="registered format; auto infers built-in formats from the suffix")
    parser.add_argument("--cache-dir", type=Path, help="directory for the per-frame prediction index")
    parser.add_argument("--scene", help="initial scene name or ID")
    parser.add_argument("--frame", type=_nonnegative_int, default=0, help="zero-based frame index")
    parser.add_argument("--demo", action="store_true", help="generated point clouds and detections without dataset files")
    parser.add_argument("--app-version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.demo or args.dataset == "demo") and args.predictions is not None:
        parser.error("--demo generates its own predictions; omit --predictions")
    if args.dataset_config is not None and (args.root is not None or args.version is not None):
        parser.error("--dataset-config cannot be combined with --root/--dataroot or --version")
    dataset = "demo" if args.demo else args.dataset
    if dataset == "demo" and args.dataset_config is not None:
        parser.error("--dataset-config requires a registered dataset, not demo")
    configuration = {}
    if args.dataset_config is not None:
        from visual_utils import load_dataset_config
        from visual_utils.errors import DatasetError
        try:
            configuration["config"] = load_dataset_config(args.dataset_config)
        except (DatasetError, OSError, ValueError, TypeError) as error:
            parser.error(str(error))
    if args.prediction_format != "auto":
        configuration["prediction_format"] = args.prediction_format
    try:
        from .ui.app import run_app
        run_app(dataset=dataset, root=args.root, version=args.version or "v1.0-mini",
                split=args.split or ("val" if dataset in {"nuscenes", "demo"} else "all"),
                predictions=args.predictions, cache_dir=args.cache_dir,
                scene=args.scene, frame=args.frame,
                autoload=dataset == "demo" or args.root is not None or args.dataset_config is not None,
                **configuration)
    except ModuleNotFoundError as error:
        if error.name != "open3d":
            raise
        parser.exit(1, "3ddetviz: Install the viewer dependencies with pip install -e './visual_utils' -e './viz_tools/3dviz' -e './viz_tools/3ddetviz'.\n")
    return 0
