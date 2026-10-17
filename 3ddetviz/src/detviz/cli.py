"""Launch detection review; help remains usable without a desktop."""

from __future__ import annotations

import argparse
from pathlib import Path

from threedviz.dataloader import available_datasets

from . import __version__


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
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", choices=("all", "val", "train", "mini_val", "mini_train", "test"), default="val")
    parser.add_argument("--predictions", type=Path, help="nuScenes results JSON or OpenPCDet result.pkl")
    parser.add_argument("--cache-dir", type=Path, help="directory for the per-frame prediction index")
    parser.add_argument("--scene", help="initial scene name or token")
    parser.add_argument("--frame", type=_nonnegative_int, default=0, help="zero-based frame index")
    parser.add_argument("--demo", action="store_true", help="generated point clouds and detections without dataset files")
    parser.add_argument("--app-version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.demo or args.dataset == "demo") and args.predictions is not None:
        parser.error("--demo generates its own predictions; omit --predictions")
    try:
        from .ui.app import run_app
        run_app(dataset="demo" if args.demo else args.dataset, root=args.root, version=args.version,
                split=args.split, predictions=args.predictions, cache_dir=args.cache_dir,
                scene=args.scene, frame=args.frame, autoload=args.demo or args.dataset == "demo" or args.root is not None)
    except ModuleNotFoundError as error:
        if error.name != "open3d":
            raise
        parser.exit(1, "3ddetviz: Install the viewer dependencies with pip install -e './3dviz' -e './3ddetviz'.\n")
    return 0
