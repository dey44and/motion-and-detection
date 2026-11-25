"""Command-line entry point for the local browser viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from visual_utils import DatasetError, load_dataset_config

from . import __version__
from .dataloader import available_datasets


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a port between 1 and 65535") from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be a port between 1 and 65535")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build a parser without starting the local server."""
    parser = argparse.ArgumentParser(
        prog="3dviz",
        description="Review point clouds, 3D detections and LiDAR segmentation in a local browser.",
    )
    parser.add_argument("--dataset", choices=available_datasets(), default="nuscenes")
    parser.add_argument(
        "--dataroot", "--root", dest="root", type=Path,
        help="dataset root containing the release metadata and sensor files",
    )
    parser.add_argument("--version", help="dataset release (nuScenes default: v1.0-mini)")
    parser.add_argument(
        "--dataset-config", type=Path,
        help="DatasetConfig JSON; paths are relative to this file; excludes --root/--version",
    )
    parser.add_argument("--split", help="dataset split (defaults to val for nuScenes or demo with predictions, otherwise all)")
    parser.add_argument("--scene", help="initial scene name or token")
    parser.add_argument("--frame", type=_nonnegative_int, default=0, help="initial frame index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="open a synthetic example without dataset files")
    parser.add_argument("--predictions", type=Path, help="detection export to attach when opening the dataset")
    parser.add_argument("--prediction-format", default="auto", help="registered detection format (default: infer from filename)")
    parser.add_argument("--cache-dir", type=Path, help="directory for per-frame detection indexes")
    parser.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=_port, help="localhost port (default: 8080)")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction,
                        help="open the localhost URL in your browser (enabled by default)")
    parser.add_argument("--app-version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the viewer or display CLI help."""
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset = "demo" if args.demo else args.dataset
    config = None
    if args.dataset_config is not None:
        if args.root is not None or args.version is not None:
            parser.error("--dataset-config cannot be combined with --root/--dataroot or --version")
        if dataset == "demo":
            parser.error("--dataset-config requires a shared dataset provider, rather than demo")
        try:
            config = load_dataset_config(args.dataset_config)
        except DatasetError as error:
            parser.error(str(error))
    split = args.split
    if split is None and args.predictions is not None and dataset in {"nuscenes", "demo"}:
        split = "val"
    launch_options = dict(
        dataset=dataset,
        root=args.root,
        version=args.version or "v1.0-mini",
        scene=args.scene,
        frame=args.frame,
        autoload=dataset == "demo" or args.root is not None or config is not None,
        **({"split": split} if split is not None else {}),
        **({"config": config} if config is not None else {}),
    )
    try:
        from .detection.predictions import available_prediction_formats
        from .web.server import run_server

        formats = available_prediction_formats()
        if args.prediction_format not in ("auto", *formats):
            parser.error(f"Unknown prediction format {args.prediction_format!r}. Choose from auto, {', '.join(formats)}.")
        open_browser = not args.web if args.open_browser is None else args.open_browser
        run_server(
            port=args.port or 8080, open_browser=open_browser, predictions=args.predictions,
            prediction_format=args.prediction_format, cache_dir=args.cache_dir, **launch_options,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"fastapi", "uvicorn", "ijson"}:
            raise
        parser.exit(
            1,
            "3dviz: Install the viewer dependencies with "
            "pip install -e './visual_utils' -e './viz_tools/3dviz'.\n",
        )

    return 0
