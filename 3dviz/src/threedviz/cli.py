"""Command-line entry point; graphical dependencies are loaded only on launch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from dataset_utils import DatasetError, load_dataset_config

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


def build_parser() -> argparse.ArgumentParser:
    """Build a parser without importing Open3D or starting the desktop app."""
    parser = argparse.ArgumentParser(
        prog="3dviz",
        description="Explore autonomous driving point clouds and 3D ground truth boxes.",
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
    parser.add_argument("--split", help="dataset split, such as val, train or train_1of16 (default: all)")
    parser.add_argument("--scene", help="initial scene name or token")
    parser.add_argument("--frame", type=_nonnegative_int, default=0, help="initial frame index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="open a synthetic example without dataset files")
    parser.add_argument("--app-version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the window, or display CLI help without loading the renderer."""
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
    try:
        from .ui.app import run_app

        run_app(
            dataset=dataset,
            root=args.root,
            version=args.version or "v1.0-mini",
            scene=args.scene,
            frame=args.frame,
            autoload=dataset == "demo" or args.root is not None or config is not None,
            **({"split": args.split} if args.split is not None else {}),
            **({"config": config} if config is not None else {}),
        )
    except ModuleNotFoundError as exc:
        if exc.name != "open3d":
            raise
        parser.exit(
            1,
            "3dviz: Open3D is required to open the viewer. "
            "Install the package with: pip install -e './3dviz[dev]'\n",
        )

    return 0
