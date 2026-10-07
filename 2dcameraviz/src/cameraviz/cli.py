"""Command-line entry point, with Tk imported only when opening the window."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
    """Create the CLI without importing Tk or requiring a display."""

    parser = argparse.ArgumentParser(
        prog="2dcameraviz",
        description="Explore six autonomous driving camera views and projected ground truth boxes.",
    )
    parser.add_argument("--dataset", choices=available_datasets(), default="nuscenes")
    parser.add_argument(
        "--dataroot", "--root", dest="root", type=Path,
        help="dataset root containing release metadata and camera images",
    )
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes release (default: v1.0-mini)")
    parser.add_argument("--scene", help="initial scene name or token")
    parser.add_argument("--frame", type=_nonnegative_int, default=0, help="initial zero-based frame index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="open a synthetic example without dataset files")
    parser.add_argument("--app-version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the viewer with actionable errors for unavailable Tk/display."""

    parser = build_parser()
    args = parser.parse_args(argv)
    dataset = "demo" if args.demo else args.dataset
    try:
        from .ui.app import run_app

        return run_app(
            dataset=dataset,
            root=args.root,
            version=args.version,
            scene=args.scene,
            frame=args.frame,
            autoload=dataset == "demo" or args.root is not None,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        parser.exit(
            1,
            "2dcameraviz: Tkinter is required to open the viewer. "
            "On Debian/Ubuntu install it with: sudo apt install python3-tk. "
            "For other platforms, use a Python installation that includes Tk.\n",
        )
    except Exception as exc:
        message = str(exc).lower()
        is_display_error = (
            type(exc).__name__ == "TclError"
            and type(exc).__module__ == "_tkinter"
            and any(part in message for part in (
                "no display name", "couldn't connect to display",
                "could not connect to display", "cannot connect to display",
            ))
        )
        if not is_display_error:
            raise
        parser.exit(
            1,
            f"2dcameraviz: Cannot open the desktop display: {exc}. "
            "Run from a graphical desktop, or configure DISPLAY/X11 forwarding for a remote session.\n",
        )
