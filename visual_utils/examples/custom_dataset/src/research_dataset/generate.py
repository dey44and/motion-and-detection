"""Generate tiny on-disk research data: python -m research_dataset.generate DIR."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate(root: Path) -> None:
    """Write two sparse-ID frames into a new directory; never overwrite data."""
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    records = []
    for index, frame_id in enumerate(("scan-15", "scan-39")):
        points = rng.uniform((5, -4, -2), (20, 4, 2), size=(1000, 3)).astype(np.float32)
        np.savez(root / f"{frame_id}.npz", points=points, intensity=rng.random(1000).astype(np.float32))
        image = np.full((240, 320, 3), (35, 45, 55), dtype=np.uint8)
        Image.fromarray(image).save(root / f"{frame_id}.png")
        records.append({
            "frame_id": frame_id, "points": f"{frame_id}.npz", "image": f"{frame_id}.png",
            "timestamp_us": index * 100_000,
            "boxes": [{"annotation_id": f"box-{index}", "center": [10., 0., 0.],
                       "size": [2., 2., 2.], "rotation": np.eye(3).tolist()}],
        })
    manifest = {"dataset_id": "research-example:v1", "scene_id": "sequence-1", "frames": records,
                "intrinsics": [[200., 0., 160.], [0., 200., 120.], [0., 0., 1.]]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new output directory")
    generate(parser.parse_args().directory)
