"""The dataset library remains usable without any viewer or GUI framework."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys


FORBIDDEN = {
    "threedviz", "cameraviz", "detviz", "open3d", "tkinter", "matplotlib",
    "torch", "tensorflow", "waymo_open_dataset",
}


def test_dataset_utilities_import_without_loading_viewer_or_gui_modules():
    code = """
import sys
import visual_utils.waymo
import visual_utils.nuscenes
import visual_utils.nuscenes.motion
import visual_utils.nuscenes.projection
import visual_utils.nuscenes.splits
forbidden = %r
loaded = {name.split('.')[0] for name in sys.modules}
assert not forbidden.intersection(loaded), forbidden.intersection(loaded)
""" % FORBIDDEN
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_shared_source_has_no_reverse_dependencies_on_applications():
    root = Path(importlib.util.find_spec("visual_utils").origin).parent
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)
