"""Dataset-independent camera projection and annotation colors."""

from .colors import class_color
from .projection import ProjectedBox, project_box, project_boxes

__all__ = ["ProjectedBox", "class_color", "project_box", "project_boxes"]
