"""A camera image fitted to its panel, with overlays in image coordinates."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ..dataloader.models import CameraFrame, CameraInfo
from ..processing.projection import ProjectedBox


class CameraPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, info: CameraInfo):
        super().__init__(master, style="Camera.TFrame")
        self.info = info
        self.camera: CameraFrame | None = None
        self.boxes: tuple[ProjectedBox, ...] = ()
        self.visible_classes: set[str] = set()
        self.class_colors: dict[str, str] = {}
        self.line_width = 2.0
        self.box_mode = "Projected 3D"
        self._image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._photo_size: tuple[int, int] | None = None
        self._resize_job: str | None = None
        self.header = ttk.Label(self, text=info.label, style="Camera.TLabel", padding=(10, 8))
        self.header.pack(fill="x")
        self.canvas = tk.Canvas(self, background="#111820", highlightthickness=0, width=320, height=180)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Destroy>", self._on_destroy)

    def set_frame(self, camera: CameraFrame | None, boxes: tuple[ProjectedBox, ...] = ()) -> None:
        self.camera = camera
        self.boxes = boxes
        self._image = None if camera is None else Image.fromarray(camera.image)
        self._photo = None
        self._photo_size = None
        self.redraw()

    def set_style(self, visible_classes: set[str], class_colors: dict[str, str], line_width: float, box_mode: str) -> None:
        self.visible_classes = visible_classes
        self.class_colors = class_colors
        self.line_width = line_width
        self.box_mode = box_mode
        self.redraw()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(40, self._finish_resize)

    def _finish_resize(self) -> None:
        self._resize_job = None
        self.redraw()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self.canvas and self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None

    def redraw(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self._image is None:
            self.header.configure(text=self.info.label)
            self.canvas.create_text(width / 2, height / 2, text="Camera unavailable", fill="#9ba9b8", tags="placeholder")
            return
        if width < 2 or height < 2:
            return
        scale = min(width / self._image.width, height / self._image.height)
        target = (max(1, round(self._image.width * scale)), max(1, round(self._image.height * scale)))
        offset_x, offset_y = (width - target[0]) / 2, (height - target[1]) / 2
        if self._photo is None or self._photo_size != target:
            resized = self._image.resize(target, Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(resized, master=self.canvas)
            self._photo_size = target
        self.canvas.create_image(offset_x, offset_y, image=self._photo, anchor="nw", tags="camera_image")
        # Rounding image dimensions can give slightly different axis scales.
        scale_x, scale_y = target[0] / self._image.width, target[1] / self._image.height
        selected = [box for box in self.boxes if box.class_name in self.visible_classes]
        count = len(selected)
        self.header.configure(text=f"{self.info.label}   ·   {count} {'box' if count == 1 else 'boxes'}")
        for box in selected:
            color = self.class_colors.get(box.class_name, "#00aaff")
            if self.box_mode == "2D rectangles":
                left, top, right, bottom = box.rectangle
                self.canvas.create_rectangle(
                    offset_x + left * scale_x, offset_y + top * scale_y,
                    offset_x + right * scale_x, offset_y + bottom * scale_y,
                    outline=color, width=self.line_width, tags="bounding_box",
                )
            else:
                for segment in box.segments:
                    (x1, y1), (x2, y2) = segment
                    self.canvas.create_line(
                        offset_x + x1 * scale_x, offset_y + y1 * scale_y,
                        offset_x + x2 * scale_x, offset_y + y2 * scale_y,
                        fill=color, width=self.line_width, tags="bounding_box",
                    )
