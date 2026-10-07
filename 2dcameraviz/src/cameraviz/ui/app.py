"""The native six-camera viewer. Dataset reads run off the Tk event thread."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import colorchooser, filedialog, font, ttk

from ..dataloader import available_datasets, create_dataset
from ..dataloader.base import DatasetAdapter, DatasetError
from ..dataloader.models import CameraInfo, FrameData, FrameInfo, SceneInfo
from ..processing.colors import class_color
from ..processing.projection import ProjectedBox, project_boxes
from ..visualization.camera_panel import CameraPanel


@dataclass(frozen=True)
class _LoadedFrame:
    scene_index: int
    frames: tuple[FrameInfo, ...]
    frame_index: int
    frame: FrameData
    projected: dict[str, tuple[ProjectedBox, ...]]


@dataclass(frozen=True)
class _LoadedDataset:
    adapter: DatasetAdapter
    scenes: tuple[SceneInfo, ...]
    loaded: _LoadedFrame


def _read_frame(adapter: DatasetAdapter, scene_index: int, frames: tuple[FrameInfo, ...], index: int) -> _LoadedFrame:
    if not frames:
        raise DatasetError("This scene has no camera frames.")
    if not 0 <= index < len(frames):
        raise DatasetError(f"Frame index {index} is outside this scene's range 0–{len(frames) - 1}.")
    frame = adapter.load_frame(frames[index].token)
    return _LoadedFrame(scene_index, frames, index, frame, project_boxes(frame))


def _read_dataset(dataset: str, root: str, version: str, scene: str | None, index: int) -> _LoadedDataset:
    adapter = create_dataset(dataset, root=Path(root).expanduser() if root.strip() else None, version=version)
    scenes = adapter.list_scenes()
    if not scenes:
        raise DatasetError("This dataset has no scenes.")
    scene_index = next((i for i, item in enumerate(scenes) if scene in (item.name, item.token)), None)
    if scene is not None and scene_index is None:
        raise DatasetError(f"Unknown scene {scene!r}. Choose a scene name or token from this release.")
    if scene_index is None:
        scene_index = next((i for i, item in enumerate(scenes) if item.frame_count), 0)
    return _LoadedDataset(adapter, scenes, _read_frame(adapter, scene_index, adapter.list_frames(scenes[scene_index].token), index))


class CameraViewer:
    def __init__(
        self, root: tk.Tk, *, dataset: str = "nuscenes", root_path: str | Path | None = None,
        version: str = "v1.0-mini", scene: str | None = None, frame: int = 0, autoload: bool = False,
    ):
        self.root = root
        self.adapter: DatasetAdapter | None = None
        self.scenes: tuple[SceneInfo, ...] = ()
        self.frames: tuple[FrameInfo, ...] = ()
        self.scene_index = 0
        self.frame_index = 0
        self.frame_data: FrameData | None = None
        self.panels: dict[str, CameraPanel] = {}
        self.class_vars: dict[str, tk.BooleanVar] = {}
        self.class_colors: dict[str, str] = {}
        self.class_labels: dict[str, ttk.Label] = {}
        self.color_buttons: dict[str, tk.Button] = {}
        self.busy = False
        self.last_error: str | None = None
        self._closed = False
        self._updating = False
        self._request_id = 0
        self._results: Queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-loader")
        self._poll_job: str | None = None
        self._slider_job: str | None = None
        self.dataset_var = tk.StringVar(root, dataset)
        self.root_var = tk.StringVar(root, "" if root_path is None else str(root_path))
        self.version_var = tk.StringVar(root, version)
        self.frame_var = tk.DoubleVar(root, 0)
        self.width_var = tk.DoubleVar(root, 2)
        self.mode_var = tk.StringVar(root, "Projected 3D")
        self._configure_style()
        self._build_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Left>", lambda event: self._key_step(event, -1))
        self.root.bind("<Right>", lambda event: self._key_step(event, 1))
        self._poll_job = self.root.after(30, self._poll_results)
        if autoload:
            self.root.after_idle(lambda: self.load_dataset(scene=scene, frame=frame))

    def _configure_style(self) -> None:
        self.root.title("2dcameraviz · Dataset viewer")
        self.root.geometry("1500x900")
        self.root.minsize(1000, 600)
        self.root.configure(background="#111820")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        font_family = font.nametofont("TkDefaultFont").actual("family")
        style.configure("TFrame", background="#272b30")
        style.configure("TLabel", background="#272b30", foreground="#e9edf2", font=(font_family, 10))
        style.configure("Heading.TLabel", font=(font_family, 11, "bold"))
        style.configure("Title.TLabel", font=(font_family, 17, "bold"))
        style.configure("TCheckbutton", background="#272b30", foreground="#e9edf2")
        style.map("TCheckbutton", background=[("active", "#333943")])
        style.configure("TButton", padding=(8, 5))
        style.configure("Camera.TFrame", background="#202a36")
        style.configure("Camera.TLabel", background="#202a36", foreground="#e9edf2", font=(font_family, 10, "bold"))
        style.configure("Viewport.TFrame", background="#111820")

    def _build_window(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        left = ttk.Frame(self.root, width=320)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.menu_canvas = tk.Canvas(left, background="#272b30", highlightthickness=0, width=300)
        self.menu_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.menu_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.menu_canvas.configure(yscrollcommand=scrollbar.set)
        menu = ttk.Frame(self.menu_canvas, padding=16)
        menu.columnconfigure(0, weight=1)
        menu_window = self.menu_canvas.create_window((0, 0), window=menu, anchor="nw")
        menu.bind("<Configure>", lambda _event: self.menu_canvas.configure(scrollregion=self.menu_canvas.bbox("all")))
        self.menu_canvas.bind("<Configure>", lambda event: self.menu_canvas.itemconfigure(menu_window, width=event.width))
        self.root.bind_all("<MouseWheel>", self._scroll_menu)
        self.root.bind_all("<Button-4>", self._scroll_menu)
        self.root.bind_all("<Button-5>", self._scroll_menu)
        ttk.Label(menu, text="2dcameraviz", style="Title.TLabel").pack(anchor="w")
        ttk.Label(menu, text="Autonomous driving datasets", padding=(0, 3, 0, 14)).pack(anchor="w")
        self._heading(menu, "Dataset")
        self.dataset_combo = ttk.Combobox(menu, textvariable=self.dataset_var, values=available_datasets(), state="readonly")
        self.dataset_combo.pack(fill="x", pady=(4, 7))
        ttk.Label(menu, text="Dataset root").pack(anchor="w")
        self.root_entry = ttk.Entry(menu, textvariable=self.root_var)
        self.root_entry.pack(fill="x", pady=(4, 5))
        self.browse_button = ttk.Button(menu, text="Browse…", command=self._browse)
        self.browse_button.pack(anchor="w", pady=(0, 7))
        ttk.Label(menu, text="Version").pack(anchor="w")
        self.version_entry = ttk.Entry(menu, textvariable=self.version_var)
        self.version_entry.pack(fill="x", pady=(4, 7))
        self.load_button = ttk.Button(menu, text="Load dataset", command=self.load_dataset)
        self.load_button.pack(fill="x")
        self._heading(menu, "Scene and frame")
        self.scene_combo = ttk.Combobox(menu, state="disabled")
        self.scene_combo.pack(fill="x", pady=(5, 7))
        self.scene_combo.bind("<<ComboboxSelected>>", self._select_scene)
        self.scene_description = ttk.Label(menu, text="Load a dataset to choose a scene.", wraplength=260)
        self.scene_description.pack(fill="x", pady=(0, 8))
        self.frame_scale = ttk.Scale(menu, from_=0, to=1, variable=self.frame_var, command=self._slider_changed)
        self.frame_scale.pack(fill="x")
        self.frame_label = ttk.Label(menu, text="Frame —")
        self.frame_label.pack(anchor="w", pady=(5, 7))
        navigation = ttk.Frame(menu)
        navigation.pack(fill="x")
        self.previous_button = ttk.Button(navigation, text="Previous", command=lambda: self.step_frame(-1))
        self.previous_button.pack(side="left", padx=(0, 6))
        self.next_button = ttk.Button(navigation, text="Next", command=lambda: self.step_frame(1))
        self.next_button.pack(side="left")
        self._heading(menu, "Bounding boxes")
        ttk.Label(menu, text="Box display").pack(anchor="w", pady=(5, 3))
        self.mode_combo = ttk.Combobox(menu, textvariable=self.mode_var, values=("Projected 3D", "2D rectangles"), state="readonly")
        self.mode_combo.pack(fill="x", pady=(0, 8))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_style())
        width_row = ttk.Frame(menu)
        width_row.pack(fill="x")
        ttk.Label(width_row, text="Line width (pixels)").pack(side="left")
        self.width_label = ttk.Label(width_row, text="2.0")
        self.width_label.pack(side="right")
        ttk.Scale(menu, from_=1, to=8, variable=self.width_var, command=self._width_changed).pack(fill="x", pady=(3, 8))
        buttons = ttk.Frame(menu)
        buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(buttons, text="Show all", command=lambda: self.show_classes(True)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Hide all", command=lambda: self.show_classes(False)).pack(side="left")
        self.class_frame = ttk.Frame(menu)
        self.class_frame.pack(fill="x")
        self.class_frame.columnconfigure(0, weight=1)
        ttk.Label(self.class_frame, text="No annotations loaded.").grid(sticky="w")
        self._heading(menu, "Status")
        self.status_label = ttk.Label(menu, text="Ready", wraplength=260)
        self.status_label.pack(fill="x", pady=(5, 0))
        self.grid_frame = ttk.Frame(self.root, style="Viewport.TFrame", padding=8)
        self.grid_frame.grid(row=0, column=1, sticky="nsew")
        self._set_layout((
            CameraInfo("CAM_FRONT_LEFT", "Front left", 0, 0), CameraInfo("CAM_FRONT", "Front", 0, 1),
            CameraInfo("CAM_FRONT_RIGHT", "Front right", 0, 2), CameraInfo("CAM_BACK_LEFT", "Rear left", 1, 0),
            CameraInfo("CAM_BACK", "Rear", 1, 1), CameraInfo("CAM_BACK_RIGHT", "Rear right", 1, 2),
        ))
        self._sync_navigation()

    @staticmethod
    def _heading(parent: tk.Misc, text: str) -> None:
        ttk.Separator(parent).pack(fill="x", pady=(15, 10))
        ttk.Label(parent, text=text, style="Heading.TLabel").pack(anchor="w")

    def _scroll_menu(self, event: tk.Event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self.menu_canvas:
            widget = getattr(widget, "master", None)
        if widget is self.menu_canvas:
            direction = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
            self.menu_canvas.yview_scroll(direction * 3, "units")

    def _browse(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, initialdir=self.root_var.get() or str(Path.cwd()))
        if selected:
            self.root_var.set(selected)

    def _set_layout(self, layout: tuple[CameraInfo, ...]) -> None:
        if not layout:
            return
        if len({item.channel for item in layout}) != len(layout) or len({(item.row, item.column) for item in layout}) != len(layout):
            raise DatasetError("Camera layout must have distinct channels and grid positions.")
        if tuple(panel.info for panel in self.panels.values()) == layout:
            return
        for panel in self.panels.values():
            panel.destroy()
        for i in range(self.grid_frame.grid_size()[0]):
            self.grid_frame.columnconfigure(i, weight=0, minsize=0, uniform="")
        for i in range(self.grid_frame.grid_size()[1]):
            self.grid_frame.rowconfigure(i, weight=0, minsize=0, uniform="")
        self.panels.clear()
        for info in layout:
            panel = CameraPanel(self.grid_frame, info)
            panel.grid(row=info.row, column=info.column, sticky="nsew", padx=4, pady=4)
            self.grid_frame.columnconfigure(info.column, weight=1, uniform="camera")
            self.grid_frame.rowconfigure(info.row, weight=1, uniform="camera")
            self.panels[info.channel] = panel

    def _submit(self, kind: str, work) -> None:
        if self._slider_job is not None:
            self.root.after_cancel(self._slider_job)
            self._slider_job = None
        self._request_id += 1
        request_id = self._request_id
        self.busy = True
        self.last_error = None
        self.status_label.configure(text="Loading…", foreground="#e9edf2")
        self._sync_navigation()
        for control in (self.load_button, self.browse_button, self.dataset_combo, self.root_entry, self.version_entry):
            control.configure(state="disabled")
        results = self._results
        def execute():
            try:
                results.put((request_id, kind, work(), None))
            except Exception as error:
                results.put((request_id, kind, None, error))
        self._executor.submit(execute)

    def load_dataset(self, *, scene: str | None = None, frame: int = 0) -> None:
        if self.busy or self._closed:
            return
        dataset, root, version = self.dataset_var.get(), self.root_var.get(), self.version_var.get()
        self._submit("dataset", lambda: _read_dataset(dataset, root, version, scene, frame))

    def _poll_results(self) -> None:
        self._poll_job = None
        if self._closed:
            return
        try:
            while True:
                request_id, kind, result, error = self._results.get_nowait()
                if request_id != self._request_id:
                    continue
                self.busy = False
                if error is None:
                    try:
                        if kind == "dataset":
                            layout = tuple(result.adapter.camera_layout) or tuple(camera.info for camera in result.loaded.frame.cameras)
                            self._set_layout(layout)
                            self.adapter, self.scenes = result.adapter, result.scenes
                            self.scene_combo.configure(values=[scene.name for scene in self.scenes])
                            self._set_classes(self.adapter.class_names, result.loaded.frame)
                            self._accept_frame(result.loaded)
                        else:
                            self._accept_frame(result)
                    except Exception as display_error:
                        error = display_error
                if error is not None:
                    self.last_error = str(error)
                    self.status_label.configure(text=f"Unable to load: {error}", foreground="#ff9b9b")
                for control in (self.load_button, self.browse_button, self.root_entry, self.version_entry):
                    control.configure(state="normal")
                self.dataset_combo.configure(state="readonly")
                self._sync_navigation()
        except Empty:
            pass
        self._poll_job = self.root.after(30, self._poll_results)

    def _set_classes(self, names: tuple[str, ...], frame: FrameData) -> None:
        names = tuple(dict.fromkeys((*names, *(box.class_name for box in frame.boxes))))
        previous = {name: variable.get() for name, variable in self.class_vars.items()}
        for child in self.class_frame.winfo_children():
            child.destroy()
        self.class_vars.clear()
        self.class_labels.clear()
        self.color_buttons.clear()
        if not names:
            ttk.Label(self.class_frame, text="No annotations available.").grid(sticky="w")
        for row, name in enumerate(names):
            variable = tk.BooleanVar(self.root, previous.get(name, True))
            self.class_vars[name] = variable
            self.class_colors.setdefault(name, class_color(name))
            ttk.Checkbutton(self.class_frame, variable=variable, command=self._apply_style).grid(row=row, column=0, sticky="w")
            label = ttk.Label(self.class_frame, text=name, wraplength=180)
            label.grid(row=row, column=1, sticky="w", padx=(0, 5), pady=3)
            self.class_labels[name] = label
            button = tk.Button(self.class_frame, text="Color", background=self.class_colors[name], foreground="white", activeforeground="white", relief="flat", width=5, command=lambda item=name: self.choose_color(item))
            button.grid(row=row, column=2, sticky="e", pady=3)
            self.color_buttons[name] = button
        self.class_frame.columnconfigure(0, weight=0)
        self.class_frame.columnconfigure(1, weight=1)

    def _accept_frame(self, loaded: _LoadedFrame) -> None:
        if any(camera.info.channel not in self.panels for camera in loaded.frame.cameras):
            self._set_layout(tuple(panel.info for panel in self.panels.values()) + tuple(camera.info for camera in loaded.frame.cameras if camera.info.channel not in self.panels))
        self.scene_index, self.frames, self.frame_index, self.frame_data = loaded.scene_index, loaded.frames, loaded.frame_index, loaded.frame
        names = tuple(dict.fromkeys((*(self.class_vars), *(box.class_name for box in loaded.frame.boxes))))
        if any(name not in self.class_vars for name in names):
            self._set_classes(names, loaded.frame)
        counts = Counter(box.class_name for box in loaded.frame.boxes)
        for name, label in self.class_labels.items():
            label.configure(text=f"{name}  ({counts[name]})")
        cameras = {camera.info.channel: camera for camera in loaded.frame.cameras}
        self._apply_style(redraw=False)
        for channel, panel in self.panels.items():
            panel.set_frame(cameras.get(channel), loaded.projected.get(channel, ()))
        self.status_label.configure(text=f"{len(cameras)} cameras · {len(loaded.frame.boxes)} annotated objects\nLeft / right arrows step through frames.", foreground="#b5c7d9")

    def _sync_navigation(self) -> None:
        self._updating = True
        try:
            self.scene_combo.configure(state="readonly" if self.scenes and not self.busy else "disabled")
            if self.scenes:
                self.scene_combo.current(self.scene_index)
                self.scene_description.configure(text=self.scenes[self.scene_index].description)
            self.frame_scale.configure(to=max(1, len(self.frames) - 1), state="normal" if len(self.frames) > 1 and not self.busy else "disabled")
            self.frame_var.set(self.frame_index)
            self.previous_button.configure(state="normal" if self.frames and self.frame_index > 0 and not self.busy else "disabled")
            self.next_button.configure(state="normal" if self.frames and self.frame_index < len(self.frames) - 1 and not self.busy else "disabled")
            if self.frame_data is not None:
                elapsed = (self.frame_data.info.timestamp_us - self.frames[0].timestamp_us) / 1e6
                self.frame_label.configure(text=f"Frame {self.frame_index + 1} / {len(self.frames)} · {elapsed:.2f} s")
        finally:
            self._updating = False

    def _select_scene(self, _event: tk.Event | None = None) -> None:
        if self.busy or self.adapter is None:
            return
        index = self.scene_combo.current()
        if index < 0 or index == self.scene_index:
            return
        adapter = self.adapter
        scene = self.scenes[index]
        self._submit("frame", lambda: _read_frame(adapter, index, adapter.list_frames(scene.token), 0))

    def _slider_changed(self, value: str) -> None:
        if self._updating or self.busy or not self.frames:
            return
        if self._slider_job is not None:
            self.root.after_cancel(self._slider_job)
        index = min(len(self.frames) - 1, max(0, round(float(value))))
        self._slider_job = self.root.after(140, lambda: self._load_index(index))

    def _load_index(self, index: int) -> None:
        if self._slider_job is not None:
            self.root.after_cancel(self._slider_job)
        self._slider_job = None
        if self.busy or self.adapter is None or index == self.frame_index:
            return
        if self._closed:
            return
        adapter, frames, scene_index = self.adapter, self.frames, self.scene_index
        self._submit("frame", lambda: _read_frame(adapter, scene_index, frames, index))

    def step_frame(self, delta: int) -> None:
        if not self.frames or self.busy:
            return
        self._load_index(min(len(self.frames) - 1, max(0, self.frame_index + delta)))

    def _key_step(self, event: tk.Event, delta: int) -> str | None:
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, tk.Entry, tk.Text)):
            return None
        self.step_frame(delta)
        return "break"

    def _width_changed(self, value: str) -> None:
        self.width_label.configure(text=f"{float(value):.1f}")
        self._apply_style()

    def _apply_style(self, *, redraw: bool = True) -> None:
        visible = {name for name, variable in self.class_vars.items() if variable.get()}
        for panel in self.panels.values():
            if redraw:
                panel.set_style(visible, self.class_colors, self.width_var.get(), self.mode_var.get())
            else:
                panel.visible_classes, panel.class_colors = visible, self.class_colors
                panel.line_width, panel.box_mode = self.width_var.get(), self.mode_var.get()

    def show_classes(self, show: bool) -> None:
        for variable in self.class_vars.values():
            variable.set(show)
        self._apply_style()

    def choose_color(self, name: str) -> None:
        _rgb, selected = colorchooser.askcolor(self.class_colors[name], parent=self.root, title=f"{name} bounding boxes")
        if selected:
            self.class_colors[name] = selected
            self.color_buttons[name].configure(background=selected)
            self._apply_style()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._request_id += 1
        for job in (self._poll_job, self._slider_job):
            if job is not None:
                self.root.after_cancel(job)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def run_app(
    *, dataset: str = "nuscenes", root: str | Path | None = None, version: str = "v1.0-mini",
    scene: str | None = None, frame: int = 0, autoload: bool = False,
) -> int:
    window = tk.Tk()
    viewer = CameraViewer(window, dataset=dataset, root_path=root, version=version, scene=scene, frame=frame, autoload=autoload)
    try:
        window.mainloop()
    finally:
        if not viewer._closed:
            viewer.close()
    return 0
