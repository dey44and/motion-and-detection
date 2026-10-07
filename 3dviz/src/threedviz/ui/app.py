"""Sidebar controls and asynchronous dataset loading for the desktop viewer."""

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
import sys
import textwrap

from open3d.visualization import gui, rendering

from threedviz.dataloader import available_datasets, create_dataset
from threedviz.processing import COLOR_SCHEMES, class_color
from threedviz.visualization.renderer import SceneRenderer


class ViewerWindow:
    def __init__(
        self,
        *,
        dataset="nuscenes",
        root=None,
        version="v1.0-mini",
        scene=None,
        frame=0,
        autoload=False,
    ):
        self.application = gui.Application.instance
        self.window = self.application.create_window("3dviz · Dataset viewer", 1440, 900)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="3dviz-loader")
        self._future: Future | None = None
        self._request_id = 0
        self._alive = True
        self._busy = False
        self._adapter = None
        self._scenes = ()
        self._frames = ()
        self._scene_index = 0
        self._frame_index = 0
        self._class_checkboxes = {}
        self._class_count_labels = {}
        self._class_color_buttons = {}

        self.viewport = gui.SceneWidget()
        self.viewport.scene = rendering.Open3DScene(self.window.renderer)
        self.renderer = SceneRenderer(self.viewport, self.window.scaling)
        self.window.add_child(self.viewport)
        self._build_sidebar(dataset, root, version)
        self.window.add_child(self.sidebar)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self._set_busy(False)
        if autoload:
            self.load_dataset(initial_scene=scene, initial_frame=frame)

    def _build_sidebar(self, dataset, root, version):
        em = self.window.theme.font_size
        self.sidebar = gui.ScrollableVert(0.5 * em, gui.Margins(em, em, em, em))
        self.sidebar.add_child(gui.Label("3dviz"))
        self.sidebar.add_child(gui.Label("Autonomous driving datasets"))

        connection = gui.CollapsableVert("Dataset", 0.4 * em)
        self.dataset_choice = gui.Combobox()
        for name in available_datasets():
            self.dataset_choice.add_item(name)
        self.dataset_choice.selected_text = dataset
        self.dataset_choice.set_on_selection_changed(self._on_dataset_choice)
        connection.add_child(self.dataset_choice)
        connection.add_child(gui.Label("Dataset root"))
        row = gui.Horiz(0.25 * em)
        self.root_edit = gui.TextEdit()
        self.root_edit.text_value = str(root) if root is not None else ""
        self.root_edit.tooltip = "Folder containing samples/ and a nuScenes version folder"
        row.add_child(self.root_edit)
        self.browse_button = gui.Button("Browse")
        self.browse_button.set_on_clicked(self._browse)
        row.add_child(self.browse_button)
        connection.add_child(row)
        connection.add_child(gui.Label("Version"))
        self.version_edit = gui.TextEdit()
        self.version_edit.text_value = version
        connection.add_child(self.version_edit)
        self.load_button = gui.Button("Load dataset")
        self.load_button.set_on_clicked(self.load_dataset)
        connection.add_child(self.load_button)
        self.sidebar.add_child(connection)

        navigation = gui.CollapsableVert("Scene and frame", 0.4 * em)
        self.scene_choice = gui.Combobox()
        self.scene_choice.set_on_selection_changed(self._on_scene_choice)
        navigation.add_child(self.scene_choice)
        self.scene_description = gui.Label("Load a dataset to select a scene.")
        navigation.add_child(self.scene_description)
        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, 1)
        self.frame_slider.set_on_value_changed(self._on_frame_slider)
        navigation.add_child(self.frame_slider)
        self.frame_label = gui.Label("Frame —")
        navigation.add_child(self.frame_label)
        row = gui.Horiz(0.3 * em)
        self.previous_button = gui.Button("Previous")
        self.previous_button.set_on_clicked(lambda: self._request_frame(self._frame_index - 1))
        self.next_button = gui.Button("Next")
        self.next_button.set_on_clicked(lambda: self._request_frame(self._frame_index + 1))
        row.add_child(self.previous_button)
        row.add_child(self.next_button)
        navigation.add_child(row)
        self.sidebar.add_child(navigation)

        cloud = gui.CollapsableVert("Point cloud", 0.4 * em)
        cloud.add_child(gui.Label("Point size (pixels)"))
        self.point_size = gui.Slider(gui.Slider.DOUBLE)
        self.point_size.set_limits(1, 12)
        self.point_size.double_value = self.renderer.settings.point_size
        self.point_size.set_on_value_changed(self._on_point_size)
        cloud.add_child(self.point_size)
        self.point_color_label = gui.Label("Point colors")
        cloud.add_child(self.point_color_label)
        self.color_choice = gui.Combobox()
        for scheme in COLOR_SCHEMES:
            self.color_choice.add_item(scheme)
        self.color_choice.selected_text = self.renderer.settings.color_scheme
        self.color_choice.set_on_selection_changed(self._on_color_scheme)
        cloud.add_child(self.color_choice)
        self.distance_label = gui.Label("Color scale maximum (metres)")
        cloud.add_child(self.distance_label)
        self.distance_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.distance_edit.set_limits(1.0, 1000.0)
        self.distance_edit.double_value = self.renderer.settings.max_distance
        self.distance_edit.set_on_value_changed(self._on_distance)
        cloud.add_child(self.distance_edit)
        self.distance_hint = gui.Label("Near to far; values above the scale are clipped.")
        cloud.add_child(self.distance_hint)
        self.color_status = gui.Label("")
        cloud.add_child(self.color_status)
        self.raw_checkbox = gui.Checkbox("Undo ego motion compensation")
        self.raw_checkbox.tooltip = "Show points in their estimated acquisition-time sensor frames"
        self.raw_checkbox.set_on_checked(self._on_raw_points)
        cloud.add_child(self.raw_checkbox)
        self.raw_hint = gui.Label("Boxes stay at the sweep reference time.")
        self.raw_hint.visible = False
        cloud.add_child(self.raw_hint)
        self.sidebar.add_child(cloud)

        boxes = gui.CollapsableVert("3D bounding boxes", 0.4 * em)
        boxes.add_child(gui.Label("Line width (pixels)"))
        self.line_width = gui.Slider(gui.Slider.DOUBLE)
        self.line_width.set_limits(1, 10)
        self.line_width.double_value = self.renderer.settings.line_width
        self.line_width.set_on_value_changed(self._on_line_width)
        boxes.add_child(self.line_width)
        row = gui.Horiz(0.3 * em)
        select_all = gui.Button("Show all")
        select_all.set_on_clicked(lambda: self._select_classes(True))
        select_none = gui.Button("Hide all")
        select_none.set_on_clicked(lambda: self._select_classes(False))
        row.add_child(select_all)
        row.add_child(select_none)
        boxes.add_child(row)
        self.class_list = gui.WidgetProxy()
        self.class_list.set_widget(gui.Label("Classes appear after loading."))
        boxes.add_child(self.class_list)
        self.sidebar.add_child(boxes)

        self.axes_checkbox = gui.Checkbox("Show coordinate axes")
        self.axes_checkbox.checked = True
        self.axes_checkbox.set_on_checked(self.viewport.scene.show_axes)
        self.sidebar.add_child(self.axes_checkbox)
        reset_button = gui.Button("Reset camera")
        reset_button.set_on_clicked(self.renderer.reset_camera)
        self.sidebar.add_child(reset_button)
        self.sidebar.add_child(gui.Label("Drag: rotate · Scroll: zoom\nRight drag: pan"))
        self.status = gui.Label("Choose a dataset root or select demo.")
        self.sidebar.add_child(self.status)
        self._on_dataset_choice(dataset, 0)

    def _on_layout(self, context):
        rect = self.window.content_rect
        width = min(25 * self.window.theme.font_size, max(1, rect.width // 2))
        self.sidebar.frame = gui.Rect(rect.x, rect.y, width, rect.height)
        self.viewport.frame = gui.Rect(rect.x + width, rect.y, max(1, rect.width - width), rect.height)

    def _on_dataset_choice(self, name, index):
        has_root = name != "demo"
        self.root_edit.enabled = has_root
        self.browse_button.enabled = has_root
        self.version_edit.enabled = has_root

    def _browse(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select dataset root", self.window.theme)
        dialog.set_on_cancel(self.window.close_dialog)

        def selected(path):
            self.window.close_dialog()
            self.root_edit.text_value = path

        dialog.set_on_done(selected)
        self.window.show_dialog(dialog)

    def _set_busy(self, busy):
        self._busy = busy
        self.load_button.enabled = not busy
        self.scene_choice.enabled = not busy and bool(self._scenes)
        self.frame_slider.enabled = not busy and len(self._frames) > 1
        self.previous_button.enabled = not busy and bool(self._frames) and self._frame_index > 0
        self.next_button.enabled = not busy and self._frame_index + 1 < len(self._frames)
        self.color_choice.enabled = not busy
        self.raw_checkbox.enabled = (
            not busy and self._adapter is not None and self._adapter.supports_point_timing
        )

    def _submit(self, job, on_success, message, *, on_error=None):
        self._request_id += 1
        request_id = self._request_id
        if self._future is not None:
            self._future.cancel()
        self._set_busy(True)
        self.status.text = message
        self._future = self._executor.submit(job)

        def completed(future):
            if not self._alive or future.cancelled():
                return

            def apply_result():
                if not self._alive or request_id != self._request_id:
                    return
                try:
                    result = future.result()
                    on_success(result)
                except Exception as error:
                    self._restore_navigation()
                    if on_error is not None:
                        on_error()
                    self.status.text = "Loading failed. Check the dataset root and files."
                    self._show_error(error)
                finally:
                    self._set_busy(False)

            self.application.post_to_main_thread(self.window, apply_result)

        self._future.add_done_callback(completed)

    def load_dataset(self, *, initial_scene=None, initial_frame=0):
        name = self.dataset_choice.selected_text
        root_text = self.root_edit.text_value.strip()
        root = Path(root_text).expanduser() if root_text else None
        version = self.version_edit.text_value.strip()
        mode = self.renderer.settings.color_scheme
        raw = self.renderer.settings.use_raw_points

        def job():
            adapter = create_dataset(name, root=root, version=version)
            scenes = adapter.list_scenes()
            if not scenes:
                raise ValueError("This dataset contains no scenes with LiDAR keyframes.")
            scene_index = 0
            if initial_scene is not None:
                matches = [i for i, item in enumerate(scenes) if initial_scene in (item.name, item.token)]
                if not matches:
                    raise ValueError(f"Scene {initial_scene!r} was not found in this dataset.")
                scene_index = matches[0]
            frames = adapter.list_frames(scenes[scene_index].token)
            if not 0 <= initial_frame < len(frames):
                raise ValueError(f"Frame index must be between 0 and {len(frames) - 1} for this scene.")
            data = adapter.load_frame(frames[initial_frame].token)
            data = adapter.prepare_frame(
                data,
                camera_rgb=mode == "camera_rgb" and adapter.supports_camera_rgb,
                point_timing=raw and adapter.supports_point_timing,
            )
            return adapter, scenes, scene_index, frames, initial_frame, data

        def loaded(result):
            adapter, scenes, scene_index, frames, frame_index, data = result
            display_mode = mode if mode != "camera_rgb" or adapter.supports_camera_rgb else "gray"
            display_raw = raw and adapter.supports_point_timing
            self._render_point_display(data, display_mode, display_raw, reset_camera=True)
            self._adapter, self._scenes = adapter, scenes
            self._scene_index, self._frames = scene_index, frames
            self._frame_index = frame_index
            self.scene_choice.clear_items()
            for item in scenes:
                self.scene_choice.add_item(item.name)
            self.scene_choice.selected_index = scene_index
            self._rebuild_classes(adapter.class_names)
            self._configure_color_choices()
            self.renderer.update_boxes()
            self._update_frame_controls(data)

        self._submit(job, loaded, f"Loading {name}…")

    def _on_scene_choice(self, name, index):
        if self._busy or self._adapter is None or index == self._scene_index:
            return
        adapter = self._adapter
        token = self._scenes[index].token
        mode, raw = self.renderer.settings.color_scheme, self.renderer.settings.use_raw_points

        def job():
            frames = adapter.list_frames(token)
            if not frames:
                raise ValueError("This scene has no LiDAR keyframes.")
            data = self._load_prepared_frame(adapter, frames[0].token, mode, raw)
            return frames, data

        def loaded(result):
            frames, data = result
            self.renderer.set_frame(data, reset_camera=True)
            self._frames = frames
            self._scene_index, self._frame_index = index, 0
            self.scene_choice.selected_index = index
            self._update_frame_controls(data)

        self._submit(job, loaded, "Loading scene…")

    def _on_frame_slider(self, value):
        self._request_frame(int(round(value)))

    def _request_frame(self, index):
        if self._busy or self._adapter is None or index == self._frame_index:
            return
        if not 0 <= index < len(self._frames):
            return
        adapter, token = self._adapter, self._frames[index].token
        mode, raw = self.renderer.settings.color_scheme, self.renderer.settings.use_raw_points

        def loaded(data):
            self.renderer.set_frame(data, reset_camera=False)
            self._frame_index = index
            self._update_frame_controls(data)

        self._submit(
            lambda: self._load_prepared_frame(adapter, token, mode, raw),
            loaded,
            f"Loading frame {index + 1}…",
        )

    @staticmethod
    def _load_prepared_frame(adapter, token, mode, raw):
        frame = adapter.load_frame(token)
        return adapter.prepare_frame(frame, camera_rgb=mode == "camera_rgb", point_timing=raw)

    def _restore_navigation(self):
        if self._scenes:
            self.scene_choice.selected_index = self._scene_index
        if self._frames:
            self.frame_slider.int_value = self._frame_index

    def _update_frame_controls(self, data):
        scene = self._scenes[self._scene_index]
        self.scene_description.text = textwrap.fill(scene.description or scene.name, width=42)
        self.frame_slider.set_limits(0, max(1, len(self._frames) - 1))
        self.frame_slider.int_value = self._frame_index
        elapsed = (data.info.timestamp_us - self._frames[0].timestamp_us) / 1_000_000
        self.frame_label.text = f"Frame {self._frame_index + 1} / {len(self._frames)} · {elapsed:.2f} s"
        self.frame_label.tooltip = data.info.token
        self.status.text = f"{len(data.points):,} points · {len(data.boxes)} boxes"
        self._update_color_controls()
        self._update_class_counts()
        self.window.set_needs_layout()

    def _rebuild_classes(self, class_names):
        em = self.window.theme.font_size
        column = gui.Vert(0.2 * em)
        self._class_checkboxes.clear()
        self._class_count_labels.clear()
        self._class_color_buttons.clear()
        settings = self.renderer.settings
        settings.selected_classes = set(class_names)
        settings.class_colors = {name: class_color(name) for name in class_names}
        for name in class_names:
            row = gui.Horiz(0.3 * em)
            checkbox = gui.Checkbox(name)
            checkbox.checked = True
            checkbox.tooltip = name
            checkbox.set_on_checked(partial(self._on_class_checked, name))
            self._class_checkboxes[name] = checkbox
            row.add_child(checkbox)
            count = gui.Label("(0)")
            self._class_count_labels[name] = count
            row.add_child(count)
            row.add_stretch()
            color = gui.Button("Color")
            color.background_color = gui.Color(*settings.class_colors[name])
            color.tooltip = f"Bounding box color for {name}"
            color.set_on_clicked(partial(self._pick_class_color, name))
            self._class_color_buttons[name] = color
            row.add_child(color)
            column.add_child(row)
        self.class_list.set_widget(column)
        self.window.set_needs_layout()

    def _update_class_counts(self):
        if self.renderer.frame is None:
            return
        counts = {}
        for box in self.renderer.frame.boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        for name, label in self._class_count_labels.items():
            label.text = f"({counts.get(name, 0)})"

    def _on_class_checked(self, name, checked):
        if checked:
            self.renderer.settings.selected_classes.add(name)
        else:
            self.renderer.settings.selected_classes.discard(name)
        self.renderer.update_boxes()

    def _select_classes(self, selected):
        for checkbox in self._class_checkboxes.values():
            checkbox.checked = selected
        self.renderer.settings.selected_classes = set(self._class_checkboxes) if selected else set()
        self.renderer.update_boxes()

    def _on_class_color(self, name, color):
        self.renderer.settings.class_colors[name] = (color.red, color.green, color.blue)
        self._class_color_buttons[name].background_color = color
        self.renderer.update_boxes()

    def _pick_class_color(self, name):
        dialog = gui.Dialog("Bounding box color")
        em = self.window.theme.font_size
        content = gui.Vert(em, gui.Margins(em, em, em, em))
        content.add_child(gui.Label(name))
        picker = gui.ColorEdit()
        picker.color_value = gui.Color(*self.renderer.settings.class_colors[name])
        picker.set_on_value_changed(partial(self._on_class_color, name))
        content.add_child(picker)
        done = gui.Button("Done")
        done.set_on_clicked(self.window.close_dialog)
        content.add_child(done)
        dialog.add_child(content)
        self.window.show_dialog(dialog)

    def _on_point_size(self, value):
        self.renderer.settings.point_size = float(value)
        self.renderer.update_points()

    def _on_color_scheme(self, name, index):
        mode = "camera_rgb" if name == "Camera RGB" else name
        self._request_point_display(mode=mode)

    def _on_raw_points(self, checked):
        self._request_point_display(raw=checked)

    def _configure_color_choices(self):
        self.color_choice.clear_items()
        for mode in COLOR_SCHEMES:
            self.color_choice.add_item(mode)
        if self._adapter.supports_camera_rgb:
            self.color_choice.add_item("Camera RGB")
        self._update_color_controls()

    def _update_color_controls(self):
        settings = self.renderer.settings
        is_rgb = settings.color_scheme == "camera_rgb"
        self.color_choice.selected_text = "Camera RGB" if is_rgb else settings.color_scheme
        for widget in (self.distance_label, self.distance_edit, self.distance_hint):
            widget.visible = not is_rgb
        self.raw_checkbox.checked = settings.use_raw_points
        self.raw_hint.visible = settings.use_raw_points
        frame = self.renderer.frame
        if is_rgb and frame is not None and frame.rgb_valid_mask is not None:
            count = int(frame.rgb_valid_mask.sum())
            self.color_status.text = f"Camera RGB: {count:,} / {len(frame.points):,} points\nUnmatched points use gray."
        else:
            self.color_status.text = ""
        self.color_status.visible = is_rgb
        self.window.set_needs_layout()

    def _render_point_display(self, frame, mode, raw, *, reset_camera=False):
        settings = self.renderer.settings
        previous_mode, previous_raw = settings.color_scheme, settings.use_raw_points
        settings.color_scheme, settings.use_raw_points = mode, raw
        try:
            self.renderer.set_frame(frame, reset_camera=reset_camera)
        except Exception:
            settings.color_scheme, settings.use_raw_points = previous_mode, previous_raw
            raise

    def _request_point_display(self, *, mode=None, raw=None):
        settings = self.renderer.settings
        mode = settings.color_scheme if mode is None else mode
        raw = settings.use_raw_points if raw is None else raw
        frame, adapter = self.renderer.frame, self._adapter
        if frame is None:
            settings.color_scheme, settings.use_raw_points = mode, raw
            self._update_color_controls()
            return
        if self._busy:
            self._update_color_controls()
            return

        def loaded(data):
            self._render_point_display(data, mode, raw)
            self._update_frame_controls(data)

        needs_rgb = mode == "camera_rgb" and frame.rgb_colors is None
        needs_timing = raw and frame.raw_points is None
        if needs_rgb or needs_timing:
            self._submit(
                lambda: adapter.prepare_frame(frame, camera_rgb=needs_rgb, point_timing=needs_timing),
                loaded,
                "Preparing point colors and acquisition times…",
                on_error=self._update_color_controls,
            )
        else:
            loaded(frame)

    def _on_distance(self, value):
        self.renderer.settings.max_distance = max(1.0, float(value))
        self.renderer.update_points(recolor=True)

    def _on_line_width(self, value):
        self.renderer.settings.line_width = float(value)
        self.renderer.update_boxes()

    def _show_error(self, error):
        print(f"3dviz: {error}", file=sys.stderr)
        dialog = gui.Dialog("Unable to load dataset")
        em = self.window.theme.font_size
        content = gui.Vert(em, gui.Margins(em, em, em, em))
        content.add_child(gui.Label(textwrap.fill(str(error), width=75)))
        button = gui.Button("OK")
        button.set_on_clicked(self.window.close_dialog)
        content.add_child(button)
        dialog.add_child(content)
        self.window.show_dialog(dialog)

    def _on_close(self):
        self._alive = False
        self._request_id += 1
        self._executor.shutdown(wait=False, cancel_futures=True)
        return True


def run_app(
    *, dataset="nuscenes", root=None, version="v1.0-mini", scene=None, frame=0, autoload=False
):
    application = gui.Application.instance
    application.initialize()
    # Retain the controller while the native event loop owns its window.
    viewer = ViewerWindow(
        dataset=dataset, root=root, version=version, scene=scene, frame=frame, autoload=autoload
    )
    try:
        application.run()
    finally:
        if viewer._alive:
            viewer._on_close()
