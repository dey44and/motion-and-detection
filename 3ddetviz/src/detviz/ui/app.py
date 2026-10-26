"""Detection review uses the existing 3D viewer's cloud and mouse controls."""

from __future__ import annotations

from dataclasses import fields, replace
from functools import partial
import json
import math
import os
from pathlib import Path
import textwrap
import time
import tempfile

from open3d.visualization import gui

from threedviz.dataloader import create_dataset
from threedviz.ui.app import ViewerWindow
from threedviz.visualization.renderer import BoxLayer

from ..controller import ReviewDatasetAdapter, refilter
from ..predictions import open_predictions
from ..processing.filters import FilterConfig


_NMS_LABELS = {"Off — as saved": "off", "Per class": "class", "CenterPoint head groups": "groups", "Class agnostic": "agnostic"}


class DetectionWindow(ViewerWindow):
    window_title = "3ddetviz · Detection review"
    viewer_title = "3ddetviz"

    def __init__(self, *, predictions=None, cache_dir=None, split="val", **kwargs):
        self._prediction_path = "" if predictions is None else str(predictions)
        self._cache_dir = cache_dir
        self._split = split
        self._filter_config = FilterConfig()
        self._syncing_filters = False
        self._filter_deadline = None
        self._gt_color = (0.08, 0.55, 0.22)
        self._detection_color = (0.9, 0.25, 0.03)
        self._show_gt = True
        self._show_detections = True
        self._progress = None
        self._sources = []
        self._load_options = None
        self._last_connection_adapter = None
        self.last_error = None
        super().__init__(**kwargs)
        self.window.set_on_tick_event(self._on_review_tick)

    def _extend_dataset_controls(self, connection, em):
        connection.add_child(gui.Label("Predictions file"))
        self.prediction_path = gui.TextEdit()
        self.prediction_path.text_value = self._prediction_path
        self.prediction_path.tooltip = "results_nusc.json or OpenPCDet result.pkl; boxes are joined by sample token"
        connection.add_child(self.prediction_path)
        row = gui.Horiz(.3 * em)
        self.prediction_browse = gui.Button("Browse predictions")
        self.prediction_browse.set_on_clicked(self._browse_predictions)
        row.add_child(self.prediction_browse)
        self.prediction_load = gui.Button("Load predictions")
        self.prediction_load.set_on_clicked(self.load_predictions)
        row.add_child(self.prediction_load)
        connection.add_child(row)
        self.source_status = gui.Label("No prediction source loaded.")
        connection.add_child(self.source_status)

    def _extend_navigation_controls(self, navigation, em):
        navigation.add_child(gui.Label("Split"))
        self.split_choice = gui.Combobox()
        for name in ("val", "train", "all", "mini_val", "mini_train", "test"):
            self.split_choice.add_item(name)
        self.split_choice.selected_text = self._split
        self.split_choice.tooltip = "On nuScenes mini, val/train select mini_val/mini_train."
        self.split_choice.set_on_selection_changed(self._on_split)
        navigation.add_child(self.split_choice)
        self.coverage_label = gui.Label("")
        navigation.add_child(self.coverage_label)

    def _build_box_section(self, em):
        boxes = gui.CollapsableVert("Ground truth and detections", .4 * em)
        self.gt_checkbox = gui.Checkbox("Show ground truth")
        self.gt_checkbox.checked = True
        self.gt_checkbox.set_on_checked(lambda checked: self._set_layer_visible("gt", checked))
        self.detection_checkbox = gui.Checkbox("Show detections")
        self.detection_checkbox.checked = True
        self.detection_checkbox.set_on_checked(lambda checked: self._set_layer_visible("detections", checked))
        for name, checkbox in (("gt", self.gt_checkbox), ("detections", self.detection_checkbox)):
            row = gui.Horiz(.3 * em)
            row.add_child(checkbox)
            row.add_stretch()
            button = gui.Button("Color")
            button.background_color = gui.Color(*(self._gt_color if name == "gt" else self._detection_color))
            button.set_on_clicked(partial(self._pick_layer_color, name))
            setattr(self, f"{name}_color_button", button)
            row.add_child(button)
            boxes.add_child(row)
        boxes.add_child(gui.Label("Line width (pixels)"))
        self.line_width = gui.Slider(gui.Slider.DOUBLE)
        self.line_width.set_limits(1, 10)
        self.line_width.double_value = self.renderer.settings.line_width
        self.line_width.set_on_value_changed(self._on_line_width)
        boxes.add_child(self.line_width)
        classes = gui.CollapsableVert("Classes", .3 * em)
        classes.set_is_open(False)
        row = gui.Horiz(.3 * em)
        for text, selected in (("Show all classes", True), ("Hide all classes", False)):
            button = gui.Button(text)
            button.set_on_clicked(partial(self._select_classes, selected))
            row.add_child(button)
        classes.add_child(row)
        self.class_list = gui.WidgetProxy()
        self.class_list.set_widget(gui.Label("Classes appear after loading."))
        classes.add_child(self.class_list)
        boxes.add_child(classes)
        return boxes

    def _configure_cloud_section(self, cloud):
        cloud.set_is_open(False)

    def _extend_sidebar(self, em):
        filters = gui.CollapsableVert("Detection filters", .4 * em)
        filters.add_child(gui.Label("Minimum confidence"))
        row = gui.Horiz(.3 * em)
        self.score_slider = gui.Slider(gui.Slider.DOUBLE)
        self.score_slider.set_limits(0, 1)
        self.score_slider.double_value = self._filter_config.score_threshold
        self.score_slider.set_on_value_changed(self._score_changed)
        row.add_child(self.score_slider)
        self.score_edit = self._number(0, 1, self._filter_config.score_threshold, self._score_edit_changed)
        self.score_edit.tooltip = "Global confidence threshold; preset per-class overrides take precedence."
        row.add_child(self.score_edit)
        filters.add_child(row)
        filters.add_child(gui.Label("Additional NMS"))
        self.nms_choice = gui.Combobox()
        for label in _NMS_LABELS:
            self.nms_choice.add_item(label)
        self.nms_choice.set_on_selection_changed(lambda *_: self._filter_changed())
        filters.add_child(self.nms_choice)
        filters.add_child(gui.Label("NMS overlap threshold (BEV IoU)"))
        row = gui.Horiz(.3 * em)
        self.iou_slider = gui.Slider(gui.Slider.DOUBLE)
        self.iou_slider.set_limits(0, 1)
        self.iou_slider.double_value = .2
        self.iou_slider.set_on_value_changed(self._iou_changed)
        self.iou_edit = self._number(0, 1, .2, self._iou_edit_changed)
        row.add_child(self.iou_slider)
        row.add_child(self.iou_edit)
        filters.add_child(row)
        self.range_checkbox = gui.Checkbox("Limit maximum horizontal distance")
        self.range_checkbox.set_on_checked(lambda _: self._filter_changed())
        filters.add_child(self.range_checkbox)
        self.min_distance_edit, self.max_distance_edit = self._number(0, 10000, 0), self._number(0, 10000, 80)
        self.min_distance_edit.tooltip = "Minimum distance always applies; zero includes nearby detections."
        self.max_distance_edit.tooltip = "Used when the maximum distance limit is enabled."
        self._add_pair(filters, "Minimum / maximum (metres)", self.min_distance_edit, self.max_distance_edit, em)
        self.height_checkbox = gui.Checkbox("Limit box center height")
        self.height_checkbox.set_on_checked(self._height_enabled)
        filters.add_child(self.height_checkbox)
        row = gui.Horiz(.3 * em)
        self.min_height_checkbox = gui.Checkbox("Minimum z")
        self.max_height_checkbox = gui.Checkbox("Maximum z")
        for checkbox in (self.min_height_checkbox, self.max_height_checkbox):
            checkbox.set_on_checked(lambda _: self._filter_changed())
            row.add_child(checkbox)
        filters.add_child(row)
        self.min_height_edit, self.max_height_edit = self._number(-10000, 10000, -5), self._number(-10000, 10000, 5)
        self._add_pair(filters, "Minimum / maximum z (metres)", self.min_height_edit, self.max_height_edit, em)
        filters.add_child(gui.Label("Pre-NMS / output limit (-1 = unlimited)"))
        self.pre_limit_edit, self.output_limit_edit = self._number(-1, 1000000, -1, integer=True), self._number(-1, 1000000, -1, integer=True)
        row = gui.Horiz(.3 * em)
        row.add_child(self.pre_limit_edit)
        row.add_child(self.output_limit_edit)
        filters.add_child(row)
        self.filter_counts = gui.Label("Load predictions to inspect filtering.")
        filters.add_child(self.filter_counts)
        self.filter_hint = gui.Label("Saved exports may already have score filtering and NMS.")
        filters.add_child(self.filter_hint)
        reset = gui.Button("Reset detection filters")
        reset.set_on_clicked(self.reset_filters)
        filters.add_child(reset)
        row = gui.Horiz(.3 * em)
        for text, callback in (("Save preset", self.save_preset), ("Load preset", self.load_preset)):
            button = gui.Button(text)
            button.set_on_clicked(callback)
            row.add_child(button)
        filters.add_child(row)
        self.sidebar.add_child(filters)
        self._sync_filter_controls()

    def _number(self, minimum, maximum, value, callback=None, *, integer=False):
        edit = gui.NumberEdit(gui.NumberEdit.INT if integer else gui.NumberEdit.DOUBLE)
        edit.set_limits(minimum, maximum)
        if integer:
            edit.int_value = value
        else:
            edit.double_value = value
        edit.set_on_value_changed(callback or (lambda _: self._filter_changed()))
        return edit

    @staticmethod
    def _add_pair(parent, label, left, right, em):
        parent.add_child(gui.Label(label))
        row = gui.Horiz(.3 * em)
        row.add_child(left)
        row.add_child(right)
        parent.add_child(row)

    def _browse_predictions(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "Choose predictions", self.window.theme)
        dialog.add_filter(".json .pkl .pickle", "Prediction files")
        dialog.add_filter("", "All files")
        dialog.set_on_cancel(self.window.close_dialog)
        def selected(path):
            self.window.close_dialog()
            self.prediction_path.text_value = path
        dialog.set_on_done(selected)
        self.window.show_dialog(dialog)

    def _set_busy(self, busy):
        super()._set_busy(busy)
        if hasattr(self, "prediction_load"):
            for widget in (self.prediction_load, self.prediction_browse, self.prediction_path, self.split_choice, self.dataset_choice, self.root_edit, self.version_edit, self.browse_button):
                widget.enabled = not busy
            if not busy:
                self._on_dataset_choice(self.dataset_choice.selected_text, 0)

    def _prune_sources(self):
        active = None if self._adapter is None else self._adapter.source
        for source in self._sources:
            if source is not active:
                source.close()
        self._sources = [] if active is None else [active]

    def _submit(self, job, on_success, message, *, on_error=None):
        self.last_error = None
        def failed():
            if on_error is not None:
                on_error()
            self._prune_sources()
            # The active frame survives failed IO. Reapply the visible policy
            # so a pending slider edit is never silently discarded by failure.
            if self.renderer.frame is not None:
                self._filter_deadline = time.monotonic() + .15
        super()._submit(job, on_success, message, on_error=failed)

    def load_dataset(self, *, initial_scene=None, initial_frame=0):
        if self._busy:
            return
        if not self._capture_filter_controls():
            return
        self._filter_deadline = None
        self._load_options = (self.prediction_path.text_value.strip(), self.split_choice.selected_text, self._filter_config)
        super().load_dataset(initial_scene=initial_scene, initial_frame=initial_frame)

    def _open_source(self, path):
        if not path:
            return None
        return open_predictions(Path(path).expanduser(), cache_dir=self._cache_dir,
                                progress=lambda fraction, message: setattr(self, "_progress", (fraction, message)),
                                cancelled=lambda: not self._alive)

    def _create_dataset_adapter(self, name, root, version):
        path, split, config = self._load_options
        base = create_dataset(name, root=root, version=version)
        # Validate split before indexing a potentially large prediction file.
        try:
            base.list_scenes_for_split(split)
            source = None if name == "demo" else self._open_source(path)
        except Exception:
            base.close()
            raise
        if source is not None:
            self._sources.append(source)
        return ReviewDatasetAdapter(base, split=split, source=source, config=config, demo=name == "demo")

    def load_predictions(self):
        if self._busy:
            return
        if self._adapter is None:
            self.load_dataset()
            return
        if not self._capture_filter_controls():
            return
        path, current, old = self.prediction_path.text_value.strip(), self.renderer.frame, self._adapter
        if old.demo:
            self._show_error(ValueError("The demo generates its own detections. Select a real dataset to load a prediction file."))
            return
        config = self._filter_config
        def job():
            source = self._open_source(path)
            adapter = ReviewDatasetAdapter(old.base, split=old.split, source=source, config=config)
            try:
                return adapter, adapter.compose(current)
            except Exception:
                if source is not None:
                    source.close()
                raise
        def loaded(result):
            adapter, data = result
            self._adapter = adapter
            if adapter.source is not None:
                self._sources.append(adapter.source)
            self._rebuild_classes(adapter.class_names)
            self._replace_review_boxes(data)
            self._update_frame_controls(data)
        self._submit(job, loaded, "Indexing predictions…")

    def _on_split(self, name, _index):
        if self._busy or self._adapter is None or name == self._adapter.split:
            return
        adapter = self._adapter
        if not self._capture_filter_controls():
            self.split_choice.selected_text = adapter.split
            return
        config = self._filter_config
        mode, raw = self.renderer.settings.color_scheme, self.renderer.settings.use_raw_points
        def job():
            scenes = adapter.base.list_scenes_for_split(name)
            if not scenes:
                raise ValueError(f"No scenes from split {name!r} are installed.")
            updated = ReviewDatasetAdapter(adapter.base, split=name, source=adapter.source, config=config, demo=adapter.demo)
            frames = updated.list_frames(scenes[0].token)
            data = self._load_prepared_frame(updated, frames[0].token, mode, raw)
            return updated, scenes, frames, data
        def loaded(result):
            updated, scenes, frames, data = result
            self._render_loaded_frame(data, reset_camera=True)
            self._adapter, self._scenes, self._frames = updated, scenes, frames
            self._scene_index = self._frame_index = 0
            self.scene_choice.clear_items()
            for scene in scenes:
                self.scene_choice.add_item(scene.name)
            self.scene_choice.selected_index = 0
            self._update_frame_controls(data)
        self._submit(job, loaded, "Loading split…", on_error=lambda: setattr(self.split_choice, "selected_text", adapter.split))

    def _request_frame(self, index):
        if self._busy or self._adapter is None or index == self._frame_index:
            return
        if not self._capture_filter_controls():
            return
        self._adapter.config = self._filter_config
        super()._request_frame(index)

    def _on_scene_choice(self, name, index):
        if self._busy or self._adapter is None or index == self._scene_index:
            return
        if not self._capture_filter_controls():
            return
        self._adapter.config = self._filter_config
        super()._on_scene_choice(name, index)

    def _layers(self, data):
        predictions = () if data.filtered is None else tuple(detection.box for detection in data.filtered.detections)
        return (BoxLayer("gt", data.boxes, self._gt_color, self._show_gt),
                BoxLayer("detections", predictions, self._detection_color, self._show_detections))

    def _render_loaded_frame(self, data, *, reset_camera=False):
        self.renderer.set_frame(data, reset_camera=reset_camera, box_layers=self._layers(data))

    def _replace_review_boxes(self, data):
        self.renderer.set_box_layers(self._layers(data))
        self.renderer.frame = data

    def _rebuild_classes(self, class_names):
        previous = {name: checkbox.checked for name, checkbox in self._class_checkboxes.items()}
        em = self.window.theme.font_size
        column = gui.Vert(.2 * em)
        self._class_checkboxes.clear()
        self._class_count_labels.clear()
        self._class_color_buttons.clear()
        for name in class_names:
            row = gui.Horiz(.3 * em)
            checkbox = gui.Checkbox(name)
            checkbox.checked = previous.get(name, True)
            checkbox.set_on_checked(partial(self._on_class_checked, name))
            self._class_checkboxes[name] = checkbox
            row.add_child(checkbox)
            label = gui.Label("")
            self._class_count_labels[name] = label
            row.add_child(label)
            column.add_child(row)
        self.renderer.settings.selected_classes = {name for name, checkbox in self._class_checkboxes.items() if checkbox.checked}
        self.class_list.set_widget(column)
        self.window.set_needs_layout()

    def _update_class_counts(self):
        data = self.renderer.frame
        if data is None:
            return
        from collections import Counter
        gt = Counter(box.class_name for box in data.boxes)
        pred = Counter(detection.box.class_name for detection in (() if data.filtered is None else data.filtered.detections))
        for name, label in self._class_count_labels.items():
            label.text = f"GT {gt[name]} · Det {pred[name]}"

    def _update_frame_controls(self, data):
        super()._update_frame_controls(data)
        adapter = self._adapter
        if adapter is None:
            return
        if adapter is not self._last_connection_adapter:
            self.connection_section.set_is_open(False)
            self._last_connection_adapter = adapter
        self._prune_sources()
        self.split_choice.selected_text = adapter.split
        source = adapter.source
        if adapter.demo:
            self.source_status.text = "Generated detections · includes duplicate proposals"
            self.coverage_label.text = "Demo validation scenes"
            self.filter_hint.text = "Generated candidates; filters always start from the original boxes."
        elif source is None:
            self.source_status.text = "No prediction source loaded."
            self.coverage_label.text = "Ground truth only"
        else:
            self.source_status.text = textwrap.fill(f"{source.info.name} · {source.info.coordinate_frame} coordinates", 42)
            covered = sum(frame.token in source.frame_tokens for frame in self._frames)
            self.coverage_label.text = f"Prediction coverage: {covered} / {len(self._frames)} scene frames"
            stage = source.info.stage
            self.filter_hint.text = textwrap.fill(
                "Pre-NMS candidates; export score floor and candidate limits still apply." if stage == "pre_nms" else
                "Saved boxes only. Earlier score filtering or NMS cannot be undone here.", 44)
        if data.predictions is None:
            text = "Predictions unavailable for this frame." if source is not None else "No predictions loaded."
            self.filter_counts.text = text
            self.status.text = f"{len(data.points):,} points · {len(data.boxes)} GT\n{text}"
        else:
            result = data.filtered
            self.filter_counts.text = (f"Loaded {result.loaded_count:,} → score {result.score_count:,}\n"
                                       f"Spatial {result.spatial_count:,} → pre-limit {result.pre_nms_count:,}\n"
                                       f"NMS {result.nms_count:,} → output {result.output_count:,}")
            self.status.text = f"{len(data.points):,} points · {len(data.boxes)} GT · {result.output_count} detections"
        self.window.set_needs_layout()

    def _set_layer_visible(self, name, checked):
        if name == "gt":
            self._show_gt = checked
        else:
            self._show_detections = checked
        if self.renderer.frame is not None:
            self.renderer.set_box_layers(self._layers(self.renderer.frame))

    def _pick_layer_color(self, name):
        dialog = gui.Dialog("Ground truth color" if name == "gt" else "Detection color")
        em = self.window.theme.font_size
        content = gui.Vert(em, gui.Margins(em, em, em, em))
        picker = gui.ColorEdit()
        picker.color_value = gui.Color(*(self._gt_color if name == "gt" else self._detection_color))
        picker.set_on_value_changed(partial(self._on_layer_color, name))
        content.add_child(picker)
        done = gui.Button("Done")
        done.set_on_clicked(self.window.close_dialog)
        content.add_child(done)
        dialog.add_child(content)
        self.window.show_dialog(dialog)

    def _on_layer_color(self, name, color):
        value = (color.red, color.green, color.blue)
        if name == "gt":
            self._gt_color = value
        else:
            self._detection_color = value
        getattr(self, f"{name}_color_button").background_color = color
        if self.renderer.frame is not None:
            self.renderer.set_box_layers(self._layers(self.renderer.frame))

    def _score_changed(self, value):
        self.score_edit.double_value = value
        self._filter_changed()

    def _score_edit_changed(self, value):
        self.score_slider.double_value = value
        self._filter_changed()

    def _iou_changed(self, value):
        self.iou_edit.double_value = value
        self._filter_changed()

    def _iou_edit_changed(self, value):
        self.iou_slider.double_value = value
        self._filter_changed()

    def _read_filter_controls(self):
        ranged, height = self.range_checkbox.checked, self.height_checkbox.checked
        return replace(self._filter_config, score_threshold=self.score_edit.double_value,
                       min_distance=self.min_distance_edit.double_value,
                       max_distance=self.max_distance_edit.double_value if ranged else None,
                       min_height=self.min_height_edit.double_value if height and self.min_height_checkbox.checked else None,
                       max_height=self.max_height_edit.double_value if height and self.max_height_checkbox.checked else None,
                       pre_nms_limit=None if self.pre_limit_edit.int_value < 0 else self.pre_limit_edit.int_value,
                       max_detections=None if self.output_limit_edit.int_value < 0 else self.output_limit_edit.int_value,
                       nms_mode=_NMS_LABELS[self.nms_choice.selected_text], nms_iou_threshold=self.iou_edit.double_value)

    def _capture_filter_controls(self):
        try:
            self._filter_config = self._read_filter_controls()
        except (ValueError, TypeError) as error:
            self.status.text = f"Invalid filters: {error}"
            return False
        self._filter_deadline = None
        return True

    def _filter_changed(self):
        if self._syncing_filters:
            return
        self._filter_deadline = time.monotonic() + .15

    def _height_enabled(self, checked):
        if checked and not self.min_height_checkbox.checked and not self.max_height_checkbox.checked:
            self.min_height_checkbox.checked = self.max_height_checkbox.checked = True
        self._filter_changed()

    def _on_review_tick(self):
        changed = False
        if self._progress is not None:
            fraction, message = self._progress
            self._progress = None
            if self._busy:
                self.status.text = f"{message} ({fraction:.0%})"
                changed = True
        if self._filter_deadline is not None and not self._busy and time.monotonic() >= self._filter_deadline:
            self._filter_deadline = None
            self.apply_filters()
            return True
        return changed

    def apply_filters(self, config=None):
        if self._busy:
            self._filter_deadline = time.monotonic() + .15
            return
        try:
            config = self._read_filter_controls() if config is None else config
        except (ValueError, TypeError) as error:
            self.status.text = f"Invalid filters: {error}"
            return
        self._filter_config = config
        if self._adapter is not None:
            self._adapter.config = config
        data = self.renderer.frame
        if data is None or data.predictions is None:
            return
        def loaded(filtered):
            self._replace_review_boxes(filtered)
            self._update_frame_controls(filtered)
        super()._submit(lambda: refilter(data, config), loaded, "Filtering detections…")

    def _sync_filter_controls(self):
        config = self._filter_config
        self._syncing_filters = True
        try:
            self.score_slider.double_value = self.score_edit.double_value = config.score_threshold
            self.nms_choice.selected_text = next(label for label, mode in _NMS_LABELS.items() if mode == config.nms_mode)
            self.iou_edit.double_value = config.nms_iou_threshold
            self.iou_slider.double_value = config.nms_iou_threshold
            self.range_checkbox.checked = config.max_distance is not None
            self.min_distance_edit.double_value = config.min_distance
            self.max_distance_edit.double_value = config.max_distance if config.max_distance is not None else 80
            self.height_checkbox.checked = config.min_height is not None or config.max_height is not None
            self.min_height_checkbox.checked = config.min_height is not None
            self.max_height_checkbox.checked = config.max_height is not None
            self.min_height_edit.double_value = config.min_height if config.min_height is not None else -5
            self.max_height_edit.double_value = config.max_height if config.max_height is not None else 5
            self.pre_limit_edit.int_value = -1 if config.pre_nms_limit is None else config.pre_nms_limit
            self.output_limit_edit.int_value = -1 if config.max_detections is None else config.max_detections
        finally:
            self._syncing_filters = False

    def reset_filters(self):
        self._filter_config = FilterConfig(score_threshold=0)
        self._sync_filter_controls()
        self._filter_changed()

    def save_preset(self):
        self._preset_dialog(save=True)

    def load_preset(self):
        self._preset_dialog(save=False)

    def _preset_dialog(self, *, save):
        dialog = gui.FileDialog(gui.FileDialog.SAVE if save else gui.FileDialog.OPEN, "Save preset" if save else "Load preset", self.window.theme)
        dialog.add_filter(".json", "Filter preset")
        dialog.set_on_cancel(self.window.close_dialog)
        def selected(path):
            self.window.close_dialog()
            try:
                self.write_preset(Path(path)) if save else self.read_preset(Path(path))
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._show_error(error)
        dialog.set_on_done(selected)
        self.window.show_dialog(dialog)

    def write_preset(self, path):
        config = self._read_filter_controls()
        configuration = {field.name: getattr(config, field.name) for field in fields(config)}
        configuration["per_class_scores"] = dict(config.per_class_scores)
        payload = {"schema_version": 1, "filters": configuration,
                   "appearance": {"gt_color": self._gt_color, "detection_color": self._detection_color,
                                  "line_width": self.renderer.settings.line_width}}
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read_preset(self, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Filter presets must contain a JSON object.")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported filter preset version.")
        config = FilterConfig(**payload["filters"])
        appearance = payload.get("appearance", {})
        if not isinstance(appearance, dict):
            raise ValueError("Preset appearance must contain a JSON object.")
        gt = tuple(appearance.get("gt_color", self._gt_color))
        pred = tuple(appearance.get("detection_color", self._detection_color))
        width = float(appearance.get("line_width", self.renderer.settings.line_width))
        if any(len(color) != 3 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in color) for color in (gt, pred)) or not math.isfinite(width) or not 1 <= width <= 10:
            raise ValueError("Invalid preset colors or line width.")
        self._filter_config = config
        self._gt_color, self._detection_color = gt, pred
        self.gt_color_button.background_color = gui.Color(*gt)
        self.detections_color_button.background_color = gui.Color(*pred)
        self.renderer.settings.line_width = self.line_width.double_value = width
        if self.renderer.frame is not None:
            self.renderer.set_box_layers(self._layers(self.renderer.frame))
        self._sync_filter_controls()
        self._filter_changed()

    def _show_error(self, error):
        self.last_error = str(error)
        super()._show_error(error)

    def _on_close(self):
        self._filter_deadline = None
        result = super()._on_close()
        for source in self._sources:
            source.close()
        return result


def run_app(**kwargs):
    application = gui.Application.instance
    application.initialize()
    viewer = DetectionWindow(**kwargs)
    try:
        application.run()
    finally:
        if viewer._alive:
            viewer._on_close()
