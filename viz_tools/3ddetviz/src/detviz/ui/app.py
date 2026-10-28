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

from threedviz.ui.app import ViewerWindow
from threedviz.visualization.renderer import BoxLayer

from ..controller import compose_review, refilter
from ..predictions import open_predictions
from ..processing.filters import FilterConfig


_NMS_LABELS = {"Off — as saved": "off", "Per class": "class", "CenterPoint head groups": "groups", "Class agnostic": "agnostic"}


class DetectionWindow(ViewerWindow):
    window_title = "3ddetviz · Detection review"
    viewer_title = "3ddetviz"

    def __init__(self, *, predictions=None, cache_dir=None, split="val", prediction_format="auto", **kwargs):
        self._prediction_path = "" if predictions is None else str(predictions)
        self._cache_dir = cache_dir
        self._prediction_format = prediction_format
        self._filter_config = FilterConfig()
        self._syncing_filters = False
        self._filter_deadline = None
        self._gt_color = (0.08, 0.55, 0.22)
        self._detection_color = (0.9, 0.25, 0.03)
        self._show_gt = True
        self._show_detections = True
        self._progress = None
        self._dataset_sources = {}
        self._pending_source = None
        self._demo_datasets = set()
        self._pending_review = None
        self._review_frame = None
        self._load_options = None
        self._last_connection_loader = None
        self.last_error = None
        super().__init__(split=split, **kwargs)
        self.window.set_on_tick_event(self._on_review_tick)

    def _extend_dataset_controls(self, connection, em):
        connection.add_child(gui.Label("Predictions file"))
        self.prediction_path = gui.TextEdit()
        self.prediction_path.text_value = self._prediction_path
        self.prediction_path.tooltip = "results_nusc.json or OpenPCDet result.pkl; boxes are joined by dataset frame ID"
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
        self.split_choice = self.dataset_split_choice = gui.Combobox()
        for name in dict.fromkeys((self._dataset_split, "all")):
            self.split_choice.add_item(name)
        self.split_choice.selected_text = self._dataset_split
        self.split_choice.tooltip = "Available split names come from the connected dataset."
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
            for widget in (self.prediction_load, self.prediction_browse, self.prediction_path, self.split_choice):
                widget.enabled = not busy
            if not busy:
                self._on_dataset_choice(self.dataset_choice.selected_text, 0)

    @property
    def _source(self):
        return self._dataset_sources.get(id(self._adapter))

    def _prune_sources(self):
        if self._pending_source is not None:
            self._pending_source.close()
            self._pending_source = None
        for loader, source in tuple(self._dataset_sources.items()):
            if loader != id(self._adapter):
                if source is not None:
                    source.close()
                del self._dataset_sources[loader]
                self._demo_datasets.discard(loader)

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
        self._load_options = (self.prediction_path.text_value.strip(), self.split_choice.selected_text)
        self._dataset_split = self.split_choice.selected_text
        super().load_dataset(initial_scene=initial_scene, initial_frame=initial_frame)

    def _open_source(self, path):
        if not path:
            return None
        return open_predictions(Path(path).expanduser(), format=self._prediction_format, cache_dir=self._cache_dir,
                                progress=lambda fraction, message: setattr(self, "_progress", (fraction, message)),
                                cancelled=lambda: not self._alive)

    def _create_dataset_adapter(self, name, root, version):
        path, split = self._load_options
        loader = super()._create_dataset_adapter(name, root, version)
        try:
            loader.list_scenes(split=split)
            source = None if name == "demo" else self._open_source(path)
        except Exception:
            loader.close()
            raise
        self._dataset_sources[id(loader)] = source
        if name == "demo":
            self._demo_datasets.add(id(loader))
        return loader

    def _load_prepared_frame(self, loader, frame_id, mode, raw):
        observation = super()._load_prepared_frame(loader, frame_id, mode, raw)
        self._pending_review = compose_review(
            observation, source=self._dataset_sources.get(id(loader)),
            config=self._filter_config, demo=id(loader) in self._demo_datasets,
        )
        return observation

    def _dataset_class_labels(self, loader):
        labels = dict(super()._dataset_class_labels(loader))
        source = self._dataset_sources.get(id(loader))
        if source is not None:
            for class_id in source.class_names:
                labels.setdefault(class_id, class_id)
        return labels

    def load_predictions(self):
        if self._busy:
            return
        if self._adapter is None:
            self.load_dataset()
            return
        if not self._capture_filter_controls():
            return
        if id(self._adapter) in self._demo_datasets:
            self._show_error(ValueError("The demo generates its own detections. Select a real dataset to load a prediction file."))
            return
        path, observation = self.prediction_path.text_value.strip(), self.renderer.frame
        config = self._filter_config
        def job():
            source = self._open_source(path)
            if source is not None:
                self._pending_source = source
            try:
                return source, compose_review(observation, source=source, config=config)
            except Exception:
                if source is not None:
                    self._pending_source = None
                    source.close()
                raise
        def loaded(result):
            source, review = result
            self._pending_source = None
            previous = self._source
            self._dataset_sources[id(self._adapter)] = source
            if previous is not None and previous is not source:
                previous.close()
            labels = self._dataset_class_labels(self._adapter)
            self.renderer.set_class_labels(labels)
            self._rebuild_classes(tuple(dict.fromkeys(labels.values())))
            self._replace_review_boxes(review)
            self._update_frame_controls(observation)
        self._submit(job, loaded, "Indexing predictions…")

    def _on_split(self, name, index):
        if self._capture_filter_controls():
            super()._on_dataset_split(name, index)

    def _request_frame(self, index):
        if self._capture_filter_controls():
            super()._request_frame(index)

    def _on_scene_choice(self, name, index):
        if self._capture_filter_controls():
            super()._on_scene_choice(name, index)

    def _layers(self, review):
        predictions = () if review.filtered is None else tuple(item.box for item in review.filtered.detections)
        return (BoxLayer("gt", review.observation.annotations.boxes, self._gt_color, self._show_gt),
                BoxLayer("detections", predictions, self._detection_color, self._show_detections))

    def _render_loaded_frame(self, observation, *, reset_camera=False):
        pending = self._pending_review
        if pending is not None and pending.observation is observation:
            review = pending
        else:
            review = replace(self._review_frame, observation=observation)
        self.renderer.set_frame(observation, reset_camera=reset_camera, box_layers=self._layers(review))
        self._review_frame = review
        if review is pending:
            self._pending_review = None

    def _replace_review_boxes(self, review):
        self.renderer.set_box_layers(self._layers(review))
        self._review_frame = review

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
        data = self._review_frame
        if data is None:
            return
        from collections import Counter
        gt = Counter(self.renderer.class_label(box.class_id) for box in data.observation.annotations.boxes)
        pred = Counter(self.renderer.class_label(item.box.class_id) for item in (() if data.filtered is None else data.filtered.detections))
        for name, label in self._class_count_labels.items():
            label.text = f"GT {gt[name]} · Det {pred[name]}"

    def _update_frame_controls(self, observation):
        super()._update_frame_controls(observation)
        loader = self._adapter
        if loader is None:
            return
        if loader is not self._last_connection_loader:
            self.connection_section.set_is_open(False)
            self._last_connection_loader = loader
        self._prune_sources()
        self.split_choice.selected_text = self._dataset_split
        source, review = self._source, self._review_frame
        if id(loader) in self._demo_datasets:
            self.source_status.text = "Generated detections · includes duplicate proposals"
            self.coverage_label.text = "Demo validation scenes"
            self.filter_hint.text = "Generated candidates; filters always start from the original boxes."
        elif source is None:
            self.source_status.text = "No prediction source loaded."
            self.coverage_label.text = "Ground truth only"
        else:
            self.source_status.text = textwrap.fill(f"{source.info.name} · {source.info.coordinate_frame} coordinates", 42)
            covered = sum(frame.frame_id in source.frame_tokens for frame in self._frames)
            self.coverage_label.text = f"Prediction coverage: {covered} / {len(self._frames)} scene frames"
            self.filter_hint.text = textwrap.fill(
                "Pre-NMS candidates; export score floor and candidate limits still apply." if source.info.stage == "pre_nms" else
                "Saved boxes only. Earlier score filtering or NMS cannot be undone here.", 44)
        count = len(observation.annotations.boxes)
        if review.predictions is None:
            text = "Predictions unavailable for this frame." if source is not None else "No predictions loaded."
            self.filter_counts.text = text
            self.status.text = f"{len(observation.points):,} points · {count} GT\n{text}"
        else:
            result = review.filtered
            self.filter_counts.text = (f"Loaded {result.loaded_count:,} → score {result.score_count:,}\n"
                                       f"Spatial {result.spatial_count:,} → pre-limit {result.pre_nms_count:,}\n"
                                       f"NMS {result.nms_count:,} → output {result.output_count:,}")
            self.status.text = f"{len(observation.points):,} points · {count} GT · {result.output_count} detections"
        self.window.set_needs_layout()

    def _set_layer_visible(self, name, checked):
        if name == "gt":
            self._show_gt = checked
        else:
            self._show_detections = checked
        if self.renderer.frame is not None:
            self.renderer.set_box_layers(self._layers(self._review_frame))

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
            self.renderer.set_box_layers(self._layers(self._review_frame))

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
        supplied = config is not None
        if self._busy:
            self._filter_deadline = time.monotonic() + .15
            return
        try:
            config = self._read_filter_controls() if config is None else config
        except (ValueError, TypeError) as error:
            self.status.text = f"Invalid filters: {error}"
            return
        self._filter_config = config
        if supplied:
            self._sync_filter_controls()
        data = self._review_frame
        if data is None or data.predictions is None:
            return
        def loaded(filtered):
            self._replace_review_boxes(filtered)
            self._update_frame_controls(filtered.observation)
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
            self.renderer.set_box_layers(self._layers(self._review_frame))
        self._sync_filter_controls()
        self._filter_changed()

    def _show_error(self, error):
        self.last_error = str(error)
        super()._show_error(error)

    def _close_sources(self):
        if self._pending_source is not None:
            self._pending_source.close()
            self._pending_source = None
        for source in self._dataset_sources.values():
            if source is not None:
                source.close()
        self._dataset_sources.clear()
        self._demo_datasets.clear()

    def _on_close(self):
        if not self._alive:
            return True
        self._filter_deadline = None
        # Source cleanup follows any active import/read on the same worker.
        self._executor.submit(self._close_sources)
        return super()._on_close()


def run_app(**kwargs):
    application = gui.Application.instance
    application.initialize()
    viewer = DetectionWindow(**kwargs)
    try:
        application.run()
    finally:
        if viewer._alive:
            viewer._on_close()
