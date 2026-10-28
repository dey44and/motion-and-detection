"""Dataset selection returns the same canonical observations used in research."""

import subprocess
import sys
import numpy as np
import pytest

from dataset_utils import (
    Capability, DatasetClosedError, DatasetConfig, DatasetConfigurationError,
    DatasetLoader, DatasetMetadata, Modality, UnsupportedCapabilityError,
    register_loader,
)
from threedviz.dataloader import DemoLoader, available_datasets, create_dataset
from threedviz.dataloader.demo import DEMO_CLASSES


def test_demo_is_deterministic_complete_and_uses_canonical_records():
    seen = set()
    with create_dataset("demo") as dataset:
        assert isinstance(dataset, DatasetLoader)
        assert dataset.metadata.splits == ("all", "val", "mini_val")
        for scene in dataset.list_scenes():
            frames = dataset.list_frames(scene.scene_id, modality=Modality.POINT_CLOUD)
            assert len(frames) == scene.frame_counts[Modality.POINT_CLOUD] == 36
            first, following = map(dataset.load_point_cloud, [frames[0].frame_id, frames[1].frame_id])
            repeated = dataset.load_point_cloud(frames[0].frame_id)
            np.testing.assert_array_equal(first.points, repeated.points)
            np.testing.assert_array_equal(first.attributes["intensity"], repeated.attributes["intensity"])
            assert first.points.shape[1] == 3 and np.isfinite(first.points).all()
            np.testing.assert_array_equal(first.T_world_from_reference, np.eye(4))
            assert not np.array_equal(first.annotations.boxes[0].center, following.annotations.boxes[0].center)
            seen.update(box.class_id for box in first.annotations.boxes)
        for split in dataset.metadata.splits:
            assert dataset.list_scenes(split=split) == dataset.list_scenes()
    assert seen == set(DEMO_CLASSES)
    with pytest.raises(DatasetClosedError):
        dataset.load_point_cloud(frames[0].frame_id)


def test_shared_registration_needs_no_second_viewer_registration(monkeypatch, tmp_path):
    from dataset_utils import registry

    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    received = []

    def constructor(config):
        received.append(config)
        return DemoLoader(config)

    register_loader("research-fixture", constructor)
    assert "research-fixture" in available_datasets()
    config = DatasetConfig(root=tmp_path, options={"setting": True})
    with create_dataset("research-fixture", config=config) as dataset:
        assert dataset.config is config
        assert dataset.load_point_cloud("demo-urban-000").points.shape[1] == 3
    assert received == [config]


def test_camera_only_provider_is_closed_once_with_actionable_error(monkeypatch):
    from threedviz.dataloader import factory

    class Cameras(DemoLoader):
        close_calls = 0

        @property
        def metadata(self):
            return DatasetMetadata("camera-only", "Cameras", {Capability.CAMERAS})

        def close(self):
            self.close_calls += 1
            super().close()

    loader = Cameras()
    monkeypatch.setattr(factory, "create_loader", lambda name, config: loader)
    with pytest.raises(UnsupportedCapabilityError, match="point-cloud"):
        factory.create_dataset("camera-only")
    assert loader.close_calls == 1


@pytest.mark.parametrize("options", [{"root": "/data"}, {"version": "custom-v2"}])
def test_configuration_does_not_silently_ignore_launch_arguments(options):
    with pytest.raises(DatasetConfigurationError, match="either DatasetConfig"):
        create_dataset("nuscenes", config=DatasetConfig(), **options)


def test_catalog_and_demo_keep_native_readers_and_gui_unloaded():
    code = '''
import sys
from threedviz.dataloader import available_datasets, create_dataset
assert {"demo", "nuscenes", "waymo"}.issubset(available_datasets())
assert create_dataset("demo").metadata.classes
for prefix in ("dataset_utils.nuscenes", "dataset_utils.waymo", "open3d", "tensorflow"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
