"""Installed plugins expose names lazily and fail without ambiguous overrides."""

from dataclasses import dataclass

import pytest
import numpy as np

from visual_utils import (
    Capability, DatasetConfig, DatasetConfigurationError, DatasetLoader,
    DatasetMetadata, FrameInfo, Modality, PointCloudFrame, SceneInfo,
    available_loaders, create_loader, register_loader,
)
from visual_utils import registry


@dataclass
class Entry:
    name: str
    value: str = "lab_dataset:Loader"
    constructor: object = None
    loads: int = 0

    def load(self):
        self.loads += 1
        if isinstance(self.constructor, Exception):
            raise self.constructor
        return self.constructor


class PointProvider(DatasetLoader):
    @property
    def metadata(self):
        return DatasetMetadata("plugin-test", "Plugin test", {Capability.POINT_CLOUD})

    def _list_scenes(self, *, split="all"):
        return (SceneInfo("scene", "Scene"),)

    def _list_frames(self, scene_id, *, modality):
        return (FrameInfo("native-15", scene_id, 0, modalities={modality}),)

    def _load_point_cloud(self, frame_id):
        return PointCloudFrame(self.list_frames("scene", modality=Modality.POINT_CLOUD)[0],
                               "native", "local", np.ones((3, 3)))


@pytest.fixture
def installed(monkeypatch):
    entries = []
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    registry._installed_loaders.cache_clear()

    def entry_points(*, group):
        assert group == "visual_utils.loaders"
        return entries

    monkeypatch.setattr(registry.package_metadata, "entry_points", entry_points)
    yield entries
    registry._installed_loaders.cache_clear()


def test_listing_installed_names_never_imports_providers(installed):
    plugin = Entry(" Laboratory ", constructor=ImportError("optional dependency"))
    installed.append(plugin)
    assert "laboratory" in available_loaders()
    assert available_loaders() == tuple(sorted(available_loaders()))
    assert plugin.loads == 0
    with pytest.raises(DatasetConfigurationError, match="Cannot import dataset plugin.*laboratory") as failure:
        create_loader("LABORATORY", DatasetConfig())
    assert failure.value.__cause__ is plugin.constructor


def test_only_selected_provider_is_imported_and_receives_config(installed):
    selected = Entry("lab", constructor=PointProvider)
    unused = Entry("unrelated", constructor=ImportError("not installed"))
    installed.extend((selected, unused))
    config = DatasetConfig(options={"custom": [1, 2]})
    with create_loader("lab", config) as loader:
        assert loader.config is config
        assert loader.load_point_cloud("native-15").points.shape == (3, 3)
    assert (selected.loads, unused.loads) == (1, 0)


@pytest.mark.parametrize("names", [("lab", "LAB"), ("nuscenes",)])
def test_duplicate_names_and_builtin_collisions_are_explicit(installed, names):
    installed.extend(Entry(name) for name in names)
    with pytest.raises(DatasetConfigurationError, match="Multiple installed|conflict"):
        available_loaders()


def test_explicit_registration_cannot_shadow_installed_provider(installed):
    installed.append(Entry("lab"))
    with pytest.raises(DatasetConfigurationError, match="already registered"):
        register_loader("LAB", lambda config: None)


@pytest.mark.parametrize("constructor, message", [(42, "callable"), (lambda config: object(), "DatasetLoader")])
def test_invalid_plugin_exports_use_the_public_configuration_error(installed, constructor, message):
    installed.append(Entry("lab", constructor=constructor))
    with pytest.raises(DatasetConfigurationError, match=message):
        create_loader("lab", DatasetConfig())
