"""Portable configuration files keep source paths independent of launch cwd."""

import json

import pytest

from visual_utils import DatasetConfigurationError, load_dataset_config


def test_paths_are_anchored_to_config_file_and_root(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "dataset.json"
    path.write_text(json.dumps({
        "root": "../datasets", "release": "custom-v2", "storage_format": "native",
        "source_paths": {"native": "raw", "annotations": str(tmp_path / "labels")},
        "options": {"sensors": ["one", "two"]},
    }))
    monkeypatch.chdir(tmp_path.parent)
    config = load_dataset_config(path)
    assert config.root == tmp_path / "datasets"
    assert config.source_paths["native"] == tmp_path / "datasets/raw"
    assert config.source_paths["annotations"] == tmp_path / "labels"
    assert config.options["sensors"] == ("one", "two")
    assert config.release == "custom-v2" and config.storage_format == "native"


def test_relative_sources_without_root_use_config_directory(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text('{"source_paths": {"native": "raw"}}')
    config = load_dataset_config(path)
    assert config.root is None
    assert config.source_paths["native"] == tmp_path / "raw"


@pytest.mark.parametrize("content", [
    "{", "[]", '{"unknown": 1}', '{"options": {"value": NaN}}',
    '{"root": "one", "root": "two"}', '{"options": {"x": 1, "x": 2}}',
    '{"source_paths": []}', '{"source_paths": {"native": null}}',
    '{"root": "s3://unsupported"}', '{"release": 42}',
])
def test_invalid_json_and_unknown_or_ambiguous_values_are_rejected(tmp_path, content):
    path = tmp_path / "dataset.json"
    path.write_text(content)
    with pytest.raises(DatasetConfigurationError):
        load_dataset_config(path)


def test_missing_and_non_utf8_files_retain_the_cause(tmp_path):
    path = tmp_path / "dataset.json"
    with pytest.raises(DatasetConfigurationError, match="dataset.json") as failure:
        load_dataset_config(path)
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    path.write_bytes(b"\xff")
    with pytest.raises(DatasetConfigurationError) as failure:
        load_dataset_config(path)
    assert isinstance(failure.value.__cause__, UnicodeDecodeError)
