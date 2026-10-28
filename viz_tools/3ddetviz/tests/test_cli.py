"""The launch interface stays usable without importing a graphical toolkit."""

import builtins
import sys
from types import ModuleType

import pytest

from detviz import cli


@pytest.fixture
def launches(monkeypatch):
    calls = []
    module = ModuleType("detviz.ui.app")
    module.run_app = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "detviz.ui.app", module)
    return calls


def test_default_launch_waits_for_dataset_and_uses_validation_split(launches):
    assert cli.main([]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": None, "version": "v1.0-mini", "split": "val",
        "predictions": None, "cache_dir": None, "scene": None, "frame": 0, "autoload": False,
    }]


@pytest.mark.parametrize("root_option", ["--root", "--dataroot"])
def test_prediction_and_navigation_arguments_forwarded(root_option, launches, tmp_path):
    root, predictions, cache = tmp_path / "nuscenes", tmp_path / "results.json", tmp_path / "cache"
    assert cli.main([
        root_option, str(root), "--version", "v1.0-trainval", "--split", "val",
        "--predictions", str(predictions), "--cache-dir", str(cache),
        "--scene", "scene-0103", "--frame", "39",
    ]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": root, "version": "v1.0-trainval", "split": "val",
        "predictions": predictions, "cache_dir": cache, "scene": "scene-0103", "frame": 39, "autoload": True,
    }]


@pytest.mark.parametrize("arguments", [["--demo"], ["--dataset", "demo"]])
def test_demo_autoloads_without_a_dataset_root(arguments, launches):
    assert cli.main(arguments) == 0
    assert launches[0]["dataset"] == "demo"
    assert launches[0]["root"] is None
    assert launches[0]["split"] == "val"
    assert launches[0]["autoload"] is True


@pytest.mark.parametrize("arguments,expected", [
    (["--frame", "-1"], "must be zero or greater"),
    (["--frame", "1.5"], "must be an integer"),
    (["--dataset", "invalid"], "invalid choice"),
    (["--demo", "--predictions", "results.json"], "generates its own predictions"),
    (["--dataset", "demo", "--predictions", "results.json"], "generates its own predictions"),
    (["--dataset-config", "dataset.json", "--root", "/data"], "cannot be combined"),
    (["--dataset-config", "dataset.json", "--version", "v1.0-mini"], "cannot be combined"),
    (["--demo", "--dataset-config", "dataset.json"], "not demo"),
])
def test_invalid_arguments_fail_before_window_launch(arguments, expected, launches, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 2
    assert expected in capsys.readouterr().err
    assert launches == []


def test_dataset_defined_split_is_forwarded_without_nuscenes_assumptions(launches):
    assert cli.main(["--dataset", "waymo", "--split", "my-validation"]) == 0
    assert launches[0]["split"] == "my-validation"


def test_non_nuscenes_datasets_default_to_all_scenes(launches):
    assert cli.main(["--dataset", "waymo"]) == 0
    assert launches[0]["split"] == "all"


def test_explicit_prediction_format_is_forwarded(launches):
    assert cli.main(["--predictions", "predictions.data", "--prediction-format", "nuscenes_json"]) == 0
    assert launches[0]["prediction_format"] == "nuscenes_json"


def test_dataset_config_is_loaded_before_starting_the_window(launches, tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text('{"root": "data", "storage_format": "native", "release": "v1.0-mini"}')
    assert cli.main(["--dataset-config", str(path)]) == 0
    assert launches[0]["config"].root == tmp_path / "data"
    assert launches[0]["autoload"] is True


@pytest.mark.parametrize("option", ["--help", "--app-version"])
def test_help_and_version_do_not_import_graphical_dependencies(option, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("open3d") or name in {"ui.app", "detviz.ui.app", "threedviz.ui.app"}:
            raise AssertionError(f"CLI {option} imported graphical module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([option])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "3ddetviz" in output
    if option == "--help":
        assert "--predictions" in output and "--cache-dir" in output


@pytest.mark.parametrize("missing", ["open3d", "unrelated_dependency"])
def test_only_missing_open3d_gets_viewer_install_guidance(missing, monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name in {"ui.app", "detviz.ui.app"}:
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    if missing == "open3d":
        with pytest.raises(SystemExit) as exit_info:
            cli.main([])
        assert exit_info.value.code == 1
        output = capsys.readouterr().err
        assert "pip install" in output
        assert "./visual_utils" in output and "./viz_tools/3dviz" in output and "./viz_tools/3ddetviz" in output
    else:
        with pytest.raises(ModuleNotFoundError, match=missing):
            cli.main([])
