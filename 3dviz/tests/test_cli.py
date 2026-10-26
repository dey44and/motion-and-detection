"""CLI behavior remains testable on machines without a graphical display."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType

import pytest

from threedviz import cli


@pytest.fixture
def launches(monkeypatch):
    calls = []
    app = ModuleType("threedviz.ui.app")
    app.run_app = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "threedviz.ui.app", app)
    return calls


def test_default_launch_waits_for_dataset_connection(launches):
    assert cli.main([]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": None, "version": "v1.0-mini",
        "scene": None, "frame": 0, "autoload": False,
    }]


@pytest.mark.parametrize("root_option", ["--dataroot", "--root"])
def test_nuscenes_arguments_are_forwarded(root_option, launches, tmp_path):
    assert cli.main([
        root_option, str(tmp_path), "--version", "v1.0-trainval",
        "--scene", "scene-0001", "--frame", "3",
    ]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": Path(tmp_path), "version": "v1.0-trainval",
        "scene": "scene-0001", "frame": 3, "autoload": True,
    }]


def test_waymo_and_explicit_split_are_forwarded(launches, tmp_path):
    assert cli.main(["--dataset", "waymo", "--root", str(tmp_path), "--split", "val", "--frame", "5"]) == 0
    assert launches == [{
        "dataset": "waymo", "root": tmp_path, "version": "v1.0-mini", "split": "val",
        "scene": None, "frame": 5, "autoload": True,
    }]


def test_explicit_all_split_is_forwarded_and_omitted_split_keeps_legacy_launch_contract(launches):
    assert cli.main(["--demo", "--split", "all"]) == 0
    assert launches[0]["split"] == "all"
    assert cli.main(["--demo"]) == 0
    assert "split" not in launches[1]


@pytest.mark.parametrize("demo_args", [["--demo"], ["--dataset", "demo"]])
def test_demo_autoloads_without_dataset_files(demo_args, launches):
    assert cli.main(demo_args) == 0
    assert launches[0]["dataset"] == "demo"
    assert launches[0]["root"] is None
    assert launches[0]["autoload"] is True


def test_negative_frame_is_rejected_before_launch(launches, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--frame", "-1"])
    assert exc.value.code == 2
    assert "must be zero or greater" in capsys.readouterr().err
    assert launches == []


def test_help_does_not_import_graphical_dependencies(monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("open3d", "tensorflow", "waymo_open_dataset")) or name in {"ui.app", "threedviz.ui.app"}:
            raise AssertionError(f"help imported graphical module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--dataroot" in output and "--split" in output and "waymo" in output


@pytest.mark.parametrize("missing", ["open3d", "unrelated_dependency"])
def test_only_missing_open3d_gets_install_guidance(missing, monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_gui_import(name, *args, **kwargs):
        if name in {"ui.app", "threedviz.ui.app"}:
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_gui_import)
    if missing == "open3d":
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code == 1
        assert "pip install" in capsys.readouterr().err
    else:
        with pytest.raises(ModuleNotFoundError, match=missing):
            cli.main([])
