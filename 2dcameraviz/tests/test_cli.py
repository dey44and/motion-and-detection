"""The standalone CLI remains usable without Tk or a desktop display."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from cameraviz import __version__, cli


@pytest.fixture
def launches(monkeypatch):
    calls = []
    app = ModuleType("cameraviz.ui.app")

    def run_app(**kwargs):
        calls.append(kwargs)
        return 0

    app.run_app = run_app
    monkeypatch.setitem(sys.modules, "cameraviz.ui.app", app)
    return calls


def test_default_launch_waits_for_dataset_connection(launches):
    assert cli.main([]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": None, "version": "v1.0-mini",
        "scene": None, "frame": 0, "autoload": False,
    }]


@pytest.mark.parametrize("root_option", ["--dataroot", "--root"])
def test_dataset_arguments_are_forwarded(root_option, launches, tmp_path):
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


def test_explicit_all_split_forwarded_without_changing_default_call_contract(launches):
    assert cli.main(["--demo", "--split", "all"]) == 0
    assert launches[0]["split"] == "all"
    assert cli.main(["--demo"]) == 0
    assert "split" not in launches[1]


@pytest.mark.parametrize("demo_args", [["--demo"], ["--dataset", "demo"]])
def test_demo_autoloads_without_files(demo_args, launches):
    assert cli.main(demo_args) == 0
    assert launches[0]["dataset"] == "demo"
    assert launches[0]["root"] is None
    assert launches[0]["autoload"] is True


@pytest.mark.parametrize("frame, message", [("-1", "must be zero or greater"), ("1.5", "must be an integer")])
def test_invalid_frame_is_rejected_before_launch(frame, message, launches, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--frame", frame])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert launches == []


@pytest.mark.parametrize("args", [["--help"], ["--app-version"]])
def test_help_and_version_do_not_import_gui(args, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *positional, **kwargs):
        if name.startswith(("tkinter", "_tkinter", "open3d", "tensorflow", "waymo_open_dataset")) or name in {"ui.app", "cameraviz.ui.app"}:
            raise AssertionError(f"CLI information imported GUI dependency {name}")
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    expected = "--dataroot" if args == ["--help"] else f"2dcameraviz {__version__}"
    assert expected in output
    if args == ["--help"]:
        assert "--split" in output and "waymo" in output


@pytest.mark.parametrize("missing", ["tkinter", "_tkinter", "unrelated_dependency"])
def test_missing_tk_gets_install_guidance(missing, monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_gui_import(name, *positional, **kwargs):
        if name in {"ui.app", "cameraviz.ui.app"}:
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_gui_import)
    if missing in {"tkinter", "_tkinter"}:
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code == 1
        assert "python3-tk" in capsys.readouterr().err
    else:
        with pytest.raises(ModuleNotFoundError, match=missing):
            cli.main([])


def test_missing_display_gets_desktop_guidance(monkeypatch, capsys):
    app = ModuleType("cameraviz.ui.app")
    TclError = type("TclError", (Exception,), {"__module__": "_tkinter"})

    def unavailable_display(**kwargs):
        raise TclError("no display name and no $DISPLAY environment variable")

    app.run_app = unavailable_display
    monkeypatch.setitem(sys.modules, "cameraviz.ui.app", app)
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 1
    assert "graphical desktop" in capsys.readouterr().err


def test_other_gui_failures_are_not_hidden(monkeypatch):
    app = ModuleType("cameraviz.ui.app")

    def broken_app(**kwargs):
        raise RuntimeError("an application bug")

    app.run_app = broken_app
    monkeypatch.setitem(sys.modules, "cameraviz.ui.app", app)
    with pytest.raises(RuntimeError, match="an application bug"):
        cli.main([])


def test_application_exit_status_is_returned(monkeypatch):
    app = ModuleType("cameraviz.ui.app")
    app.run_app = lambda **kwargs: 7
    monkeypatch.setitem(sys.modules, "cameraviz.ui.app", app)
    assert cli.main(["--demo"]) == 7


def test_dataset_configuration_is_resolved_and_forwarded(launches, tmp_path):
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps({
        "root": "data", "release": "custom-v1", "source_paths": {"native": "records"},
    }), encoding="utf-8")
    assert cli.main([
        "--dataset", "waymo", "--dataset-config", str(config_path), "--split", "val",
    ]) == 0
    call = launches[0]
    assert call["autoload"] is True
    assert call["dataset"] == "waymo"
    assert call["split"] == "val"
    assert call["config"].root == tmp_path / "data"
    assert call["config"].release == "custom-v1"
    assert call["config"].source_paths["native"] == tmp_path / "data" / "records"


@pytest.mark.parametrize("conflicting", [
    ["--root", "/data"], ["--dataroot", "/data"],
    ["--version", "v1.0-mini"], ["--demo"], ["--dataset", "demo"],
])
def test_ambiguous_dataset_configuration_is_rejected(conflicting, launches, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--dataset-config", "dataset.json", *conflicting])
    assert exc.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
    assert launches == []


def test_invalid_dataset_configuration_reports_cli_error(launches, tmp_path, capsys):
    path = tmp_path / "dataset.json"
    path.write_text('{"unknown_setting": true}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--dataset-config", str(path)])
    assert exc.value.code == 2
    assert "unknown_setting" in capsys.readouterr().err
    assert launches == []
