"""The standalone CLI remains usable without Tk or a desktop display."""

from __future__ import annotations

import builtins
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
        if name.startswith(("tkinter", "_tkinter", "open3d")) or name in {"ui.app", "cameraviz.ui.app"}:
            raise AssertionError(f"CLI information imported GUI dependency {name}")
        return original_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 0
    output = capsys.readouterr().out
    expected = "--dataroot" if args == ["--help"] else f"2dcameraviz {__version__}"
    assert expected in output


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
