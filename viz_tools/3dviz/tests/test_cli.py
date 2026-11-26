"""The browser CLI forwards dataset and detection launch settings."""

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
    server = ModuleType("threedviz.web.server")
    server.run_server = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "threedviz.web.server", server)
    return calls


def test_default_launch_waits_for_dataset_connection(launches):
    assert cli.main([]) == 0
    assert launches == [{
        "dataset": "nuscenes", "root": None, "version": "v1.0-mini",
        "scene": None, "frame": 0, "autoload": False,
        "port": 8080, "open_browser": True, "predictions": None,
        "prediction_format": "auto", "cache_dir": None,
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
        "port": 8080, "open_browser": True, "predictions": None,
        "prediction_format": "auto", "cache_dir": None,
    }]


def test_waymo_and_explicit_split_are_forwarded(launches, tmp_path):
    assert cli.main(["--dataset", "waymo", "--root", str(tmp_path), "--split", "val", "--frame", "5"]) == 0
    assert launches == [{
        "dataset": "waymo", "root": tmp_path, "version": "v1.0-mini", "split": "val",
        "scene": None, "frame": 5, "autoload": True,
        "port": 8080, "open_browser": True, "predictions": None,
        "prediction_format": "auto", "cache_dir": None,
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
        if name.startswith(("open3d", "tensorflow", "waymo_open_dataset", "fastapi", "uvicorn")) or name in {"ui.app", "threedviz.ui.app"}:
            raise AssertionError(f"help imported graphical module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--dataroot" in output and "--split" in output and "waymo" in output


def test_dataset_config_is_resolved_and_forwarded_before_server_launch(launches, tmp_path):
    config_file = tmp_path / "sources.json"
    config_file.write_text('{"root": "dataset", "storage_format": "native"}', encoding="utf-8")
    assert cli.main(["--dataset", "waymo", "--dataset-config", str(config_file), "--split", "val"]) == 0
    options = launches[0]
    assert options["config"].root == tmp_path / "dataset"
    assert options["config"].storage_format == "native"
    assert options["root"] is None and options["split"] == "val" and options["autoload"]


@pytest.mark.parametrize("conflicting", [["--root", "/data"], ["--version", "v1.0-mini"], ["--demo"]])
def test_dataset_config_rejects_ambiguous_or_demo_launches(conflicting, launches, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--dataset-config", str(tmp_path / "sources.json"), *conflicting])
    assert error.value.code == 2 and "--dataset-config" in capsys.readouterr().err
    assert not launches


def test_browser_options_work_on_the_default_command(launches):
    assert cli.main(["--demo", "--port", "8097", "--open-browser", "--split", "val"]) == 0
    assert launches == [{
        "dataset": "demo", "root": None, "version": "v1.0-mini",
        "scene": None, "frame": 0, "autoload": True, "split": "val",
        "port": 8097, "open_browser": True, "predictions": None,
        "prediction_format": "auto", "cache_dir": None,
    }]


@pytest.mark.parametrize("args", [["--web"], ["--no-open-browser"]])
def test_browser_can_run_without_opening_a_tab(args, launches):
    assert cli.main(args) == 0
    assert not launches[0]["open_browser"]


def test_prediction_launch_options_are_forwarded(launches, tmp_path):
    export = tmp_path / "result.pkl"
    cache = tmp_path / "indexes"
    assert cli.main([
        "--root", str(tmp_path), "--predictions", str(export),
        "--prediction-format", "pcdet_pickle", "--cache-dir", str(cache),
    ]) == 0
    assert launches[0]["predictions"] == export
    assert launches[0]["prediction_format"] == "pcdet_pickle"
    assert launches[0]["cache_dir"] == cache
    assert launches[0]["split"] == "val" and launches[0]["autoload"]


def test_unknown_prediction_format_is_a_launch_error(launches, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--prediction-format", "unknown"])
    assert error.value.code == 2 and "Choose from auto" in capsys.readouterr().err
    assert not launches


@pytest.mark.parametrize("port", ["0", "65536", "-1", "bad"])
def test_invalid_browser_ports_are_rejected(port, launches, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--port", port])
    assert error.value.code == 2 and "between 1 and 65535" in capsys.readouterr().err
    assert not launches


@pytest.mark.parametrize("missing", ["fastapi", "uvicorn", "unrelated_dependency"])
def test_missing_browser_dependencies_have_specific_guidance(missing, monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_web_import(name, *args, **kwargs):
        if name in {"web.server", "threedviz.web.server"}:
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_web_import)
    if missing == "unrelated_dependency":
        with pytest.raises(ModuleNotFoundError, match=missing):
            cli.main([])
    else:
        with pytest.raises(SystemExit) as error:
            cli.main([])
        assert error.value.code == 1 and "pip install" in capsys.readouterr().err


def test_invalid_dataset_config_reports_cli_error_before_server_import(launches, tmp_path, capsys):
    config_file = tmp_path / "sources.json"
    config_file.write_text("{invalid}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--dataset-config", str(config_file)])
    assert error.value.code == 2 and "error:" in capsys.readouterr().err
    assert not launches
