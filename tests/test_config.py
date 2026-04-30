"""Tests for the unified configuration system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.utils import get_temperature_points, load_experiment_config


def test_config_loads_and_validates():
    config = load_experiment_config()
    assert config["system"]["seed"] == 2026
    assert config["train"]["epochs"] == 200
    assert config["temperature"]["range"] == [0, 110]
    assert config["temperature"]["step"] == 2
    assert "original" in config["nonideal"]
    assert "improved" in config["nonideal"]
    assert len(config["models"]["active"]) >= 2


def test_get_temperature_points():
    config = load_experiment_config()
    points = get_temperature_points(config)
    assert points[0] == 0.0
    assert points[-1] == 110.0
    assert len(points) == 56  # (110-0)/2 + 1 = 56
    # Check step is exactly 2
    for i in range(len(points) - 1):
        assert abs(points[i + 1] - points[i] - 2.0) < 1e-9


def test_bin_edges_cover_temperature_range():
    config = load_experiment_config()
    t_min, t_max = config["temperature"]["range"]
    for label, edges in config["nonideal"]["bins"].items():
        if label == "active_sweep":
            continue
        assert edges[0] <= t_min
        assert edges[-1] >= t_max


def test_active_sweep_bins_exist():
    config = load_experiment_config()
    active = config["nonideal"]["bins"]["active_sweep"]
    for label in active:
        assert label in config["nonideal"]["bins"]


def test_all_active_models_have_configs():
    config = load_experiment_config()
    for model_name in config["models"]["active"]:
        assert model_name in config["models"]
        assert "arch" in config["models"][model_name]


def test_config_rejects_bad_path():
    with pytest.raises(FileNotFoundError):
        load_experiment_config(Path("nonexistent.yaml"))
