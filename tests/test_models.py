"""Tests for the model registry, factory, and injection points."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import create_model, get_injection_points, list_models


def test_list_models_returns_available_models():
    names = list_models()
    assert "resnet18_cifar" in names
    assert "resnet34_cifar" in names
    assert "resnet50_cifar" in names


def test_create_resnet18():
    model = create_model("resnet18_cifar",
                         {"arch": "resnet18_cifar", "base_channels": 64})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_resnet34():
    model = create_model("resnet34_cifar",
                         {"arch": "resnet34_cifar", "base_channels": 64})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_resnet50():
    model = create_model("resnet50_cifar",
                         {"arch": "resnet50_cifar", "base_channels": 64})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_unknown_model_raises():
    with pytest.raises(KeyError):
        create_model("nonexistent_model", {})


def test_injection_points_resnet18():
    model = create_model("resnet18_cifar",
                         {"arch": "resnet18_cifar", "base_channels": 64})
    points = get_injection_points(model, "resnet18_cifar")
    assert len(points) == 8  # (2+2+2+2) BasicBlock


def test_injection_points_resnet34():
    model = create_model("resnet34_cifar",
                         {"arch": "resnet34_cifar", "base_channels": 64})
    points = get_injection_points(model, "resnet34_cifar")
    assert len(points) == 16  # (3+4+6+3) BasicBlock


def test_injection_points_resnet50():
    model = create_model("resnet50_cifar",
                         {"arch": "resnet50_cifar", "base_channels": 64})
    points = get_injection_points(model, "resnet50_cifar")
    assert len(points) == 16  # (3+4+6+3) Bottleneck


def test_injection_points_unknown_arch_raises():
    model = create_model("resnet18_cifar",
                         {"arch": "resnet18_cifar", "base_channels": 64})
    with pytest.raises(KeyError):
        get_injection_points(model, "nonexistent_arch")


def test_all_models_have_injection_registered():
    MODEL_CFGS = {
        "resnet18_cifar": {"arch": "resnet18_cifar", "base_channels": 64},
        "resnet34_cifar": {"arch": "resnet34_cifar", "base_channels": 64},
        "resnet50_cifar": {"arch": "resnet50_cifar", "base_channels": 64},
    }
    for name in list_models():
        model_cfg = MODEL_CFGS[name]
        model = create_model(name, model_cfg)
        points = get_injection_points(model, name)
        assert len(points) > 0, f"{name} has no injection points"
        for pt_name, pt_module in points:
            assert isinstance(pt_name, str)
            assert pt_module is not None
