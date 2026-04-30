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
from models.factory import _MODEL_BUILDERS
from models.injection import INJECTION_REGISTRY


def test_list_models_returns_available_models():
    names = list_models()
    assert "resnet18_cifar" in names
    assert "resnet34_cifar" in names
    assert "vgg16_cifar" in names
    assert "mobilenetv2_cifar" in names


def test_create_resnet18():
    model = create_model("resnet18_cifar", {"arch": "resnet18_cifar", "base_channels": 64})
    assert model is not None
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_resnet34():
    model = create_model("resnet34_cifar", {"arch": "resnet34_cifar", "base_channels": 64})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_vgg16():
    model = create_model("vgg16_cifar", {"arch": "vgg16_cifar", "base_channels": 64, "batch_norm": True})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_mobilenetv2():
    model = create_model("mobilenetv2_cifar",
                         {"arch": "mobilenetv2_cifar", "width_mult": 1.0})
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_create_unknown_model_raises():
    with pytest.raises(KeyError):
        create_model("nonexistent_model", {})


def test_injection_points_resnet18():
    model = create_model("resnet18_cifar", {"arch": "resnet18_cifar", "base_channels": 64})
    points = get_injection_points(model, "resnet18_cifar")
    assert len(points) == 8  # 2+2+2+2 blocks
    for name, module in points:
        assert name.startswith("layer")


def test_injection_points_resnet34():
    model = create_model("resnet34_cifar", {"arch": "resnet34_cifar", "base_channels": 64})
    points = get_injection_points(model, "resnet34_cifar")
    assert len(points) == 16  # 3+4+6+3 blocks


def test_injection_points_vgg16():
    model = create_model("vgg16_cifar", {"arch": "vgg16_cifar", "base_channels": 64, "batch_norm": True})
    points = get_injection_points(model, "vgg16_cifar")
    assert len(points) == 5  # 5 MaxPool2d stages


def test_injection_points_mobilenetv2():
    model = create_model("mobilenetv2_cifar",
                         {"arch": "mobilenetv2_cifar", "width_mult": 1.0})
    points = get_injection_points(model, "mobilenetv2_cifar")
    assert len(points) > 0
    # MobileNetV2 has 17 inverted residual blocks
    assert len(points) == 17


def test_injection_points_unknown_arch_raises():
    model = create_model("resnet18_cifar", {"arch": "resnet18_cifar", "base_channels": 64})
    with pytest.raises(KeyError):
        get_injection_points(model, "nonexistent_arch")


def test_all_models_have_injection_registered():
    MODEL_CFGS = {
        "resnet18_cifar": {"arch": "resnet18_cifar", "base_channels": 64},
        "resnet34_cifar": {"arch": "resnet34_cifar", "base_channels": 64},
        "vgg16_cifar": {"arch": "vgg16_cifar", "base_channels": 64, "batch_norm": True},
        "mobilenetv2_cifar": {"arch": "mobilenetv2_cifar", "width_mult": 1.0},
    }
    for name in list_models():
        model_cfg = MODEL_CFGS[name]
        model = create_model(name, model_cfg)
        points = get_injection_points(model, name)
        assert len(points) > 0, f"{name} has no injection points"
        for pt_name, pt_module in points:
            assert isinstance(pt_name, str), f"{name}: injection point name is not str"
            assert pt_module is not None, f"{name}: injection point module is None"
