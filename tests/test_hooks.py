from __future__ import annotations

import torch

from models import resnet18_cifar
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def test_resnet_forward_and_feature_return_shapes() -> None:
    model = resnet18_cifar()
    x = torch.randn(2, 3, 32, 32)
    logits, features = model(x, return_features=True)
    assert logits.shape == (2, 10)
    assert len(features) == 8


def test_hook_capture_and_replacement_work() -> None:
    model = resnet18_cifar()
    x = torch.randn(2, 3, 32, 32)
    blocks = list(model.iter_residual_blocks())

    capture_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    capture_hooks.attach()
    _ = model(x)
    captured = capture_hooks.last_features()
    capture_hooks.remove()
    assert len(captured) == 8

    nonideal = TemperatureNonIdeality(
        NonIdealConfig(mode="original", ka=0.0, kb=0.0, sigma0=0.1, lam=0.0),
        noise_seed=5,
    )
    replace_hooks = NonIdealHookManager(blocks, nonideality=nonideal, mode="replace_with_nonideal")
    replace_hooks.attach()
    replace_hooks.set_temperature(25.0)
    perturbed_logits = model(x)
    replaced = replace_hooks.last_features()
    replace_hooks.remove()

    assert perturbed_logits.shape == (2, 10)
    assert len(replaced) == 8
