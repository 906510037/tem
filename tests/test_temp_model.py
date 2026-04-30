from __future__ import annotations

import torch

from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def test_original_mode_without_noise_matches_gain_and_bias() -> None:
    module = TemperatureNonIdeality(
        NonIdealConfig(mode="original", ka=0.1, kb=0.2, sigma0=0.0, lam=0.0),
        noise_seed=7,
    )
    x = torch.ones(2, 3)
    out = module(x, temperature=35.0)
    expected = (1.0 + 0.1 * 10.0) * x + 0.2 * 10.0
    assert torch.allclose(out, expected)


def test_improved_mode_uses_bin_center() -> None:
    module = TemperatureNonIdeality(
        NonIdealConfig(
            mode="improved",
            ka=0.1,
            kb=0.2,
            sigma0=0.0,
            lam=0.0,
            rho=0.5,
            bin_edges=[0, 25, 50, 75, 100],
        ),
        noise_seed=7,
    )
    x = torch.ones(1, 1)
    out = module(x, temperature=40.0)
    expected = (1.0 + 0.1 * (40.0 - 37.5)) * x + 0.2 * (40.0 - 37.5)
    assert torch.allclose(out, expected)


def test_noise_seed_reset_makes_sampling_repeatable() -> None:
    module = TemperatureNonIdeality(
        NonIdealConfig(mode="original", ka=0.0, kb=0.0, sigma0=0.1, lam=0.0),
        noise_seed=11,
    )
    x = torch.zeros(4, 4)
    first = module(x, temperature=25.0)
    module.reset_noise_seed(11)
    second = module(x, temperature=25.0)
    assert torch.allclose(first, second)
