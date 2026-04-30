"""Temperature-dependent non-ideality model.

The model is intentionally behavioral rather than transistor-level. It maps
deployment non-idealities to feature perturbations after residual blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from .compensation import CompensationSpec


@dataclass(frozen=True)
class NonIdealConfig:
    """Configuration for the behavioral non-ideality model."""

    mode: str
    ka: float
    kb: float
    sigma0: float
    lam: float
    rho: float = 1.0
    bin_edges: Optional[Sequence[float]] = None
    nominal_gain: float = 1.0
    nominal_bias: float = 0.0
    compensation_strength: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NonIdealConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


class TemperatureNonIdeality(nn.Module):
    """Apply gain drift, bias drift, and random noise to intermediate features."""

    def __init__(self, config: NonIdealConfig, noise_seed: int = 0) -> None:
        super().__init__()
        self.config = config
        self._generator = torch.Generator(device="cpu")
        self._noise_seed = int(noise_seed)
        self.reset_noise_seed(self._noise_seed)
        self._compensation = None
        if config.mode == "improved":
            if config.bin_edges is None:
                raise ValueError("bin_edges are required for improved mode.")
            self._compensation = CompensationSpec.from_sequence(config.bin_edges)

    def reset_noise_seed(self, seed: int) -> None:
        """Reset the noise generator so modes can share the same noise path."""
        self._noise_seed = int(seed)
        self._generator.manual_seed(self._noise_seed)

    @property
    def noise_seed(self) -> int:
        return self._noise_seed

    def sigma(self, temperature: float) -> float:
        """Return the temperature-dependent noise strength."""
        return self.config.sigma0 * (1.0 + self.config.lam * max(temperature - 25.0, 0.0))

    def gain(self, temperature: float, reference_temperature: float = 25.0) -> float:
        drift_gain = 1.0 + self.config.ka * (temperature - reference_temperature)
        return self.config.nominal_gain * drift_gain

    def bias(self, temperature: float, reference_temperature: float = 25.0) -> float:
        return self.config.nominal_bias + self.config.kb * (temperature - reference_temperature)

    def _sample_base_noise(self, x: Tensor) -> Tensor:
        """Sample CPU noise first for deterministic cross-device comparisons."""
        base = torch.randn(
            x.shape,
            generator=self._generator,
            device="cpu",
            dtype=torch.float32,
        )
        return base.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor, temperature: float) -> Tensor:
        if self.config.mode == "ideal":
            return x

        reference_temperature = 25.0
        sigma_scale = 1.0
        if self.config.mode == "improved":
            assert self._compensation is not None
            bin_center = self._compensation.get_bin_center(temperature)
            # Compensation is finite: move partway from nominal 25 C toward the
            # selected bin center instead of cancelling all temperature drift.
            reference_temperature = 25.0 + self.config.compensation_strength * (bin_center - 25.0)
            sigma_scale = self.config.rho

        gain = self.gain(temperature, reference_temperature)
        bias = self.bias(temperature, reference_temperature)
        sigma = self.sigma(temperature) * sigma_scale
        noise = self._sample_base_noise(x) * sigma
        return gain * x + bias + noise
