from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

COMPONENTS = ("center", "rotation", "internal", "atom_label")


@dataclass(frozen=True)
class BinnedMultipliers:
    """Fixed positive component multipliers in equal-width physical-time bins."""

    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.values) != 4 or not self.values[0]:
            raise ValueError("time multipliers require four nonempty component rows")
        if any(len(row) != len(self.values[0]) for row in self.values):
            raise ValueError("time multiplier rows must have equal length")
        if any(not math.isfinite(v) or v <= 0 for row in self.values for v in row):
            raise ValueError("time multipliers must be finite and positive")

    def evaluate(self, component: str, time: torch.Tensor) -> torch.Tensor:
        row = self.values[COMPONENTS.index(component)]
        index = (time.float() * len(row)).long().clamp(0, len(row) - 1)
        return time.new_tensor(row, dtype=torch.float32)[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "equal_width_physical_time",
            "bins": len(self.values[0]),
            "components": dict(zip(COMPONENTS, map(list, self.values))),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BinnedMultipliers":
        if value["kind"] != "equal_width_physical_time":
            raise ValueError("unsupported time multiplier kind")
        if set(value["components"]) != set(COMPONENTS):
            raise ValueError("time multipliers must specify all four components")
        result = cls(
            tuple(
                tuple(float(v) for v in value["components"][name])
                for name in COMPONENTS
            )
        )
        if int(value["bins"]) != len(result.values[0]):
            raise ValueError("time multiplier bin count mismatch")
        return result


@dataclass(frozen=True)
class LogChebyshevCurve:
    """A bounded positive weight exp(sum_k c_k T_k(2t-1))."""

    coefficients: tuple[float, ...]
    minimum: float = 0.001
    maximum: float = 8.0

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("a loss-weight curve needs at least one coefficient")
        if not 0.0 < self.minimum <= self.maximum:
            raise ValueError("loss-weight bounds must satisfy 0 < minimum <= maximum")

    def __call__(self, time: torch.Tensor) -> torch.Tensor:
        x = 2.0 * time.float() - 1.0
        first = torch.ones_like(x)
        log_weight = self.coefficients[0] * first
        if len(self.coefficients) > 1:
            second = x
            log_weight = log_weight + self.coefficients[1] * second
            for coefficient in self.coefficients[2:]:
                following = 2.0 * x * second - first
                log_weight = log_weight + coefficient * following
                first, second = second, following
        return torch.exp(log_weight.clamp(-20.0, 20.0)).clamp(
            self.minimum, self.maximum
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LogChebyshevCurve":
        return cls(
            coefficients=tuple(float(x) for x in value["coefficients"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
        )


@dataclass(frozen=True)
class TimeLossWeights:
    """Independent positive time weights for the separable structured loss."""

    center: LogChebyshevCurve
    rotation: LogChebyshevCurve
    internal: LogChebyshevCurve
    time_multipliers: BinnedMultipliers | None = None
    components = COMPONENTS

    def evaluate(
        self, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            getattr(self, name)(time) * self.multiplier(name, time)
            for name in COMPONENTS[:3]
        )

    def multiplier(self, component: str, time: torch.Tensor) -> torch.Tensor:
        if self.time_multipliers is None:
            return torch.ones_like(time)
        return self.time_multipliers.evaluate(component, time)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format_version": 1,
            "kind": "component_log_chebyshev",
            "components": {
                "center": self.center.to_dict(),
                "rotation": self.rotation.to_dict(),
                "internal": self.internal.to_dict(),
            },
        }
        if self.time_multipliers is not None:
            result["format_version"] = 2
            result["time_multipliers"] = self.time_multipliers.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TimeLossWeights":
        if value.get("kind") == "cartesian_brownian_endpoint":
            return CartesianLossWeights.from_dict(value)
        if int(value.get("format_version", -1)) not in (1, 2):
            raise ValueError("unsupported loss-weight format")
        if int(value["format_version"]) == 2 and "time_multipliers" not in value:
            raise ValueError("version 2 requires time multipliers")
        if value.get("kind") != "component_log_chebyshev":
            raise ValueError("unsupported loss-weight curve kind")
        components = value["components"]
        return cls(
            center=LogChebyshevCurve.from_dict(components["center"]),
            rotation=LogChebyshevCurve.from_dict(components["rotation"]),
            internal=LogChebyshevCurve.from_dict(components["internal"]),
            time_multipliers=(
                BinnedMultipliers.from_dict(value["time_multipliers"])
                if "time_multipliers" in value
                else None
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimeLossWeights":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CartesianLossWeights:
    """Standard Brownian endpoint MSE weights, with optional time-bin balancing.

    Matches Flowfusion scalefloss(BrownianMotion, t, pow=2, eps=.05).
    The label objective retains the reference model's separate base weight.
    """

    values: tuple[tuple[float, ...], ...] = ((1.0,) * 10, (1.0,) * 10)
    coordinate_epsilon: float = 0.05
    components = ("coordinate", "atom_label")

    def __post_init__(self):
        if (
            len(self.values) != 2
            or not self.values[0]
            or len(self.values[0]) != len(self.values[1])
            or any(not math.isfinite(v) or v <= 0 for row in self.values for v in row)
        ):
            raise ValueError(
                "Cartesian weights need two equal-length positive finite rows"
            )
        if not math.isfinite(self.coordinate_epsilon) or self.coordinate_epsilon <= 0:
            raise ValueError("coordinate epsilon must be positive and finite")

    def multiplier(self, component, time):
        row = self.values[self.components.index(component)]
        index = (time.float() * len(row)).long().clamp(0, len(row) - 1)
        return time.new_tensor(row, dtype=torch.float32)[index]

    def coordinate_weight(self, time):
        return (
            self.multiplier("coordinate", time)
            / (1 + self.coordinate_epsilon - time).square()
        )

    def to_dict(self):
        return dict(
            format_version=1,
            kind="cartesian_brownian_endpoint",
            coordinate_epsilon=self.coordinate_epsilon,
            bins=len(self.values[0]),
            components=dict(zip(self.components, map(list, self.values))),
        )

    @classmethod
    def from_dict(cls, value):
        if value.get("kind") != "cartesian_brownian_endpoint" or set(
            value["components"]
        ) != set(cls.components):
            raise ValueError("expected Cartesian coordinate and atom_label weights")
        result = cls(
            tuple(
                tuple(float(v) for v in value["components"][k]) for k in cls.components
            ),
            float(value["coordinate_epsilon"]),
        )
        if value["bins"] != len(result.values[0]):
            raise ValueError("Cartesian bin count mismatch")
        return result
