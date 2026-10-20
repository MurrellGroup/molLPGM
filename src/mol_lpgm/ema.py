"""Constant-decay, float32 model EMA, initialized from the starting weights."""

from __future__ import annotations

import math
import torch


class ModelEMA:
    def __init__(
        self, model: torch.nn.Module, decay: float, *, initialized_step: int = 0
    ):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = float(decay)
        self.initialized_step = int(initialized_step)
        self.num_updates = 0
        self.parameter_names = set(dict(model.named_parameters()))
        self.shadow = {
            name: (
                value.detach().float().clone()
                if value.is_floating_point()
                else value.detach().clone()
            )
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        destinations, sources = [], []
        for name, value in model.state_dict().items():
            if name in self.parameter_names:
                destinations.append(self.shadow[name])
                sources.append(value.detach().to(self.shadow[name].dtype))
            else:
                # Buffers (e.g. fixed Fourier frequencies) are copied, not smoothed.
                self.shadow[name].copy_(value)
        torch._foreach_lerp_(destinations, sources, 1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return dict(
            decay=self.decay,
            initialized_step=self.initialized_step,
            num_updates=self.num_updates,
            model=self.shadow,
        )

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay conflicts with checkpoint")
        if state["model"].keys() != self.shadow.keys():
            raise ValueError("EMA state does not match the model")
        for name, value in state["model"].items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA shape mismatch for {name}")
            self.shadow[name].copy_(value)
        self.num_updates = int(state["num_updates"])
        self.initialized_step = int(state["initialized_step"])
