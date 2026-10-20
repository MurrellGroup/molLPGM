from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import nn

_ADAMW_PREFIXES = (
    "atom_class_embedding.",
    "fragment_length_embedding.",
    "internal_head.",
    "center_head.",
    "rotation_head.",
    "coordinate_head.",
    "atom_label_head.",
    "time_embedding.",
)


def _use_adamw(name: str, parameter: nn.Parameter) -> bool:
    return (
        parameter.ndim != 2
        or name.startswith(_ADAMW_PREFIXES)
        or ".modulation." in name
        or name.endswith(".fragment_pair_bias")
    )


class MolecularMuon:
    """Muon on residual matrices, AdamW on embeddings/conditioning/I/O."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        learning_rate: float,
        adamw_lr_multiplier: float = 0.1,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        fused_adamw: bool = True,
    ) -> None:
        if not math.isfinite(adamw_lr_multiplier) or adamw_lr_multiplier <= 0:
            raise ValueError(
                "AdamW learning-rate multiplier must be finite and positive"
            )
        self.adamw_lr_multiplier = adamw_lr_multiplier
        muon_parameters: list[nn.Parameter] = []
        adamw_parameters: list[nn.Parameter] = []
        self.muon_parameter_names: list[str] = []
        self.adamw_parameter_names: list[str] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            if _use_adamw(name, parameter):
                adamw_parameters.append(parameter)
                self.adamw_parameter_names.append(name)
            else:
                muon_parameters.append(parameter)
                self.muon_parameter_names.append(name)
        if not muon_parameters or not adamw_parameters:
            raise ValueError("molecular Muon requires both optimizer groups")
        self.muon = torch.optim.Muon(
            muon_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="original",
        )
        self.adamw = torch.optim.AdamW(
            adamw_parameters,
            lr=self.adamw_lr_multiplier * learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=fused_adamw,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.muon.param_groups:
            group["lr"] = learning_rate
        for group in self.adamw.param_groups:
            group["lr"] = self.adamw_lr_multiplier * learning_rate

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "adamw_lr_multiplier": self.adamw_lr_multiplier,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("format_version") != 1:
            raise ValueError("unsupported molecular Muon optimizer state")
        if "adamw_lr_multiplier" not in state_dict:
            raise ValueError("optimizer state does not specify the AdamW multiplier")
        multiplier = float(state_dict["adamw_lr_multiplier"])
        if multiplier != self.adamw_lr_multiplier:
            raise ValueError(
                "AdamW learning-rate multiplier conflicts with optimizer checkpoint"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
