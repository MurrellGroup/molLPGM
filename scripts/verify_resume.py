"""Verify exact optimizer, raw model and EMA preservation across a stage boundary."""

import argparse
import json
from pathlib import Path
import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) == type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b))
        )
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("restored", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = [
        torch.load(x, map_location="cpu", weights_only=False)
        for x in (args.source, args.restored)
    ]
    checks = {
        k: equal(a[k], b[k])
        for k in ("step", "model_config", "model", "optimizer", "ema")
    }
    checks.update(
        {
            k: equal(a["training_state"][k], b["training_state"][k])
            for k in ("process", "schedule", "logit_offset_weight")
        }
    )
    args.output.write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("Resume state differs")


if __name__ == "__main__":
    main()
