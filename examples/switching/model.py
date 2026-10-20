"""Small position-and-time-only marginal predictor."""

import torch
from torch import nn


class MarginalModel(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        # Only the current coordinate and time enter the learned velocity.
        return self.net(xt).squeeze(-1)

    def endpoint_prediction(self, xt):
        return xt[:, 0] + (1 - xt[:, 1]) * self(xt)


@torch.no_grad()
def integrate(model, x0, time, save_times=None):
    """RK4 on the learned marginal ODE. No endpoint correction or latent process."""
    x = x0.clone()
    recorded = [x.cpu()] if save_times is None else []
    selected = set(range(len(time))) if save_times is None else set(save_times)
    if save_times is not None and 0 in selected:
        recorded.append(x.cpu())

    def f(x, t):
        # RK stages must survive the next CUDA-graph invocation.
        return model(torch.stack((x, torch.full_like(x, float(t))), dim=-1)).clone()

    for i, (t, s) in enumerate(zip(time[:-1], time[1:])):
        h = float(s - t)
        k1 = f(x, t)
        k2 = f(x + h * k1 / 2, (t + s) / 2)
        k3 = f(x + h * k2 / 2, (t + s) / 2)
        k4 = f(x + h * k3, s)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if i + 1 in selected:
            recorded.append(x.cpu())
    return torch.stack(recorded).numpy()
