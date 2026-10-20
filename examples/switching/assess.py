"""Independent distribution, intermediate-marginal and integration checks."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate import compare
from model import MarginalModel, integrate
from process import Process, density, observations, paths, target


def cdf(x):
    t = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    result = torch.zeros_like(t)
    for w, m, s in zip(Process().weights, Process().means, Process().scales):
        result += w * 0.5 * (1 + torch.erf((t - m) / (s * 2**0.5)))
    return result.numpy()


def analytic_ks(x):
    x = np.sort(np.asarray(x, dtype=float))
    f = cdf(x)
    n = len(x)
    return float(max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n)))


def target_quantile(q):
    lo = np.full_like(q, -7.0)
    hi = np.full_like(q, 7.0)
    for _ in range(48):
        mid = (lo + hi) / 2
        lower = cdf(mid) < q
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("reference/model.pt"))
    ap.add_argument("--output", type=Path, default=Path("results"))
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=740261018)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MarginalModel().cuda().eval()
    model.load_state_dict(checkpoint["ema"])
    compiled = torch.compile(model, mode="reduce-overhead")
    rng = np.random.default_rng(args.seed)
    observed = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    grid = np.unique(np.concatenate((np.linspace(0, 1, args.steps + 1), observed)))
    indices = np.searchsorted(grid, observed)
    x0 = rng.standard_normal(args.samples).astype(np.float32)
    pieces = []
    started = time.perf_counter()
    for offset in range(0, args.samples, 8192):
        n = min(8192, args.samples - offset)
        start = np.zeros(8192, np.float32)
        start[:n] = x0[offset : offset + n]
        result = integrate(compiled, torch.from_numpy(start).cuda(), grid, indices)
        pieces.append(result[:, :n])
        print(
            json.dumps(
                dict(
                    event="sampled",
                    count=offset + n,
                    seconds=time.perf_counter() - started,
                )
            ),
            flush=True,
        )
    generated = np.concatenate(pieces, axis=1)
    reference = target(rng, args.samples)
    check = target(rng, args.samples)
    metrics = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_step=checkpoint["step"],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        sampling=dict(
            count=args.samples, integrator="RK4", steps=len(grid) - 1, seed=args.seed
        ),
        endpoint=compare(generated[-1], reference),
        target_sampling_reference=compare(check, reference),
        endpoint_analytic_ks=analytic_ks(generated[-1]),
        target_analytic_ks=analytic_ks(check),
        time_marginals=[],
        reference_noise=[],
        left_mass=dict(
            threshold=0.0,
            generated=float((generated[-1] < 0).mean()),
            target=float(cdf(np.array(0.0))),
        ),
    )
    for i, t in enumerate(observed):
        conditional, _ = observations(args.samples, rng, times=t)
        independent, _ = observations(args.samples, rng, times=t)
        metrics["time_marginals"].append(
            dict(time=float(t), **compare(generated[i], conditional[:, 0]))
        )
        metrics["reference_noise"].append(
            dict(time=float(t), **compare(independent[:, 0], conditional[:, 0]))
        )
    # Quantile sources remove Monte Carlo noise from the learned endpoint CDF check.
    q = (np.arange(8192) + 0.5) / 8192
    quantile_x0 = (
        (2**0.5 * torch.erfinv(torch.tensor(2 * q - 1, dtype=torch.float64)))
        .float()
        .cuda()
    )
    coarse = integrate(
        compiled, quantile_x0, np.linspace(0, 1, args.steps + 1), [args.steps]
    )[-1]
    fine = integrate(
        compiled, quantile_x0, np.linspace(0, 1, 2 * args.steps + 1), [2 * args.steps]
    )[-1]
    fine_cdf = cdf(fine)
    metrics["integration_convergence"] = dict(
        coarse_steps=args.steps,
        fine_steps=2 * args.steps,
        max_absolute_difference=float(abs(coarse - fine).max()),
        mean_absolute_difference=float(abs(coarse - fine).mean()),
        finite=bool(np.isfinite(generated).all() and np.isfinite(fine).all()),
        quantile_order_preserved=bool(np.all(np.diff(fine) > 0)),
        quantile_cdf_max_error=float(abs(fine_cdf - q).max()),
        quantile_wasserstein_1=float(abs(fine - target_quantile(q)).mean()),
    )
    # Independent sets of target draws quantify Monte Carlo fluctuations, not fit error.
    for _ in range(20):
        a, b = target(rng, args.samples), target(rng, args.samples)
        pair = compare(a, b)
        metrics["reference_noise"].append(
            dict(
                time=1.0,
                replicate=True,
                wasserstein_1=pair["wasserstein_1"],
                ks_distance=pair["ks_distance"],
            )
        )
    # Show every second Gaussian percentile; the histogram above uses random samples.
    from statistics import NormalDist

    figure_time = np.linspace(0, 1, 1025)
    conditional, signs, endpoints, vertices = paths(
        figure_time, 250, np.random.default_rng(831261016)
    )
    probabilities = np.arange(2, 100, 2, dtype=np.float64) / 100
    source = np.array(
        [NormalDist().inv_cdf(q) for q in probabilities], dtype=np.float32
    )
    display_start = np.zeros(8192, np.float32)
    display_start[: len(source)] = source
    learned = integrate(compiled, torch.from_numpy(display_start).cuda(), figure_time)[
        :, : len(source)
    ]
    support = np.linspace(-1.5, 2.5, 1600)
    offsets = np.cumsum([0] + [len(v) for v in vertices])
    np.savez_compressed(
        args.output / "figure-data.npz",
        checkpoint_sha256=np.array(metrics["checkpoint_sha256"]),
        time=figure_time,
        conditional=conditional,
        generated=learned,
        source_quantiles=probabilities,
        regimes=signs,
        target=reference,
        target_check=check,
        endpoint_samples=generated[-1],
        density_x=support,
        density_y=density(support),
        conditional_vertices=np.concatenate(vertices),
        conditional_offsets=offsets,
    )
    np.savez_compressed(
        args.output / "evaluation-data.npz",
        time=observed,
        generated=generated,
        target=reference,
        target_check=check,
        quantile_source=quantile_x0.cpu().numpy(),
        quantile_generated=fine,
        quantile_probabilities=q,
    )
    metrics["figure_checks"] = dict(
        exact_conditional_endpoints=bool(np.array_equal(conditional[-1], endpoints)),
        exact_source_quantiles=bool(np.array_equal(learned[0], source)),
        signs=list(map(int, np.unique(signs))),
        all_finite=bool(np.isfinite(learned).all()),
        mean_switches=float(np.mean([len(v) - 2 for v in vertices])),
    )
    metrics["acceptance"] = dict(
        endpoint_cdf_under_one_percent=metrics["integration_convergence"][
            "quantile_cdf_max_error"
        ]
        < 0.01,
        endpoint_w1_under_one_percent_sd=metrics["integration_convergence"][
            "quantile_wasserstein_1"
        ]
        < 0.0061,
        integration_max_under_1e_4=metrics["integration_convergence"][
            "max_absolute_difference"
        ]
        < 1e-4,
        intermediate_ks_under_two_percent=max(
            m["ks_distance"] for m in metrics["time_marginals"]
        )
        < 0.02,
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ["endpoint", "integration_convergence", "acceptance"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
