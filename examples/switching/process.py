"""The one-dimensional switching process, simulated at exact jump times.

Between jumps X follows dX/dt = (Z*y-X)/(1-t). There is no Brownian noise.
"""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Process:
    source_sd: float = 1.0
    leave_rate: float = 6.0
    return_rate: float = 6.0
    weights: tuple = (0.2, 0.4, 0.4)
    means: tuple = (-0.5, 0.9, 0.5)
    scales: tuple = (0.1, 0.1, 0.5)

    def dictionary(self):
        return asdict(self)


def target(rng, n, p=Process()):
    k = rng.choice(len(p.weights), n, p=p.weights)
    return np.asarray(p.means)[k] + np.asarray(p.scales)[k] * rng.standard_normal(n)


def density(x, p=Process()):
    x = np.asarray(x)
    return sum(
        w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        for w, m, s in zip(p.weights, p.means, p.scales)
    )


def next_jump(t, z, exponential, p=Process()):
    """Invert the integrated hazards 6(1-t) and 6/(1-t). A value 1 means no jump."""
    r = 1 - t
    positive = 1 - np.sqrt(np.maximum(0, r * r - 2 * exponential / p.leave_rate))
    negative = 1 - r * np.exp(-exponential / p.return_rate)
    return np.where(z > 0, positive, negative)


def advance(x, y, z, t, s):
    return z * y + (x - z * y) * ((1 - s) / (1 - t))


def observations(n, rng, p=Process(), times=None):
    """Fresh independent observations (X_t, t, Z_t*X_1), with no integration grid."""
    t = rng.random(n) if times is None else np.broadcast_to(times, (n,)).copy()
    x = p.source_sd * rng.standard_normal(n)
    y = target(rng, n, p)
    z = rng.choice([-1, 1], n).astype(np.int8)
    start = np.zeros(n)
    alive = np.arange(n)
    jumps = np.zeros(n, dtype=np.int16)
    while len(alive):
        j = next_jump(start[alive], z[alive], rng.exponential(size=len(alive)), p)
        end = np.minimum(j, t[alive])
        x[alive] = advance(x[alive], y[alive], z[alive], start[alive], end)
        switched = j < t[alive]
        active = alive[switched]
        z[active] *= -1
        start[active] = j[switched]
        jumps[active] += 1
        alive = active
    return np.column_stack((x, t, z * y)).astype(np.float32), jumps


def paths(time, n, rng, p=Process()):
    """Record exact continuous paths, their right-continuous sign, and all jump vertices."""
    time = np.asarray(time)
    x0 = p.source_sd * rng.standard_normal(n)
    y = target(rng, n, p)
    initial_sign = rng.choice([-1, 1], n)
    values, signs = np.empty((len(time), n)), np.empty((len(time), n), np.int8)
    vertices = []
    for i in range(n):
        t, x, z = 0.0, x0[i], int(initial_sign[i])
        ts, xs, zs = [t], [x], [z]
        while True:
            s = float(next_jump(t, z, rng.exponential(), p))
            if s >= 1:
                break
            x = float(advance(x, y[i], z, t, s))
            t, z = s, -z
            ts.append(t)
            xs.append(x)
            zs.append(z)
        if z != 1:
            raise RuntimeError(
                "Terminal sign must be positive; numerical underflow in jump time"
            )
        ts.append(1.0)
        xs.append(float(y[i]))
        zs.append(1)
        ts, xs, zs = np.array(ts), np.array(xs), np.array(zs)
        ix = np.clip(np.searchsorted(ts, time, side="right") - 1, 0, len(ts) - 2)
        values[:, i] = advance(xs[ix], y[i], zs[ix], ts[ix], time)
        signs[:, i] = zs[ix]
        vertices.append(np.column_stack((ts, xs, zs)))
    return values, signs, y, vertices
