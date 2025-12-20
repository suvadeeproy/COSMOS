
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class TransitParams:
    depth: float
    duration: int
    period: int
    phase: int

def _inject_box(flux: np.ndarray, p: TransitParams) -> np.ndarray:
    f = flux.copy()
    for t in range(p.phase % p.period, len(f), p.period):
        f[t:min(len(f), t+p.duration)] *= (1.0 - p.depth)
    return f

def make_synth(n: int=2000, L: int=512, sigma: float=8e-4, prob: float=0.5, seed: int=42):
    rs = np.random.default_rng(seed)
    X, y = [], []
    t = np.arange(L, dtype=np.float32)
    for _ in range(n):
        base = 1.0 + 1e-3*np.sin(2*np.pi*t/rs.integers(200, 600))
        base += rs.normal(0.0, sigma, size=L)
        if rs.random() < prob:
            p = TransitParams(
                depth=float(rs.uniform(5e-4, 1e-2)),
                duration=int(rs.integers(10, 40)),
                period=int(rs.integers(80, 200)),
                phase=int(rs.integers(0, 200)),
            )
            s = _inject_box(base, p); y.append(1)
        else:
            s = base; y.append(0)
        X.append((s/np.median(s)).astype(np.float32))
    return np.stack(X), np.array(y, dtype=np.int64)
