
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Sample:
    X: np.ndarray
    y_reg: np.ndarray
    y_cls: np.ndarray

def make_synth(n=3000, ambient_max=5, codim_max=3, seed=42) -> Sample:
    rs = np.random.default_rng(seed)
    ambient_dims = rs.integers(2, 6, size=(n, ambient_max))
    codim = rs.integers(1, codim_max+1, size=(n,1))
    fsum = ambient_dims.sum(axis=1, keepdims=True)
    fvar = ambient_dims.var(axis=1, keepdims=True)
    fcod = codim.astype(float)
    X = np.concatenate([fsum, fvar, fcod], axis=1).astype(np.float32)

    h11 = (0.6*fsum[:,0] + 3.0*fcod[:,0] + 2.0*np.sqrt(fvar[:,0]) + rs.normal(0,1.0,n)).clip(1,35)
    h21 = (35 - 0.5*fsum[:,0] + 1.5*np.sqrt(fvar[:,0]) + rs.normal(0,1.5,n)).clip(1,50)
    chi  = 2*(h11 - h21)
    y_reg = np.stack([h11, h21, chi], axis=1).astype(np.float32)
    thr = float(np.median(chi))
    y_cls = (chi > thr).astype(np.int64)
    return Sample(X, y_reg, y_cls)
