
from __future__ import annotations
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def softwall_potential(z: np.ndarray, k: float) -> np.ndarray:
    eps = 1e-8
    return (k**2)*(z**2) + 0.75/np.maximum(z**2, eps)

def solve_softwall(k: float, z_max: float = 8.0, N: int = 600, n_eigs: int = 6):
    z_min = 1e-3
    z = np.linspace(z_min, z_max, N).astype(np.float64)
    dz = z[1] - z[0]
    V = softwall_potential(z, k).astype(np.float64)

    main = 2.0*np.ones(N-2) / (dz*dz) + V[1:-1]
    off  = -1.0*np.ones(N-3) / (dz*dz)
    H = diags([off, main, off], offsets=[-1,0,1], format="csr")

    vals, vecs = eigsh(H, k=min(n_eigs, N-2), which='SA', tol=1e-6, maxiter=2000)
    psi = np.zeros((vecs.shape[1], N))
    psi[:,1:-1] = vecs.T
    for i in range(psi.shape[0]):
        norm = np.sqrt(np.trapz(psi[i]**2, z))
        if norm>0: psi[i] /= norm
    return z, vals, psi
