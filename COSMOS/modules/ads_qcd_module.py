import numpy as np, streamlit as st, matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from utils.session import put_metric

def potential(z, kind, k, z0):
    eps = 1e-8
    if kind == "soft-wall":
        return (k**2)*(z**2) + 0.75/np.maximum(z**2, eps)
    elif kind == "hard-wall":
        V = 0.75/np.maximum(z**2, eps)
        V[z>z0] = 1e9
        return V
    else:  # soft + hard
        V = (k**2)*(z**2) + 0.75/np.maximum(z**2, eps)
        V[z>z0] = 1e9
        return V

def solve_fd(kind: str, k: float, z_max: float = 8.0, N: int = 400, n_eigs: int = 6, z0: float = 8.0):
    z_min = 1e-3
    z = np.linspace(z_min, z_max, N)
    dz = z[1]-z[0]
    V = potential(z, kind, k, z0)
    main = 2.0*np.ones(N-2)/(dz*dz) + V[1:-1]
    off = -1.0*np.ones(N-3)/(dz*dz)
    H = diags([off, main, off], [-1,0,1], format="csr")
    vals, vecs = eigsh(H, k=min(n_eigs, N-2), which='SA', tol=1e-6, maxiter=4000)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:,idx]
    psi = np.zeros((len(vals), N)); psi[:,1:-1] = vecs.T
    for i in range(len(vals)):
        nrm = np.sqrt(np.trapz(psi[i]**2, z)); 
        if nrm>0: psi[i] /= nrm
    return z, vals, psi, V

def wkb_levels(kind, k, z0, nmax=6, zmax=12.0, Nz=4000):
    z = np.linspace(1e-4, zmax, Nz)
    V = potential(z, kind, k, z0)
    levels = []
    for n in range(nmax):
        target = np.pi*(n+0.5)
        lo, hi = V.min()+1e-6, V.max()+k*50 + 100.0
        for _ in range(40):
            mid = 0.5*(lo+hi)
            rad = np.maximum(mid - V, 0.0)
            integral = np.trapz(np.sqrt(rad), z)
            if integral < target: lo = mid
            else: hi = mid
        levels.append(0.5*(lo+hi))
    return np.array(levels)

def orthonormality(psi, z):
    S = psi @ (psi * np.gradient(z)).T
    return S

def virial_check(z, psi, V):
    dpsi = np.gradient(psi, z, axis=1)
    T = 0.5*np.trapz((dpsi**2), z, axis=1)
    Vexp = np.trapz(psi**2 * V, z, axis=1)
    z_dV = np.gradient(V, z) * z
    vir = np.trapz(psi**2 * z_dV, z, axis=1)
    return T, Vexp, vir

def panel():
    st.markdown("#### AdS/QCD — WKB vs FD, eigenfunctions, potential, convergence")
    cols = st.columns(5)
    with cols[0]:
        kind = st.selectbox("Potential", ["soft-wall","hard-wall","soft+hard"], index=0)
    with cols[1]:
        k = st.slider("k (soft slope)", 0.05, 0.8, 0.25, 0.01)
    with cols[2]:
        zmax = st.slider("z_max", 4.0, 15.0, 8.0, 0.5)
    with cols[3]:
        z0 = st.slider("z0 (hard wall)", 3.0, 12.0, 8.0, 0.5)
    with cols[4]:
        N = st.slider("grid N", 200, 1400, 700, 50)
    ne = st.slider("eigenpairs", 3, 12, 6, 1)

    try:
        z, vals, psi, V = solve_fd(kind, k, zmax, N, ne, z0)
    except Exception as e:
        st.error(f"FD failed: {e}; trying smaller N")
        z, vals, psi, V = solve_fd(kind, k, zmax, max(400, N//2), min(ne,5), z0)

    wkb = wkb_levels(kind, k, z0, nmax=min(8, ne+2), zmax=max(zmax, 12.0))

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(np.arange(len(vals)), vals, 'o-', label="FD")
        ax.plot(np.arange(len(wkb)), wkb, 'x--', label="WKB")
        if kind == "soft-wall":
            ax.plot(np.arange(len(vals)), 4.0*k*(np.arange(len(vals))+1.0), 'g:', label="4k(n+1)")
        ax.set_xlabel("n"); ax.set_ylabel("m^2"); ax.legend(); ax.set_title("Spectrum: FD vs WKB vs analytic trend")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        for i in range(min(3, psi.shape[0])):
            ax2.plot(z, psi[i], label=f"ψ_{i}")
        ax2.set_xlabel("z"); ax2.set_ylabel("ψ(z)"); ax2.legend(); ax2.set_title("Eigenfunctions (first three)")
        st.pyplot(fig2)
    with c3:
        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.plot(z, V); ax3.set_xlabel("z"); ax3.set_ylabel("V(z)"); ax3.set_title("Potential profile")
        st.pyplot(fig3)

    # Diagnostics
    n = np.arange(len(vals))
    slope = np.polyfit(n+1, vals, 1)[0] if len(vals) >= 2 else np.nan
    rel_err = abs(slope - 4.0*k)/max(4.0*k, 1e-9) if kind=="soft-wall" else np.nan

    S = orthonormality(psi[:min(4, psi.shape[0])], z)
    offdiag = np.max(np.abs(S - np.diag(np.diag(S)))) if S.size else np.nan

    T, Vexp, vir = virial_check(z, psi[:min(4, psi.shape[0])], V)

    # Live metrics
    put_metric("ads_qcd", "slope_estimate", float(slope))
    put_metric("ads_qcd", "softwall_slope_rel_err", float(rel_err if not np.isnan(rel_err) else np.nan))
    put_metric("ads_qcd", "orthonormality_offdiag_max", float(offdiag))

    st.markdown("##### Convergence (ground state)")
    sizes = [250, 400, 600, 900, 1200]
    vals0 = []
    for gN in sizes:
        _, v, _, _ = solve_fd(kind, k, zmax, gN, 3, z0)
        vals0.append(float(v[0]))
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(sizes, vals0, 'o-'); ax4.set_xlabel("N grid"); ax4.set_ylabel("m0^2")
    ax4.set_title("Convergence of m0^2"); st.pyplot(fig4)

    with st.expander("Physics discussion (live: what each plot tells you)"):
        if kind=="soft-wall":
            st.markdown(f"""
**1) Spectrum**  
Linear \(m_n^2\sim 4k(n+1)\) is a holographic cartoon of **Regge trajectories**. Fitted slope **{slope:.3f}** vs target **{4*k:.3f}** → relative error **{rel_err:.2%}**.

**2) Eigenfunctions**  
Higher \(n\) has more nodes; boundary behavior near \(z\to 0\) controlled by the \(3/(4z^2)\) term. Shapes diagnose grid/BC issues.

**3) Potential**  
Confinement is encoded by growth at large \(z\) (soft wall k sets slope) or by a hard cutoff at \(z_0\).

**4) Convergence**  
\(m_0^2(N)\) should stabilize with finer grids; drift flags discretization error.

**What this says about the universe**  
Spectra arise from **operators + boundary conditions**. Semiclassical WKB already tracks the pattern—showing how much hadron phenomenology is captured by effective confining potentials.
""")
        else:
            st.markdown(f"""
**1) Spectrum**  
WKB vs FD mismatch quantifies boundary sensitivity from the hard wall at \(z_0={z0:.2f}\).

**2) Eigenfunctions**  
Node counts and near‑wall behavior reflect the imposed cutoff—useful for sanity checks.

**3) Potential**  
The sharp wall produces quantization by geometric constraint rather than smooth confinement.

**4) Convergence**  
As above; stability with N builds confidence in the numerics.
""")
