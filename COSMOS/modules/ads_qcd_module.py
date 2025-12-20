import numpy as np, streamlit as st, matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from utils.telemetry import log_event

def potential(z, kind, k, z0):
    eps = 1e-8
    if kind == "soft-wall":
        return (k**2)*(z**2) + 0.75/np.maximum(z**2, eps)
    elif kind == "hard-wall":
        V = 0.75/np.maximum(z**2, eps)
        V[z>z0] = 1e9
        return V
    else:  # soft+hard
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
    dz = np.gradient(z)
    dVdz = np.gradient(V, z)
    dpsi = np.gradient(psi, z, axis=1)
    T = 0.5*np.trapz((dpsi**2), z, axis=1)
    Vexp = np.trapz(psi**2 * V, z, axis=1)
    vir = np.trapz(psi**2 * (z * dVdz), z, axis=1)
    return T, Vexp, vir

def panel():
    st.markdown("#### AdS/QCD — potential family, WKB vs FD, orthonormality, virial, convergence")
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
        ax.set_xlabel("n"); ax.set_ylabel("m^2"); ax.legend(); ax.set_title("Spectrum")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        for i in range(min(3, psi.shape[0])):
            ax2.plot(z, psi[i], label=f"ψ_{i}")
        ax2.set_xlabel("z"); ax2.set_ylabel("ψ(z)"); ax2.legend(); ax2.set_title("Eigenfunctions")
        st.pyplot(fig2)
    with c3:
        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.plot(z, V); ax3.set_xlabel("z"); ax3.set_ylabel("V(z)"); ax3.set_title("Potential")
        st.pyplot(fig3)

    S = orthonormality(psi[:min(4, psi.shape[0])], z)
    T, Vexp, vir = virial_check(z, psi[:min(4, psi.shape[0])], V)

    st.markdown("##### Orthonormality & virial diagnostics (first 4 states)")
    st.dataframe(S, use_container_width=True)
    st.dataframe({"T": T[:4], "V": Vexp[:4], "z dV/dz": vir[:4]}, use_container_width=True)

    st.markdown("##### Convergence (ground state)")
    sizes = [250, 400, 600, 900, 1200]
    vals0 = []
    for n in sizes:
        _, v, _, _ = solve_fd(kind, k, zmax, n, 3, z0)
        vals0.append(float(v[0]))
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(sizes, vals0, 'o-'); ax4.set_xlabel("N grid"); ax4.set_ylabel("m0^2")
    ax4.set_title("Convergence m0^2"); st.pyplot(fig4)

    with st.expander("Physics discussion • What changed with your controls?"):
        st.markdown(f"""
- **Potential choice** changes turning points and thus WKB action → spectrum slope/spacing.
- **k** controls soft‑wall curvature; for soft‑wall, expect \(m_n^2\sim 4k(n+1)\).
- **z0** introduces a hard cutoff → level crowding and wavefunction support truncation.
- **Grid N** and **z_max** affect discretization/box size; convergence plot shows truncation error.
- **Orthonormality** close to identity and stable virial‑style values support solver correctness.
""")
    # Telemetry
    rel = abs(float(vals[0]) - float(wkb[0]))/float(wkb[0]) if len(wkb)>0 and wkb[0]!=0 else float("nan")
    offdiag = float(np.linalg.norm(S - np.diag(np.diag(S))))
    log_event("ads_qcd", {
        "kind": kind, "k": float(k), "z0": float(z0), "N": int(N),
        "m0_fd": float(vals[0]), "m0_rel_err_wkb": rel, "orth_offdiag_norm": offdiag
    })
