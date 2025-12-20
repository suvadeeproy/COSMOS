import numpy as np, streamlit as st, matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def potential_softwall(z, k):
    eps = 1e-8
    return (k**2)*(z**2) + 0.75/np.maximum(z**2, eps)

def solve_fd(k: float, z_max: float = 8.0, N: int = 400, n_eigs: int = 6):
    z_min = 1e-3
    z = np.linspace(z_min, z_max, N)
    dz = z[1]-z[0]
    V = potential_softwall(z, k)
    main = 2.0*np.ones(N-2)/(dz*dz) + V[1:-1]
    off = -1.0*np.ones(N-3)/(dz*dz)
    H = diags([off, main, off], [-1,0,1], format="csr")
    vals, vecs = eigsh(H, k=min(n_eigs, N-2), which='SA', tol=1e-5, maxiter=4000)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:,idx]
    psi = np.zeros((len(vals), N))
    psi[:,1:-1] = vecs.T
    for i in range(len(vals)):
        nrm = np.sqrt(np.trapz(psi[i]**2, z))
        if nrm>0: psi[i] /= nrm
    return z, vals, psi

def panel():
    st.markdown("#### Soft-wall AdS/QCD — spectrum, eigenfunctions, convergence")
    k = st.slider("k (slope)", 0.05, 0.6, 0.25, 0.01)
    zmax = st.slider("z_max", 4.0, 12.0, 8.0, 0.5)
    N = st.slider("grid points", 200, 1000, 500, 50)
    ne = st.slider("eigenpairs", 3, 10, 6, 1)

    try:
        z, vals, psi = solve_fd(k, zmax, N, ne)
    except Exception as e:
        st.error(f"Eigen-solve failed (reducing grid/relaxing tol): {e}")
        z, vals, psi = solve_fd(k, zmax, max(300, N//2), min(ne, 5))

    m2_num = np.array(vals)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(np.arange(len(m2_num)), m2_num, 'o-', label="FD")
        ax.plot(np.arange(len(m2_num)), 4.0*k*(np.arange(len(m2_num))+1.0), 'x--', label="4k(n+1)")
        ax.set_xlabel("n"); ax.set_ylabel("m^2"); ax.legend(); ax.set_title("Spectrum vs analytic")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        for i in range(min(3, psi.shape[0])):
            ax2.plot(z, psi[i], label=f"ψ_{i}")
        ax2.set_xlabel("z"); ax2.set_ylabel("ψ(z)"); ax2.legend(); ax2.set_title("Eigenfunctions")
        st.pyplot(fig2)
    with c3:
        V = potential_softwall(z, k)
        fig3, ax3 = plt.subplots(figsize=(4,3))
        ax3.plot(z, V); ax3.set_xlabel("z"); ax3.set_ylabel("V(z)"); ax3.set_title("Potential")
        st.pyplot(fig3)

    st.markdown("##### Convergence study (ground state)")
    sizes = [200, 300, 400, 600, 800]
    vals0 = []
    for n in sizes:
        _, v, _ = solve_fd(k, zmax, n, 3)
        vals0.append(float(v[0]))
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(sizes, vals0, 'o-'); ax4.set_xlabel("N grid"); ax4.set_ylabel("m0^2")
    ax4.set_title("Convergence of ground state"); st.pyplot(fig4)
