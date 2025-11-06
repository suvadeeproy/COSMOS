
# --- STREAMLIT CLOUD IMPORT SAFETY ---
import sys, os
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# -------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from cosmos_workbench.astro.synthetic import make_synth
from cosmos_workbench.astro.models.cnn1d import CNN1D
from cosmos_workbench.astro.utils.calib import ece, temp_scale, reliability_curve

from cosmos_workbench.landscape.data.cicy_synth import make_synth as make_cicy
from cosmos_workbench.landscape.models.gbm import make_models, fit, predict
from sklearn.inspection import permutation_importance

from cosmos_workbench.holo.models.softwall import SoftWallParam
from cosmos_workbench.holo.solvers.sl_fd import solve_softwall

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

st.set_page_config(page_title="COSMOS — Scientific ML Workbench", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench")

st.sidebar.subheader("Session controls")
research_mode = st.sidebar.checkbox("Research mode (physics + assumptions)", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Exoplanets → Landscape → Holography → Benchmarks")

tab1, tab2, tab3, tab4 = st.tabs(["1. Exoplanets", "2. Landscape", "3. Holography", "4. Benchmarks/About"])

# 1) Exoplanets
with tab1:
    st.markdown("### 1. Exoplanet transits — calibrated ML with uncertainty")
    if research_mode:
        st.info("We generate normalized photometry with/without a box-like transit, fit a 1D-CNN, and evaluate discrimination (ROC/PR) and calibration (reliability/ECE). We also show epistemic uncertainty via MC-dropout.")

    n = st.slider("Synthetic series", 500, 5000, 2000, step=250)
    L = st.selectbox("Length", [256,512,1024], index=1)
    sigma = st.slider("Noise σ", 0.0001, 0.0020, 0.0008, step=0.0001, format="%.4f")
    X, y = make_synth(n=n, L=L, sigma=sigma)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1D(length=L, dropout_p=0.2).to(device)
    xb = torch.from_numpy(X[:1024]).to(device); yb = torch.from_numpy(y[:1024]).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(4):
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(model(xb), yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
    p = 1/(1+np.exp(-logits))
    e_before = ece(p, y, bins=12)
    T = temp_scale(logits, y.astype(float))
    pT = 1/(1+np.exp(-logits/max(T,1e-4)))
    e_after = ece(pT, y, bins=12)

    c1, c2 = st.columns(2)
    with c1:
        idx = st.slider("Inspect LC index", 0, n-1, 0)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(range(L), X[idx]); ax.set_xlabel("time"); ax.set_ylabel("norm. flux"); ax.set_title("Sample light curve")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(p[y==0], bins=20, alpha=0.5, label="no transit")
        ax2.hist(p[y==1], bins=20, alpha=0.5, label="transit")
        ax2.set_xlabel("pred prob"); ax2.set_ylabel("count"); ax2.set_title("Predictions by class (pre-calibration)"); ax2.legend()
        st.pyplot(fig2)

        fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
        fig3, ax3 = plt.subplots(figsize=(6,3))
        ax3.plot(fpr, tpr, '-', label=f"ROC AUC={roc_auc:.3f}")
        ax3.plot([0,1],[0,1],'k--'); ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR"); ax3.set_title("ROC"); ax3.legend()
        st.pyplot(fig3)
        fig4, ax4 = plt.subplots(figsize=(6,3))
        ax4.plot(rec, prec, '-'); ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision"); ax4.set_title(f"PR (AP={ap:.3f})")
        st.pyplot(fig4)

    with c2:
        conf_b, acc_b, _ = reliability_curve(p, y, bins=12)
        conf_a, acc_a, _ = reliability_curve(pT, y, bins=12)
        fig5, ax5 = plt.subplots(figsize=(6,3))
        ax5.plot([0,1],[0,1],'k--',label="ideal")
        ax5.plot(conf_b, acc_b, 'o-', label="before")
        ax5.plot(conf_a, acc_a, 's-', label="after T")
        ax5.set_xlabel("confidence"); ax5.set_ylabel("accuracy"); ax5.set_title("Reliability"); ax5.legend()
        st.pyplot(fig5)

        mc = st.slider("MC-dropout samples", 5, 50, 20, step=5)
        model.train()
        preds = []
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(device)
            for _ in range(mc):
                preds.append(1/(1+np.exp(-model(X_t).cpu().numpy())))
        preds = np.stack(preds, axis=0)
        std_p = preds.std(axis=0)
        fig6, ax6 = plt.subplots(figsize=(6,3))
        ax6.hist(std_p[y==0], bins=20, alpha=0.5, label="no transit")
        ax6.hist(std_p[y==1], bins=20, alpha=0.5, label="transit")
        ax6.set_xlabel("MC std of p"); ax6.set_ylabel("count"); ax6.set_title("Epistemic uncertainty by class"); ax6.legend()
        st.pyplot(fig6)

    st.write(f"ECE before: **{e_before:.3f}**, after T: **{e_after:.3f}** (T={T:.2f})")

# 2) Landscape
with tab2:
    st.markdown("### 2. CICY / landscape surrogates (heuristic, with diagnostics)")
    if research_mode:
        st.warning("Surrogate ML for (h^{1,1}, h^{2,1}, \chi). Use for ranking/hypotheses; confirm with authoritative data.")

    N = st.slider("Synthetic CICY-like samples", 800, 8000, 3000, step=400)
    amb = st.slider("Ambient max", 3, 6, 5)
    cod = st.slider("Codim max", 1, 3, 2)
    S = make_cicy(n=N, ambient_max=amb, codim_max=cod)

    models = fit(make_models(seed=0), S.X, S.y_reg, S.y_cls)
    y_pred, p_sign = predict(models, S.X)

    col_sc, col_err = st.columns(2)
    with col_sc:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(S.y_reg[:,0], S.y_reg[:,1], s=6, alpha=0.35, label="true")
        ax.scatter(y_pred[:,0], y_pred[:,1], s=6, alpha=0.35, label="pred")
        ax.set_xlabel("h^{1,1}"); ax.set_ylabel("h^{2,1}"); ax.set_title("True vs predicted"); ax.legend()
        st.pyplot(fig)
    with col_err:
        err_h11 = S.y_reg[:,0] - y_pred[:,0]
        err_h21 = S.y_reg[:,1] - y_pred[:,1]
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.hist(err_h11, bins=20, alpha=0.6, label="h11 error")
        ax2.hist(err_h21, bins=20, alpha=0.6, label="h21 error")
        ax2.set_xlabel("error"); ax2.set_ylabel("count"); ax2.set_title("Error histograms"); ax2.legend()
        st.pyplot(fig2)

    if st.checkbox("Show permutation feature importances"):
        imp = permutation_importance(models.reg_h11, S.X, S.y_reg[:,0], n_repeats=5, random_state=0)
        fig3, ax3 = plt.subplots(figsize=(5,3))
        order = np.argsort(imp.importances_mean)
        ax3.barh([f"f{i}" for i in order], imp.importances_mean[order])
        ax3.set_title("Permuted importance (h11 regressor)")
        st.pyplot(fig3)

# 3) Holography
with tab3:
    st.markdown("### 3. Soft-wall AdS/QCD — parametric fit & SL eigen-solver")
    sub1, sub2 = st.tabs(["Parametric fit", "SL eigen-solver"])

    with sub1:
        n_levels = st.slider("Levels", 4, 16, 8, key="levels_fit")
        k_true = st.slider("true k", 0.05, 0.6, 0.25, step=0.01, key="k_true_fit")
        noise = st.slider("noise σ", 0.0, 0.08, 0.02, step=0.005, key="noise_fit")
        rng = np.random.default_rng(0)
        n = np.arange(n_levels, dtype=np.float32)
        m2 = 4.0*k_true*(n+1.0) + rng.normal(0.0, noise, size=n_levels)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SoftWallParam(init_k=0.1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        n_t = torch.tensor(n, dtype=torch.float32, device=device)
        m2_t = torch.tensor(m2, dtype=torch.float32, device=device)
        for _ in range(400):
            opt.zero_grad(); pred = model(n_t); loss = F.mse_loss(pred, m2_t); loss.backward(); opt.step()
        k_est = float(torch.relu(model.k).detach().cpu().numpy())

        col_main, col_res = st.columns(2)
        with col_main:
            st.write(f"Fitted k: **{k_est:.3f}** (true {k_true:.3f})")
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(n, m2, 'o', label='data')
            ax.plot(n, model(n_t).detach().cpu().numpy(), '-', label='fit')
            ax.set_xlabel("n"); ax.set_ylabel("m^2"); ax.set_title("Data vs fit"); ax.legend()
            st.pyplot(fig)
        with col_res:
            fit_vals = model(n_t).detach().cpu().numpy()
            resid = m2 - fit_vals
            fig2, ax2 = plt.subplots(figsize=(5,3))
            ax2.stem(n, resid, basefmt=" ")
            ax2.set_xlabel("n"); ax2.set_ylabel("residual"); ax2.set_title("Residuals")
            st.pyplot(fig2)

    with sub2:
        k_sl = st.slider("k (soft-wall slope)", 0.05, 0.6, 0.25, step=0.01, key="k_sl")
        zmax = st.slider("z_max", 4.0, 12.0, 8.0, step=0.5, key="zmax_sl")
        Ngrid = st.slider("grid points", 200, 1200, 600, step=50, key="Ngrid_sl")
        neigs = st.slider("eigenpairs", 3, 10, 6, step=1, key="neigs_sl")

        z, vals, psi = solve_softwall(float(k_sl), float(zmax), int(Ngrid), int(neigs))
        m2_num = np.array(sorted(vals))

        csp, cwv = st.columns(2)
        with csp:
            fig3, ax3 = plt.subplots(figsize=(5,3))
            ax3.plot(np.arange(len(m2_num)), m2_num, 'o-', label="FD spectrum")
            ax3.plot(np.arange(len(m2_num)), 4.0*k_sl*(np.arange(len(m2_num))+1.0), 'x--', label="4k(n+1)")
            ax3.set_xlabel("n"); ax3.set_ylabel("m^2"); ax3.set_title("Spectrum: numerical vs analytic")
            ax3.legend(); st.pyplot(fig3)
        with cwv:
            fig4, ax4 = plt.subplots(figsize=(5,3))
            for i in range(min(3, psi.shape[0])):
                ax4.plot(z, psi[i], label=f"ψ_{i}(z)")
            ax4.set_xlabel("z"); ax4.set_ylabel("ψ(z)"); ax4.set_title("Low-lying eigenfunctions")
            ax4.legend(); st.pyplot(fig4)

# 4) Benchmarks
with tab4:
    st.markdown("### 4. Benchmarks, open sources & about")
    st.write("Fetch small open tables at runtime; nothing is stored.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("NASA Exoplanet Archive (50 rows)"):
            import pandas as pd, requests
            try:
                url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=ps&select=pl_name,hostname,discoverymethod,pl_orbper,pl_rade&format=csv&max_records=50"
                df = pd.read_csv(url)
                st.dataframe(df)
            except Exception as e:
                st.error(f"Fetch failed: {e}")
    with col2:
        st.markdown("CICY list (external): https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/")

    st.markdown('---')
    if st.button("Show local physics notes"):
        try:
            txt = open("docs/physics_explanations.md","r",encoding="utf-8").read()
            st.markdown(txt)
        except Exception as e:
            st.error(f"Cannot load notes: {e}")
