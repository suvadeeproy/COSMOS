import numpy as np, streamlit as st, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from utils.uncertainty import mapie_regression_intervals
from utils.session import put_metric

def make_synth(n=4000, seed=0, noise=1.0):
    rs = np.random.default_rng(seed)
    fsum = rs.integers(8, 25, size=(n,1))
    fvar = rs.uniform(0.5, 10.0, size=(n,1))
    fcod = rs.integers(1, 3, size=(n,1))
    X = np.concatenate([fsum, fvar, fcod], axis=1).astype(np.float32)
    h11 = (0.6*fsum[:,0] + 3.0*fcod[:,0] + 2.0*np.sqrt(fvar[:,0]) + rs.normal(0,noise,n)).clip(1,35)
    h21 = (35 - 0.5*fsum[:,0] + 1.5*np.sqrt(fvar[:,0]) + rs.normal(0,1.5*noise,n)).clip(1,50)
    chi = 2*(h11 - h21)
    y_reg = np.stack([h11, h21, chi], axis=1).astype(np.float32)
    q = np.quantile(chi, 0.5)
    y_cls = (chi > q).astype(int)
    if y_cls.sum() == 0 or y_cls.sum() == len(y_cls):
        q = np.quantile(chi, 0.45)
        y_cls = (chi > q).astype(int)
    return X, y_reg, y_cls

def perm_importance(model, X, y, scorer):
    rs = np.random.default_rng(0)
    baseline = scorer(y, model.predict(X))
    imps = []
    for j in range(X.shape[1]):
        Xp = X.copy()
        rs.shuffle(Xp[:,j])
        imps.append(baseline - scorer(y, model.predict(Xp)))
    return np.array(imps)

def partial_dependence_1d(model, X, j, grid=None):
    if grid is None:
        vmin, vmax = np.percentile(X[:,j], [1, 99])
        grid = np.linspace(vmin, vmax, 40)
    Xref = X.copy()
    yh = []
    for v in grid:
        Xref[:,j] = v
        yh.append(model.predict(Xref).mean())
    return grid, np.array(yh)

def panel():
    st.markdown("#### Landscape — GBM with conformal intervals, permutation FI, partial dependence")
    cols = st.columns(4)
    with cols[0]:
        n = st.slider("n samples", 1000, 20000, 4000, 500)
    with cols[1]:
        noise = st.slider("noise level", 0.2, 3.0, 1.0, 0.1)
    with cols[2]:
        trees = st.slider("n_estimators", 100, 800, 300, 50)
    with cols[3]:
        depth = st.slider("max_depth", 2, 6, 3, 1)

    X, y_reg, y_cls = make_synth(n=n, seed=0, noise=noise)
    Xtr, Xte, Yr_tr, Yr_te, ycl_tr, ycl_te = train_test_split(
        X, y_reg, y_cls, test_size=0.25, random_state=0, stratify=y_cls
    )

    reg = GradientBoostingRegressor(random_state=0, n_estimators=trees, max_depth=depth)
    reg.fit(Xtr, Yr_tr[:,0])
    y_pred, lo, hi = mapie_regression_intervals(reg, Xtr, Yr_tr[:,0], Xte, alpha=0.1)

    cls = GradientBoostingClassifier(random_state=1, n_estimators=trees, max_depth=depth)
    cls.fit(Xtr, ycl_tr.astype(int))
    auc = roc_auc_score(ycl_te.astype(int), cls.predict_proba(Xte)[:,1])

    valid = ~np.isnan(lo) & ~np.isnan(hi)
    coverage = float(np.mean((Yr_te[:,0][valid] >= lo[valid]) & (Yr_te[:,0][valid] <= hi[valid]))) if valid.any() else float('nan')

    c1, c2 = st.columns(2)
    with c1:
        r2 = r2_score(Yr_te[:,0], y_pred)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(Yr_te[:,0], y_pred, s=8, alpha=0.6)
        lo_ = np.nan_to_num(lo, nan=y_pred); hi_ = np.nan_to_num(hi, nan=y_pred)
        yerr_lower = np.clip(y_pred - lo_, 0, None); yerr_upper = np.clip(hi_ - y_pred, 0, None)
        ax.errorbar(Yr_te[:,0], y_pred, yerr=[yerr_lower, yerr_upper], fmt=".", alpha=0.2, lw=0.5)
        ax.plot([Yr_te[:,0].min(), Yr_te[:,0].max()], [Yr_te[:,0].min(), Yr_te[:,0].max()], 'k--', lw=1)
        ax.set_xlabel("true h11"); ax.set_ylabel("pred h11"); ax.set_title(f"h11 regression with intervals (R2={r2:.3f})")
        st.pyplot(fig)
    with c2:
        st.write(f"Classifier AUC for sign(chi): **{auc:.3f}**")
        st.write(f"Empirical coverage @90% nominal: **{coverage:.3f}**" if valid.any() else "Coverage: N/A (intervals not available)")

    st.markdown("##### Permutation feature importance (ΔR2)")
    imps = perm_importance(reg, Xte, Yr_te[:,0], scorer=r2_score)
    fig3, ax3 = plt.subplots(figsize=(4,3))
    ax3.bar(["fsum","fvar","fcod"], imps)
    ax3.set_ylabel("ΔR2 when permuted"); st.pyplot(fig3)

    st.markdown("##### Partial dependence (select feature)")
    j = st.selectbox("Feature", index=0, options=[0,1,2], format_func=lambda k:["fsum","fvar","fcod"][k])
    grid, yh = partial_dependence_1d(reg, Xte.copy(), j)
    fig4, ax4 = plt.subplots(figsize=(4,3))
    ax4.plot(grid, yh, "-")
    ax4.set_xlabel(["fsum","fvar","fcod"][j]); ax4.set_ylabel("E[h11 | feature]")
    st.pyplot(fig4)

    from numpy import argsort
    put_metric("landscape", "R2_h11", float(r2))
    put_metric("landscape", "AUC_sign_chi", float(auc))
    put_metric("landscape", "coverage90", float(coverage))
    put_metric("landscape", "perm_rank", [str(k) for k in (argsort(imps)[::-1])])

    with st.expander("Physics discussion (live: what each plot tells you)"):
        st.markdown(f"""
**1) Scatter with intervals**  
Points near the diagonal indicate small bias; spread reflects irreducible variance + noise. Error bars (when available) are **distribution‑free** (conformal), so the fraction of true values inside the bars empirically estimates **coverage≈{coverage:.3f}**.

**2) AUC**  
Measures separability of sign(chi) regimes; **AUC={auc:.3f}** implies the features contain usable signal about topological class.

**3) Permutation feature importance (ΔR2)**  
Bars show **how much** predictive power collapses when a feature is scrambled. This quantifies reliance, not causation.

**4) Partial dependence**  
Curve shows the conditional mean response in \(h_{11}\) as a single feature varies—useful for hypothesis‑building about structure.

**What this says about principles**  
You are practicing **honest uncertainty** (finite‑sample coverage without distributional assumptions) and **mechanistic interpretability** (importance + response). That’s the right template for geometric/landscape inference beyond toy data.
""")
