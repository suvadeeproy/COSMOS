import numpy as np, streamlit as st, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from utils.uncertainty import mapie_regression_intervals

def make_synth(n=2000, seed=0):
    rs = np.random.default_rng(seed)
    fsum = rs.integers(8, 25, size=(n,1))
    fvar = rs.uniform(0.5, 10.0, size=(n,1))
    fcod = rs.integers(1, 3, size=(n,1))
    X = np.concatenate([fsum, fvar, fcod], axis=1).astype(np.float32)
    h11 = (0.6*fsum[:,0] + 3.0*fcod[:,0] + 2.0*np.sqrt(fvar[:,0]) + rs.normal(0,1.0,n)).clip(1,35)
    h21 = (35 - 0.5*fsum[:,0] + 1.5*np.sqrt(fvar[:,0]) + rs.normal(0,1.5,n)).clip(1,50)
    chi = 2*(h11 - h21)
    y_reg = np.stack([h11, h21, chi], axis=1).astype(np.float32)
    q = np.quantile(chi, 0.5)
    y_cls = (chi > q).astype(int)
    if y_cls.sum() == 0 or y_cls.sum() == len(y_cls):
        q = np.quantile(chi, 0.45)
        y_cls = (chi > q).astype(int)
    return X, y_reg, y_cls

def panel():
    st.markdown("#### Landscape — GBM with conformal intervals (robust labels)")
    X, y_reg, y_cls = make_synth(n=2000, seed=0)

    Xtr, Xte, Yr_tr, Yr_te, ycl_tr, ycl_te = train_test_split(
        X, y_reg, y_cls, test_size=0.25, random_state=0, stratify=y_cls
    )

    reg = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3)
    reg.fit(Xtr, Yr_tr[:,0])
    y_pred, lo, hi = mapie_regression_intervals(reg, Xtr, Yr_tr[:,0], Xte, alpha=0.05)

    cls = GradientBoostingClassifier(random_state=1, n_estimators=300, max_depth=3)
    cls.fit(Xtr, ycl_tr.astype(int))
    auc = roc_auc_score(ycl_te.astype(int), cls.predict_proba(Xte)[:,1])

    c1, c2 = st.columns(2)
    with c1:
        r2 = r2_score(Yr_te[:,0], y_pred)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(Yr_te[:,0], y_pred, s=8, alpha=0.6)
        lo_ = np.nan_to_num(lo, nan=y_pred)
        hi_ = np.nan_to_num(hi, nan=y_pred)
        yerr_lower = np.clip(y_pred - lo_, 0, None)
        yerr_upper = np.clip(hi_ - y_pred, 0, None)
        ax.errorbar(Yr_te[:,0], y_pred, yerr=[yerr_lower, yerr_upper], fmt=".", alpha=0.2, lw=0.5)
        ax.plot([Yr_te[:,0].min(), Yr_te[:,0].max()],
                [Yr_te[:,0].min(), Yr_te[:,0].max()], 'k--', lw=1)
        ax.set_xlabel("true h11"); ax.set_ylabel("pred h11"); ax.set_title(f"h11 regression (R2={r2:.3f})")
        st.pyplot(fig)
    with c2:
        st.write(f"Classifier AUC for sign(chi): **{auc:.3f}**")
        st.write("Labels built via quantile threshold with stratified split to avoid single‑class failures.")
