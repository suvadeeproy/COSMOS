import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics-Plus v4", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics-Plus v4)")
st.caption("Live physics commentary • Uncertainty with guarantees • Verified numerics • Robust ops")

set_global_seed(42)

tabs = st.tabs(["1. Exoplanets", "2. Landscape", "3. AdS/QCD", "4. Benchmarks", "5. About"])

with tabs[0]:
    from modules.exoplanet_module import panel as exo_panel
    exo_panel()

with tabs[1]:
    from modules.landscape_module import panel as land_panel
    land_panel()

with tabs[2]:
    from modules.ads_qcd_module import panel as ads_panel
    ads_panel()

with tabs[3]:
    from modules.benchmarks_module import panel as bm_panel
    bm_panel()

with tabs[4]:
    st.subheader("What this app demonstrates")
    st.markdown("""
**Scientific method in code**: model → inference → diagnostics → uncertainty → cross-checks → reproducibility.

- **Exoplanets**: Periodic change-point detection under noise (BLS-style) with a transparent **box model** and **residual SNR**.
- **Landscape**: Predict a geometric invariant with **distribution-free prediction intervals** (conformal) and show **why** via permutation importance and partial dependence.
- **AdS/QCD**: Solve a Sturm–Liouville eigenproblem, then **validate** against analytics (soft-wall linearity) and **WKB**. Check orthonormality and **grid convergence**.
- **Benchmarks**: Keep a static snapshot for CI/regressions and a **live** metrics table populated by your current session.

**Why this matters**: It is a compact lab for scientific ML that is honest about error bars, interpretable about mechanisms, and rigorous about numerics—exactly what research groups and top-tier engineering teams expect.
""")
    st.subheader("Using this app")
    st.markdown("""
- Start with *Exoplanets*: toggle real data if `lightkurve` is available, otherwise tune synthetic signals and watch the **live notes** quantify SNR and aliasing risk.
- Move to *Landscape*: adjust sample size/noise/trees/depth; check how coverage tracks the nominal level and how features influence predictions.
- Explore *AdS/QCD*: switch potential families; compare **FD** vs **WKB**; use the **live metrics** to see slope errors and orthonormality quality.
- Open *Benchmarks*: read the static CSV and the **Live session metrics** to summarize your run.
""")
    st.subheader("Reproducibility & licensing")
    st.markdown("""
- Reproducible seeds, minimal dependencies, robust file resolution, and basic spectral tests.
- **License**: MIT for this repository; third-party packages retain their own licenses (see `THIRD_PARTY.md`).
""")
