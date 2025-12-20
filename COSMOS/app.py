import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v4", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v4)")
st.caption("Controls · Live physics commentary · Session telemetry · Reproducible numerics")

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
    st.subheader("What this app is")
    st.markdown("""
This is a compact **scientific machine‑learning lab**. Each module demonstrates an end‑to‑end workflow:
**model → inference → diagnostics → uncertainty → interpretation**, with **live commentary** and **session telemetry**.
""")
    st.subheader("Why it matters")
    st.markdown("""
- **Exoplanets:** Transparent period detection and uncertainty‑aware fitting in noisy time series.
- **Landscape:** **Distribution‑free** predictive intervals (conformal prediction) and mechanistic interpretability.
- **AdS/QCD:** A validated eigen‑solver (FD vs WKB) with orthonormality and convergence—hallmarks of trustworthy numerics.
""")
    st.subheader("How to use")
    st.markdown("""
- Adjust controls at the top of each module. Watch the **explainer** expanders for physics insight tied to your choices.
- Check **Benchmarks → Live Telemetry** to see how your session's metrics (SNR, R²/AUC, m₀², etc.) evolved.
- Optional: enable real Kepler/TESS data by installing `lightkurve`/`astropy`.
""")
    st.subheader("Reproducibility & ethics")
    st.markdown("""
- Fixed random seeds; minimal dependencies; no opaque remote calls by default.
- Third‑party packages keep their licenses (see **THIRD_PARTY.md**). This app is MIT‑licensed.
""")
    st.subheader("Citations")
    st.markdown("""
Kovács–Zucker–Mazeh (BLS), Mandel–Agol (transits), Pedregosa et al. (scikit‑learn), MAPIE (conformal),
Erlich–Katz–Son–Stephanov & Karch et al. (AdS/QCD), Teschl (SL theory).
""")
