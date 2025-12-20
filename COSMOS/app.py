import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v6", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v6)")
st.caption("Physics‑first explanations • Explicit formulas • Verified numerics • Reproducible runs")

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
    st.subheader("Scope")
    st.markdown("""
- **Exoplanets** — detection from photometry with a box model and diagnostics that quantify credibility.  
- **Landscape** — predictive modeling with finite‑sample uncertainty and interpretable knobs.  
- **AdS/QCD** — eigenvalue problems and semiclassical checks with basic spectral tests.  
- **Benchmarks** — session metrics for regression testing and at‑a‑glance summaries.
""")
    st.subheader("Method")
    st.markdown("""
Reproducible seeds, minimal dependencies, robust file resolution, and explicit numerical checks. Formulas and definitions are presented near the relevant plots to maintain continuity between theory and computation.
""" )
    st.subheader("License")
    st.markdown("""
MIT for this repository. Third‑party packages preserve their licenses; see `THIRD_PARTY.md`.
""" )
