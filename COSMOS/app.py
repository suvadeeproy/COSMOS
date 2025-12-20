import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v6", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v6)")
st.caption("Math-forward plot explanations • free-tier deployable")

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
    st.subheader("What the extra math is for")
    st.markdown(r"""
The goal is to eliminate “pretty plots with no content.”
Each module pins its graphics to explicit definitions (likelihood/SSE, R²/AUC/coverage, Sturm–Liouville/WKB).
""")
    st.subheader("License & credits")
    st.markdown("MIT for this repo; third-party licenses in `THIRD_PARTY.md`.")
