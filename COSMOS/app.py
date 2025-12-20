import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics-Plus v3", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v3)")
st.caption("Controls · Interpretability · WKB vs FD · Orthonormality · Robust paths")

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
    st.markdown("**Physics notes**:")
    st.markdown("""
- **Exoplanets**: Coarse BLS-like scan and box-fit give a transparent period/depth estimate;
  SNR is computed from post-fit residuals. Real data supported via `lightkurve` when available.
- **Landscape**: Regression of \(h_{11}\) with conformal intervals (MAPIE) + classification of sign(\(\chi\)).
  We include **permutation feature importance** and **partial dependence** for interpretability without heavy deps.
- **AdS/QCD**: Choose **soft-wall/hard-wall/combined** potentials. We solve the SL problem by finite differences,
  compare with **WKB quantization**, and display **orthonormality** and a heuristic **virial** check.
""")
    st.markdown("**License**: MIT for this repo; third‑party libs retain their own licenses (see `THIRD_PARTY.md`).")
