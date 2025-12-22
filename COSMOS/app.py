import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v5", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v5)")
st.caption("Plot‑aware physics commentary • Uncertainty with guarantees • Verified numerics • Robust ops")

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
    st.subheader("What each page teaches (beyond the plots)")
    st.markdown("""
- **Exoplanets** — How periodic, shallow signals emerge from noise; what depth/duty cycle say about planet/geometry; why residuals matter.  
- **Landscape** — What *guaranteed* uncertainty (conformal) looks like in practice; how to read reliance (permutation FI) vs response (partial dependence).  
- **AdS/QCD** — How spectra encode confinement; why WKB already tracks patterns; how orthonormality & convergence certify numerics.  
- **Benchmarks** — How to monitor a session with objective metrics; why static snapshots are essential for CI.
""")
    st.subheader("Using the app responsibly")
    st.markdown("""
This tool illustrates **principled** pipelines: it does not hide assumptions. Real research requires careful data vetting (systematics),
out‑of‑distribution checks, and independent solver cross‑validation. The UI nudges you toward those habits.
""")
    st.subheader("License & credits")
    st.markdown("""
MIT for this repo. Third‑party packages retain their own licenses (see `THIRD_PARTY.md`).  
Design emphasizes **free‑tier** deployability and **transparent** physics/ML pedagogy.
""")
