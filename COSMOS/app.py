import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v6 (Math)", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (v6, Math‑Enriched)")
st.caption("Plot‑aware physics commentary • Formal statements • Verified numerics • Robust ops")

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
    st.subheader("What each page teaches (now with math)")
    st.markdown("""
- **Exoplanets** — BLS objective; \(\delta\approx (R_p/R_\star)^2\); \(D\approx R_\star/(\pi a)\); Kepler's law; SNR; aliasing logic.  
- **Landscape** — Split‑conformal coverage statement; permutation importance functional; partial dependence definition and limits.  
- **AdS/QCD** — Sturm–Liouville operator; soft‑wall \(V\); WKB quantization; virial theorem; 2nd‑order discretization error.  
- **Benchmarks** — Metric semantics tied to first‑principles.
""")
    st.subheader("License & credits"); st.markdown("MIT for this repo; third‑party packages keep their own licenses.")
