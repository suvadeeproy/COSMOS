import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Physics‑Plus v6", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Physics‑Plus v6)")
st.caption("More math in the commentary • Uncertainty with guarantees • Verified numerics • Robust ops")

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
    st.subheader("What this app is (physicist language)")
    st.markdown("""
A unified “mini-lab” built around one habit: **turn plots into falsifiable statements**.

- Exoplanets: infer \(P\), \(d\approx(R_p/R_\star)^2\), and diagnose systematics via residuals.  
- Landscape: learn \(y=f(x)+\epsilon\) with a surrogate and certify uncertainty with finite-sample guarantees.  
- AdS/QCD: spectra as eigenvalues of operators; confinement encoded by \(V(z)\) and boundaries; numerics validated by convergence and semiclassics.

The point is not to show off plots; it’s to show the *workflow*: model → diagnostics → uncertainty → interpretation.
""")
    st.subheader("Licensing")
    st.markdown("MIT for this repo; third‑party packages retain their own licenses (see `THIRD_PARTY.md`).")
