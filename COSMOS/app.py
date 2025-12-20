import streamlit as st
from utils.reproducibility import set_global_seed

st.set_page_config(page_title="COSMOS — Scientific ML (Ready)", layout="wide")
st.title("COSMOS — Unified Scientific ML Workbench (Ready)")
st.caption("Hardened build · Optional real data · No fragile deps")

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
    st.markdown("**Citations (not redistributed)**: Akeson+2013; Mandel & Agol 2002; Karch+2006; Pedregosa+2011; MAPIE documentation.")
    st.markdown("**License**: MIT for this repo; third‑party libs keep their own licenses (see `THIRD_PARTY.md`).")
