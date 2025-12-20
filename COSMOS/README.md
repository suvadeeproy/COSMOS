# COSMOS — Scientific ML Workbench (Ready Build)

A hardened Streamlit app with zero-cost deploy and **no fragile deps**.
- Exoplanets: optional `lightkurve` (real data) with **safe fallback** to synthetic if missing.
- Landscape: GBM regression/classification with **conformal intervals** (MAPIE). No SHAP required.
- AdS/QCD: finite-difference Sturm–Liouville solver with **convergence** panel.
- Benchmarks: local CSV (no network).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
