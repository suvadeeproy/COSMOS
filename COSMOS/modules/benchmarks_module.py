import streamlit as st, pandas as pd
from pathlib import Path
from utils.session import get_metrics

def _resolve_csv():
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "static" / "benchmarks_snapshot.csv",
        Path.cwd() / "static" / "benchmarks_snapshot.csv",
        here.parent / "static" / "benchmarks_snapshot.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    target = (here.parent.parent / "static")
    target.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "module": ["exoplanet", "landscape", "ads_qcd"],
        "metric": ["period_estimate", "R2_h11", "m0^2_k=0.25"],
        "value": [1.234, 0.91, 1.12],
        "note": ["auto-generated", "auto-generated", "auto-generated"]
    })
    out = target / "benchmarks_snapshot.csv"; df.to_csv(out, index=False); return out

def panel():
    st.markdown("#### Benchmarks — static & live")
    st.caption("Static snapshot is for reproducibility/CI; live metrics summarize your interactive exploration.")
    csv_path = _resolve_csv()
    df = pd.read_csv(csv_path)
    st.markdown("**Static snapshot (for regressions):**")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Live session metrics** (pushed by other tabs):")
    metrics = get_metrics()
    if not metrics:
        st.info("Interact with the other tabs to populate live metrics.")
        return
    rows = []
    for scope, d in metrics.items():
        for k, v in d.items():
            rows.append({"module": scope, "metric": k, "value": v})
    live_df = pd.DataFrame(rows)
    st.dataframe(live_df, use_container_width=True)

    with st.expander("Math meaning of the metrics"):
        st.markdown("""
- **SNR ≈ depth / σ(residual)**: signal amplitude over residual noise scale.  
- **R² = 1 - SS_res / SS_tot**: fraction of variance explained (regression).  
- **AUC = P(score(x⁺) > score(x⁻))**: ranking-based separability (classification).  
- **Coverage**: empirical estimate of \(P(y \in [\hat y^- , \hat y^+])\).  
- **Soft-wall slope error**: compares fitted \(d m_n^2 / d(n+1)\) to the analytic trend \(4k\).  
- **Orthonormality off-diagonal max**: diagnostic of basis quality; should decrease as discretization improves.
""")
