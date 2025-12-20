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
    csv_path = _resolve_csv()
    st.caption(f"Static CSV: `{csv_path}`")
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)

    st.markdown("##### Live session metrics")
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
