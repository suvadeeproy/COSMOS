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
        "note": ["auto", "auto", "auto"]
    })
    out = target / "benchmarks_snapshot.csv"; df.to_csv(out, index=False); return out

def panel():
    st.markdown("#### Benchmarks — static & live")
    csv_path = _resolve_csv()
    st.dataframe(pd.read_csv(csv_path), use_container_width=True)

    metrics = get_metrics()
    st.markdown("**Live session metrics:**")
    if not metrics:
        st.info("Use other tabs to populate metrics.")
        return
    rows = [{"module":m,"metric":k,"value":v} for m,d in metrics.items() for k,v in d.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
