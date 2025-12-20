import streamlit as st, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from utils.telemetry import get_df

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
    st.markdown("#### Benchmarks & Live Telemetry")
    tab1, tab2 = st.tabs(["Deterministic Benchmarks", "Live Telemetry (this session)"])

    with tab1:
        csv_path = _resolve_csv()
        st.caption(f"Static CSV: `{csv_path}`")
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

    with tab2:
        tdf = get_df()
        if tdf.empty:
            st.info("No telemetry yet. Interact with the other tabs—results will appear here.")
        else:
            st.dataframe(tdf, use_container_width=True)
            # Small module-wise summaries
            mods = tdf["module"].unique()
            for m in mods:
                st.markdown(f"**{m}**")
                sub = tdf[tdf["module"]==m]
                if m == "exoplanet" and {"best_period","snr"}.issubset(sub.columns):
                    fig, ax = plt.subplots(figsize=(4,2.5))
                    ax.plot(sub["ts_readable"], sub["snr"], "-o"); ax.set_ylabel("SNR"); ax.set_title("SNR over session")
                    st.pyplot(fig)
                if m == "landscape" and {"r2","auc"}.issubset(sub.columns):
                    fig, ax = plt.subplots(figsize=(4,2.5))
                    ax.plot(sub["ts_readable"], sub["r2"], "-o", label="R2")
                    ax.plot(sub["ts_readable"], sub["auc"], "-o", label="AUC")
                    ax.legend(); ax.set_title("R2/AUC over session"); st.pyplot(fig)
                if m == "ads_qcd" and {"m0_fd","m0_rel_err_wkb"}.issubset(sub.columns):
                    fig, ax = plt.subplots(figsize=(4,2.5))
                    ax.plot(sub["ts_readable"], sub["m0_fd"], "-o", label="m0^2 FD")
                    ax2 = ax.twinx()
                    ax2.plot(sub["ts_readable"], sub["m0_rel_err_wkb"], "-.", label="rel err WKB", color="tab:red")
                    ax.set_title("Ground state & WKB rel err"); ax.legend(loc="upper left")
                    st.pyplot(fig)
