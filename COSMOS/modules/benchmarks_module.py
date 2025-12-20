import streamlit as st, pandas as pd
def panel():
    st.markdown("#### Benchmarks (local, deterministic)")
    df = pd.read_csv("static/benchmarks_snapshot.csv")
    st.dataframe(df, use_container_width=True)
