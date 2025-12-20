import streamlit as st
import pandas as pd

def panel():
    st.markdown("#### Benchmarks (local; no network)")
    st.write("These are tiny synthetic snapshots so the app never crashes.")
    df = pd.read_csv("static/benchmarks_snapshot.csv")
    st.dataframe(df, use_container_width=True)
