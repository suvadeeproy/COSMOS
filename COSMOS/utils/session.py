import streamlit as st

def put_metric(scope: str, key: str, value):
    st.session_state.setdefault("metrics", {})
    st.session_state["metrics"].setdefault(scope, {})
    st.session_state["metrics"][scope][key] = value

def get_metrics():
    return st.session_state.get("metrics", {})
