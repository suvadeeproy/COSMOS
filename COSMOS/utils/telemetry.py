import time, math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import streamlit as st
import pandas as pd

@dataclass
class Event:
    ts: float
    module: str
    metrics: Dict[str, Any]

def _ensure_state():
    if "telemetry" not in st.session_state:
        st.session_state["telemetry"] = []  # type: List[Event]

def log_event(module: str, metrics: Dict[str, Any]):
    _ensure_state()
    clean = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, (int, float)):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                clean[k] = float(v)
            else:
                clean[k] = v
        except Exception:
            pass
    st.session_state["telemetry"].append(Event(time.time(), module, clean))

def get_df() -> pd.DataFrame:
    _ensure_state()
    if not st.session_state["telemetry"]:
        return pd.DataFrame(columns=["ts","module","metrics"])
    rows = []
    for ev in st.session_state["telemetry"]:
        row = {"ts": ev.ts, "module": ev.module}
        row.update(ev.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts_readable"] = pd.to_datetime(df["ts"], unit="s")
    return df
