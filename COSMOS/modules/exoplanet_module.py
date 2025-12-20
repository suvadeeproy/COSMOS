import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def _try_import_lightkurve():
    try:
        import lightkurve as lk
        return lk
    except Exception as e:
        st.info("`lightkurve` unavailable; synthetic fallback.")
        return None

def _synthetic_series(L=1024, seed=0):
    rs = np.random.default_rng(seed)
    t = np.arange(L)
    base = 1.0 + 1e-3*np.sin(2*np.pi*t/300) + rs.normal(0, 6e-4, size=L)
    start = 100; dur = 40; depth = 0.003
    s = base.copy()
    s[start:start+dur] *= (1.0 - depth)
    return base/base.mean(), s/s.mean()

def panel():
    st.markdown("#### Exoplanets: real BLS folding (fallback to synthetic)")
    enable_real = st.checkbox("Enable real data (requires lightkurve)", value=True)
    lk = _try_import_lightkurve() if enable_real else None

    if lk is not None:
        target = st.text_input("Target (TIC/KIC)", "TIC 259960355")
        try:
            sr = lk.search_lightcurve(target, author="SPOC")
            lc = sr.download().remove_outliers().flatten()
            pg = lc.to_periodogram(method="bls")
            best_period = float(pg.period_at_max_power.value)
            folded = lc.fold(period=best_period)
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"Best period (BLS): **{best_period:.6f} d**")
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(lc.time.value, lc.flux.value, ".", ms=2)
                ax.set_xlabel("Time [d]"); ax.set_ylabel("Flux"); ax.set_title("Real light curve")
                st.pyplot(fig)
            with c2:
                fig2, ax2 = plt.subplots(figsize=(6,3))
                ax2.plot(folded.time.value, folded.flux.value, ".", ms=2)
                ax2.set_xlabel("Phase [d]"); ax2.set_ylabel("Flux"); ax2.set_title("Folded by BLS period")
                st.pyplot(fig2)
            return
        except Exception as e:
            st.warning(f"Real-data path failed: {e}. Using synthetic.")

    clean, transit = _synthetic_series(L=1024, seed=0)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(clean, label="clean")
    ax.plot(transit, label="with transit")
    ax.legend(); ax.set_title("Synthetic transit example")
    st.pyplot(fig)
