import numpy as np, streamlit as st, matplotlib.pyplot as plt

def _try_lk():
    try:
        import lightkurve as lk
        return lk
    except Exception:
        return None

def _synthetic_example(L=1024, seed=0):
    rs = np.random.default_rng(seed)
    t = np.arange(L)
    base = 1.0 + 1e-3*np.sin(2*np.pi*t/300) + rs.normal(0, 6e-4, size=L)
    start, dur, depth = 100, 40, 0.003
    s = base.copy()
    s[start:start+dur] *= (1.0 - depth)
    return base/base.mean(), s/s.mean()

def panel():
    st.markdown("#### Exoplanets — BLS folding (real if available)")
    use_real = st.toggle("Use real data via lightkurve (if installed)", value=False)
    lk = _try_lk() if use_real else None
    if lk is not None:
        target = st.text_input("Target (TIC/KIC)", "TIC 259960355")
        try:
            sr = lk.search_lightcurve(target, author="SPOC")
            if sr is None or len(sr.table) == 0:
                raise RuntimeError("No light curves found for this target/author.")
            lc = sr.download()
            if lc is None:
                raise RuntimeError("Download returned None (quota/network issue).")
            lc = lc.remove_outliers().flatten()
            pg = lc.to_periodogram(method="bls")
            best = float(pg.period_at_max_power.value)
            folded = lc.fold(period=best)
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"BLS best period: **{best:.6f} d**")
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(lc.time.value, lc.flux.value, ".", ms=2)
                ax.set_xlabel("Time [d]"); ax.set_ylabel("Flux"); ax.set_title("Real light curve")
                st.pyplot(fig)
            with c2:
                fig2, ax2 = plt.subplots(figsize=(6,3))
                ax2.plot(folded.time.value, folded.flux.value, ".", ms=2)
                ax2.set_xlabel("Phase [d]"); ax2.set_ylabel("Flux"); ax2.set_title("Folded (BLS)")
                st.pyplot(fig2)
            return
        except Exception as e:
            st.warning(f"Real-path failed: {e}. Falling back to synthetic.")
    clean, transit = _synthetic_example(L=1024, seed=0)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(clean, label="clean")
    ax.plot(transit, label="with transit")
    ax.legend(); ax.set_title("Synthetic transit example")
    st.pyplot(fig)
