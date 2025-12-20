import numpy as np, streamlit as st, matplotlib.pyplot as plt
from utils.session import put_metric

@st.cache_data(show_spinner=False)
def _synthetic_series(L=2048, seed=0, period=200, depth=0.0025, dur=40, jitter=6e-4):
    rs = np.random.default_rng(seed)
    t = np.arange(L).astype(float)
    base = 1.0 + 1e-3*np.sin(2*np.pi*t/300) + rs.normal(0, jitter, size=L)
    curve = base.copy()
    for start in range(100, L-1, int(period)):
        curve[start:start+dur] *= (1.0 - depth)
    flux = curve/curve.mean()
    return t, flux

def _try_lk():
    try:
        import lightkurve as lk
        return lk
    except Exception:
        return None

def _bls_period(time, flux, minp=0.5, maxp=20, n=200):
    periods = np.linspace(minp, maxp, n)
    sse = []
    for P in periods:
        phase = (time % P)/P
        k = max(3, int(0.05*len(flux)))
        idx = np.argsort(flux)[:k]
        mu_in = flux[idx].mean()
        mu_out = np.delete(flux, idx).mean()
        model = np.where(np.isin(np.arange(len(flux)), idx), mu_in, mu_out)
        sse.append(((flux - model)**2).sum())
    sse = np.array(sse)
    best_idx = int(np.argmin(sse))
    return periods[best_idx], periods, sse

def _box_fit(time, flux, period):
    phase = (time % period)/period
    grid = np.linspace(0, 1, 200, endpoint=False)
    widths = np.linspace(0.01, 0.2, 40)
    best = (np.inf, 0, 0)  # sse, phi0, w
    mu_out = np.median(flux)
    for w in widths:
        for phi0 in grid:
            mask = ((phase >= phi0) & (phase <= (phi0 + w) % 1.0)) if phi0+w<=1 else ((phase>=phi0) | (phase <= (phi0+w-1)))
            mu_in = np.median(flux[mask]) if mask.any() else mu_out
            model = np.where(mask, mu_in, mu_out)
            sse = ((flux - model)**2).sum()
            if sse < best[0]:
                best = (sse, phi0, w)
    sse, phi0, w = best
    mask = ((phase >= phi0) & (phase <= (phi0 + w) % 1.0)) if phi0+w<=1 else ((phase>=phi0) | (phase <= (phi0+w-1)))
    depth = max(0.0, mu_out - np.median(flux[mask])) if mask.any() else 0.0
    return dict(phi0=phi0, width=w, depth=depth, mu_out=mu_out, sse=sse)

def panel():
    st.markdown("#### Exoplanets — BLS-like search, box-fit, residuals, SNR")
    use_real = st.toggle("Use real data via lightkurve (if installed)", value=False)
    lk = _try_lk() if use_real else None

    if lk is not None and use_real:
        target = st.text_input("Target (TIC/KIC)", "TIC 259960355")
        try:
            sr = lk.search_lightcurve(target, author="SPOC")
            if sr is None or len(sr.table) == 0: raise RuntimeError("No light curves found.")
            lc = sr.download()
            if lc is None: raise RuntimeError("Download returned None (quota/network).")
            lc = lc.remove_outliers().flatten()
            time = lc.time.value.astype(float)
            flux = lc.flux.value.astype(float)
            st.success("Fetched and flattened real light curve.")
        except Exception as e:
            st.warning(f"Real-path failed: {e}. Falling back to synthetic.")
            lk = None

    if lk is None:
        cols = st.columns(5)
        with cols[0]:
            L = st.slider("Length", 512, 4096, 2048, 64)
        with cols[1]:
            period = st.slider("Injected period", 50, 600, 200, 5)
        with cols[2]:
            depth = st.slider("Depth", 0.0005, 0.01, 0.0025, 0.0005)
        with cols[3]:
            dur = st.slider("Duration (samples)", 5, 200, 40, 5)
        with cols[4]:
            jitter = st.slider("Noise (jitter)", 1e-4, 2e-3, 6e-4, step=1e-4, format="%.4f")
        time, flux = _synthetic_series(L=L, period=period, depth=depth, dur=dur, jitter=jitter)

    bestP, periods, sse = _bls_period(time, flux, minp=np.ptp(time)/400, maxp=np.ptp(time)/5, n=200)
    fit = _box_fit(time, flux, bestP)

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Best period (coarse BLS-like): **{bestP:.4f}**")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(periods, sse, '-'); ax.axvline(bestP, color='k', ls='--')
        ax.set_xlabel("Period"); ax.set_ylabel("SSE"); ax.set_title("Period scan (lower is better)")
        st.pyplot(fig)
    with c2:
        phase = (time % bestP)/bestP
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(phase, flux, ".", ms=2, alpha=0.6)
        ax2.set_xlabel("Phase"); ax2.set_ylabel("Flux"); ax2.set_title("Folded light curve")
        st.pyplot(fig2)

    # Residuals and SNR-ish metric
    phase = (time % bestP)/bestP
    inmask = ((phase >= fit['phi0']) & (phase <= (fit['phi0']+fit['width']) % 1.0)) if fit['phi0']+fit['width']<=1 else ((phase>=fit['phi0']) | (phase <= fit['phi0']+fit['width']-1))
    mu_out = fit['mu_out']
    model = np.where(inmask, mu_out - fit['depth'], mu_out)
    resid = flux - model
    snr = fit['depth']/max(np.std(resid), 1e-9)

    # Live metrics
    put_metric("exoplanet", "best_period", float(bestP))
    put_metric("exoplanet", "depth", float(fit['depth']))
    put_metric("exoplanet", "duration_frac", float(fit['width']))
    put_metric("exoplanet", "snr", float(snr))

    st.write(f"**Box fit**: depth≈{fit['depth']:.4e}, width≈{fit['width']:.3f}, SNR≈{snr:.2f}")
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(resid, "-", lw=0.7)
    ax3.set_title("Residuals"); ax3.set_xlabel("sample"); ax3.set_ylabel("flux - model")
    st.pyplot(fig3)

    with st.expander("Physics discussion (live)"):
        st.markdown(f"""
- Deeper transits (depth increases) and longer duty cycles (width increases) increase periodogram contrast; here SNR ~ depth / std(residual) = **{snr:.2f}**.
- Best period **{bestP:.4f}** sets the folding phase; aliasing arises if the search grid is coarse or noise is structured.
- The box model is the small-planet, no limb-darkening limit; analytic Mandel-Agol profiles would slightly change inferred depth.
""")
