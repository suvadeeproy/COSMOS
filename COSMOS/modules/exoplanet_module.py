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
        k = max(3, int(0.05*len(flux)))  # crude "in-transit" selection
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
    st.markdown("#### Exoplanets — BLS‑like search, box‑fit, residuals, SNR")
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
        ax.set_xlabel("Trial Period"); ax.set_ylabel("SSE")
        ax.set_title("Period scan: minima indicate candidate orbital periods")
        st.pyplot(fig)
    with c2:
        phase = (time % bestP)/bestP
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(phase, flux, ".", ms=2, alpha=0.6)
        ax2.set_xlabel("Phase"); ax2.set_ylabel("Flux")
        ax2.set_title("Folded light curve at best period")
        st.pyplot(fig2)

    phase = (time % bestP)/bestP
    inmask = ((phase >= fit['phi0']) & (phase <= (fit['phi0']+fit['width']) % 1.0)) if fit['phi0']+fit['width']<=1 else ((phase>=fit['phi0']) | (phase <= fit['phi0']+fit['width']-1))
    mu_out = fit['mu_out']
    model = np.where(inmask, mu_out - fit['depth'], mu_out)
    resid = flux - model
    snr = fit['depth']/max(np.std(resid), 1e-9)

    put_metric("exoplanet", "best_period", float(bestP))
    put_metric("exoplanet", "depth", float(fit['depth']))
    put_metric("exoplanet", "duration_frac", float(fit['width']))
    put_metric("exoplanet", "snr", float(snr))

    st.write(f"**Box fit**: depth≈{fit['depth']:.4e}, width≈{fit['width']:.3f}, SNR≈{snr:.2f}")
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(resid, "-", lw=0.7)
    ax3.set_title("Residuals after box model"); ax3.set_xlabel("sample"); ax3.set_ylabel("flux - model")
    st.pyplot(fig3)

    with st.expander("Physics + math discussion (plot‑by‑plot)"):
        st.markdown(f"""
**Signal model (what we’re actually doing)**  
We treat the light curve as a deterministic template + noise:
\[
f(t)=f_\mathrm{{out}}-d\,\mathbb{{1}}_\mathrm{{transit}}(t;P,\phi_0,w)+\epsilon(t).
\]
The “BLS-like” scan is a coarse matched filter over \(P\) for a box template.

---
**1) Period scan (SSE vs trial period)**  
We score candidate periods by
\[
\mathrm{{SSE}}(P)=\sum_i\big(f_i-\hat f_i(P)\big)^2,
\]
so **minima** indicate periods where the box template captures a repeating dip. In real survey analysis, BLS is close to a likelihood‑ratio test under Gaussian noise.

**Physics:** a stable minimum corresponds to an orbital clock; multiple minima often indicate aliases or harmonics.

---
**2) Folded light curve**  
Folding uses \(\varphi=(t\bmod P)/P\). A physical transit produces a phase‑localized deficit.  
Small-planet, no-limb-darkening approximation:
\[
d \approx \left(\frac{{R_p}}{{R_\star}}\right)^2.
\]
Duration is geometry + Kepler:
\[
a^3=\frac{{GM_\star}}{{4\pi^2}}P^2,\qquad
T_\mathrm{{dur}}\sim \frac{{P}}{{\pi}}\frac{{R_\star}}{{a}}\sqrt{{1-b^2}}.
\]
This is why the **duty cycle** \(w\approx T_\mathrm{{dur}}/P\) is informative about \(R_\star/a\) and impact parameter \(b\).

---
**3) Residuals**  
Residuals \(r_i=f_i-\hat f_i\) should be structureless if the template captures the signal. Coherent structure suggests missing physics:
limb darkening (curved transit), spots, flares, or detrending artifacts.
A quick detectability proxy:
\[
\mathrm{{SNR}}\approx \frac{{d}}{{\sigma_r}}={snr:.2f}.
\]

---
**What this teaches about the universe**  
This page is “inference from shadows”: gravity and orbital mechanics are being read out from a 1‑D time series. The plots are not decorations—they are *diagnostics* translating photometry into dynamical and geometric constraints.
""")
