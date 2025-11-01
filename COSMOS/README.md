
# COSMOS — Scientific ML Workbench

**Author**: Suvadeep Roy (2025)  
**Purpose**: A single, demonstrable, research-leaning Streamlit app that shows mastery of
- ML for astrophysical signals (exoplanet transits),
- ML surrogates for string/M-theory inspired landscape problems,
- Holographic / AdS/QCD soft-wall numerics,
with proper diagnostics (ROC, PR, ECE, calibration) and **transparent assumptions**.

This repo is intentionally kept **ethically clean**:
- No proprietary datasets are redistributed.
- External data are fetched at runtime from public endpoints.
- Every physics module states its scope and limitations in the UI.

## Modules

1. **Exoplanets**
   - Synthetic box-transit generator.
   - 1D-CNN classifier (PyTorch).
   - Diagnostics: ROC, PR, AP, reliability/ECE, MC-dropout.
   - Based on ideas in Mandel & Agol (2002) for realistic transits (not bundled).

2. **Landscape surrogates**
   - Toy CICY-like feature generator.
   - Gradient boosting regressors for (h^{1,1}, h^{2,1}, chi).
   - Error histograms + permutation importance for interpretability.
   - Inspired by Candelas–Lynker–Schimmrigk (1988), but **not** a replacement for real cohomology codes.

3. **Holography / AdS-QCD**
   - Parametric soft-wall fit m_n^2 ≈ 4k(n+1).
   - Sturm–Liouville finite-difference solver for V(z)=k^2 z^2 + 3/(4 z^2).
   - Spectrum + eigenfunctions plotted.
   - Based on Karch et al. (2006).

## License
The code in this repository is released under the MIT License (see `LICENSE`).  
Python dependencies retain their own licenses.

## Citing
See `docs/REFERENCES.md` for APA7 citations.
