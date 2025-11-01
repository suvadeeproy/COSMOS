
# Third-party and external resources

This repository is designed to run ONLY on openly available, free resources.

Data/endpoints used at runtime (no data is redistributed):
1. **NASA Exoplanet Archive**: accessed via public API at runtime to fetch small CSV tables.
   - NASA Exoplanet Archive. (2013). *Confirmed Planets Table*.
     IPAC, NExScI, Caltech. Retrieved from NASA Exoplanet Archive API at runtime.
   - Please see https://exoplanetarchive.ipac.caltech.edu for terms of use.

2. **CICY list (external link only, not redistributed)**:
   - Candelas, P., Lynker, M., & Schimmrigk, R. (1988).
     *Calabi–Yau manifolds in weighted P(4)*, Nucl. Phys. B 306, 113.
   - Oxford CICY database: https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/

Python packages (installed via `pip install -r requirements.txt`):
- Streamlit (Apache 2.0)
- NumPy (BSD)
- Pandas (BSD)
- scikit-learn (BSD)
- SciPy (BSD)
- Matplotlib (PSF-style)
- PyTorch (BSD-style)
- requests (Apache 2.0)
- optuna (MIT)

These remain under THEIR OWN LICENSES and are not re-licensed by this repository.
