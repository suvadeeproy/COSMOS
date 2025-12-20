from typing import Tuple
import numpy as np

def mapie_regression_intervals(estimator, X_train, y_train, X_test, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformal intervals via MAPIE if available; otherwise NaNs gracefully."""
    try:
        from mapie.regression import MapieRegressor
        mr = MapieRegressor(estimator=estimator, cv="prefit")
        mr.fit(X_train, y_train)
        y_pred, y_pis = mr.predict(X_test, alpha=alpha)
        return y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0]
    except Exception:
        y_pred = estimator.predict(X_test)
        nan = np.full_like(y_pred, np.nan, dtype=float)
        return y_pred, nan, nan
