import numpy as np
def mapie_regression_intervals(estimator, Xtr, ytr, Xte, alpha=0.05):
    try:
        from mapie.regression import MapieRegressor
        mr = MapieRegressor(estimator=estimator, cv='prefit')
        mr.fit(Xtr, ytr)
        yp, pis = mr.predict(Xte, alpha=alpha)
        return yp, pis[:,0,0], pis[:,1,0]
    except Exception:
        yp = estimator.predict(Xte)
        nan = np.full_like(yp, np.nan, dtype=float)
        return yp, nan, nan
