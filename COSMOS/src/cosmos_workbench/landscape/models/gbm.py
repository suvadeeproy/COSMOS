
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

@dataclass
class GBMModels:
    reg_h11: GradientBoostingRegressor
    reg_h21: GradientBoostingRegressor
    cls_chi_sign: GradientBoostingClassifier

def make_models(seed=0) -> GBMModels:
    return GBMModels(
        GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3),
        GradientBoostingRegressor(random_state=seed+1, n_estimators=200, max_depth=3),
        GradientBoostingClassifier(random_state=seed+2, n_estimators=200, max_depth=3),
    )

def fit(models: GBMModels, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray) -> GBMModels:
    models.reg_h11.fit(X, y_reg[:,0])
    models.reg_h21.fit(X, y_reg[:,1])
    models.cls_chi_sign.fit(X, y_cls)
    return models

def predict(models: GBMModels, X: np.ndarray):
    h11 = models.reg_h11.predict(X)
    h21 = models.reg_h21.predict(X)
    chi = 2*(h11-h21)
    p = models.cls_chi_sign.predict_proba(X)[:,1]
    return (np.stack([h11,h21,chi], axis=1), p)
