
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def sigmoid(z): return 1/(1+np.exp(-z))

def reliability_curve(p, y, bins=15):
    edges = np.linspace(0,1,bins+1)
    idx = np.digitize(p, edges)-1
    conf, acc = [], []
    for b in range(bins):
        m = idx==b
        if m.sum()==0:
            conf.append(np.nan); acc.append(np.nan); continue
        conf.append(p[m].mean()); acc.append(y[m].mean())
    return np.array(conf), np.array(acc), edges

def ece(p, y, bins=15):
    conf, acc, edges = reliability_curve(p, y, bins)
    e = 0.0
    for b in range(bins):
        m = (p>=edges[b])&(p<edges[b+1]); w = m.mean()
        if w>0: e += w*abs(conf[b]-acc[b])
    return float(e)

def temp_scale(logits, y):
    def nll(T):
        z = logits/max(T[0],1e-4); p = sigmoid(z)
        eps=1e-7
        return -np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
    r = minimize(nll, x0=[1.0], bounds=[(0.05,10.0)])
    return float(r.x[0])
