import numpy as np
from modules.ads_qcd_module import solve_fd

def test_spectrum_monotone():
    _, vals, _ = solve_fd(0.25, 8.0, 400, 6)
    assert np.all(np.diff(vals) > 0)

def test_convergence_groundstate():
    _, v1, _ = solve_fd(0.25, 8.0, 300, 3)
    _, v2, _ = solve_fd(0.25, 8.0, 800, 3)
    assert abs(v2[0] - v1[0]) / v2[0] < 0.05
