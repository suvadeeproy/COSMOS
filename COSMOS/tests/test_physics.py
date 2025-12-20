import numpy as np
from modules.ads_qcd_module import solve_fd
def test_monotone():
    _, vals, _, _ = solve_fd("soft-wall", 0.25, 8.0, 400, 6, 8.0)
    assert np.all(np.diff(vals) > 0)
def test_convergence():
    _, v1, _, _ = solve_fd("soft-wall", 0.25, 8.0, 350, 3, 8.0)
    _, v2, _, _ = solve_fd("soft-wall", 0.25, 8.0, 800, 3, 8.0)
    assert abs(v2[0] - v1[0]) / v2[0] < 0.08
