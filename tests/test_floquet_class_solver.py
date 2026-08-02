#import unittest
import numpy as np

from stablib.floquetParam import floquetParametric


def test_floquet_sine_cosine():

    # ------------------------------------------------------------------
    # Problem definition
    # ------------------------------------------------------------------

    At = lambda t: np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t),  np.cos(t)]
    ])

    omega = 1.0

    system = floquetParametric(omega, At)

    system.run_floquet_analysis(
        plotIVP=False,
        rtol=1e-4
    )

    # ------------------------------------------------------------------
    # Numerical solution
    # ------------------------------------------------------------------

    t = system.results["STM"].t
    y = system.results["STM"].y

    # ------------------------------------------------------------------
    # Analytical solution
    # ------------------------------------------------------------------

    theta = 1.0 - np.cos(t)
    scale = np.exp(np.sin(t))

    y_exact = np.zeros_like(y)

    y_exact[0] = scale * np.cos(theta)
    y_exact[1] = -scale * np.sin(theta)
    y_exact[2] = scale * np.sin(theta)
    y_exact[3] = scale * np.cos(theta)

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    err = y - y_exact

    rmse = np.sqrt(np.mean(err**2))
    max_abs = np.max(np.abs(err))
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(y_exact)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    assert rmse < 1e-4
    assert max_abs < 5e-4
    assert rel_l2 < 1e-4

    # Floquet multipliers should lie inside/on the unit circle
    eig = system.results["eigenvalues_mon"]

    assert np.all(np.abs(eig) <= 1.0 + 1e-8)

#if __name__ == '__main__':
#    unittest.main()