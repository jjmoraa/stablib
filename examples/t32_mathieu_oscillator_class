import numpy as np
from stablib.floquetParam import floquetParametric

# Define parameters for the Mathieu oscillator from Allen's paper.
m = 1.0
k0 = 1.0
k1 = 0.4
damp = 0.04
omega = 0.8  # [rad/s]

w0 = k0 / m  # =omega_0^2
w1 = k1 / m  # =omega_1^2
cc = damp / m  # =2*zeta*omega_0

# Lambda version of mathieu_a
mathieu_a = lambda t: np.array([[0.0, 1.0], [-w0 - w1 * np.cos(omega * t), -cc]])

# Lambda version of mathieu_b
mathieu_b = lambda m: np.array([[0], [1 / m]])

A = mathieu_a
omega = 1
mathieu_osc = floquetParametric(omega, A)
mathieu_osc.run_floquet_analysis(
    plotIVP=True,
    rtol=1e-4
)

# etc.
mathieu_c = np.array([[1, 0]])

At = lambda t:mathieu_a(t, w0, w1, cc, omega)

mathieu_osc.run_modal_projection(out_spec_matrix = mathieu_c, n_harmonics = 1)