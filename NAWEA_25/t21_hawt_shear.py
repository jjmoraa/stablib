import numpy as np
import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models.model3_flap import cantilever_modes, mass, damping, stiffness, aero_damping, aero_stiffness, stack_matrices
# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

L = 117.0
n_modes = 1
n_blades = 3

rho = 1.225

U0 = 10.0
shear = 0.2

omegas = np.linspace(0.1, 1.5, 60)

# ------------------------------------------------------------
# Blade properties
# ------------------------------------------------------------

def unit_mass(r):
    x = r / L
    return 3200.0 * (1.0 - x**1.8)**1.2 + 150.0

def c(r):
    x = r / L
    return 0.15 * (1.0 - x**1.5)**1.5 + 0.01

def EI(r):
    x = r / L
    return 1.0e3 + (1.5e11 - 1.0e3) / (1.0 + 1.0e8 * x**8)

def chord(r):
    x = r / L
    return 5.8 * (1.0 - x**1.7)**0.7 + 1.0

# ------------------------------------------------------------
# Mode shapes and diffs
# ------------------------------------------------------------

modes = cantilever_modes(n_modes, L)
def diff(f, h=1e-5):
    return lambda r: (f(r + h) - f(r - h)) / (2 * h)

modes = cantilever_modes(n_modes, L)

modes_1 = [diff(mode) for mode in modes]
modes_2 = [diff(diff(mode)) for mode in modes]

# ------------------------------------------------------------
# Single-blade structural matrices
# ------------------------------------------------------------

M_blade = mass(modes, unit_mass, L)
C_blade = damping(modes, c, L)
K_blade = stiffness(modes_2, EI, L)

# ------------------------------------------------------------
# Time-dependent matrices
# ------------------------------------------------------------

A_vector = [None] * len(omegas)
for iom, omega in enumerate(omegas):
    A_vector[iom] = lambda t, omega=omega: stab.state_space.A_fromMCK(
    *stack_matrices(
        M_blade,
        C_blade,
        K_blade,
        aero_damping(modes, rho, chord, L, omega, t, U0, shear),
        aero_stiffness(modes, rho, chord, L, omega, t, U0, shear)
    )
)

model = floquetParametricRange(omegas, A_vector)

model.runAnalyses(
    out_spec_matrix=None,
    harmonics=1,
    rtol=1e-5
)

model.sort_results()
print(f'\ndone')

import matplotlib.pyplot as plt
import numpy as np

omega_hz = omegas / (2 * np.pi)

f0 = model.q_of_interest['vf_0_sorted'][:, :, :]

plt.figure()

if f0.ndim == 1:
    plt.plot(omega_hz, f0, 'o', label='f_0')
else:
    for i in range(f0.shape[1]):
        plt.plot(omega_hz, f0[:, i], 'o', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('f_0 (Hz)')
plt.title('Campbell Diagram (f_0 vs Rotation Speed)')
plt.grid(True)
plt.legend()
plt.tight_layout()


zeta = model.q_of_interest['zeta_for_range_sorted'][:, :, :]

plt.figure()

if zeta.ndim == 1:
    plt.plot(omega_hz, zeta, 'o', label='zeta')
else:
    for i in range(zeta.shape[1]):
        plt.plot(omega_hz, zeta[:, i], 'o', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('Damping ratio')
plt.title('Damping Ratio vs Rotation Speed')
plt.grid(True)
plt.legend()
plt.tight_layout()


pf = model.q_of_interest['participation_factor_for_range_sorted'][:, :, :]

plt.figure()

if pf.ndim == 1:
    plt.plot(omega_hz, pf, 'o', label='zeta')
else:
    for i in range(pf.shape[1]):
        plt.plot(omega_hz, pf[:, i], 'o', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('Participation factors')
plt.title('Participation Factors vs Rotation Speed')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()