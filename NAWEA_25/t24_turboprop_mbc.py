import numpy as np
import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.dyn_sys_Param import dynSysParametricRange
from stablib.models.model3_flap import cantilever_modes, mass, damping, stiffness, aero_damping, aero_stiffness, stack_matrices
# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

L = 1.5
n_modes = 1
n_blades = 3

rho = 1.225

U0 = 35.0
shear = U0 / L

omegas = np.linspace(10.0, 40.0, 100)

# ------------------------------------------------------------
# Blade properties
# ------------------------------------------------------------

def unit_mass(r):
    x = r / L
    return 35.0 * (1.0 - x**1.5)**1.2 + 4.0

def c(r):
    x = r / L
    return 2.0 * (1.0 - x**1.5)**1.5 + 0.05

def EI(r):

    x = r / L

    EI_tip = 1.0e3
    EI_root = 5.0e4

    return (
        EI_tip
        + (EI_root - EI_tip)
        * (1.0 - x)
        / (1.0 + 8.0 * x)
    )

def chord(r):
    x = r / L
    return 0.30 * (1.0 - x**1.3)**0.5 + 0.08

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

model = dynSysParametricRange(omegas, A_vector)

model.runMBCAnalyses([0,1,2])
model.sort_results()

print(f'\ndone')

import matplotlib.pyplot as plt
import numpy as np

omega_hz = omegas / (2 * np.pi)

f0 = model.q_of_interest['vf_0_sorted'][:, :, :]

plt.figure()

if f0.ndim == 1:
    plt.plot(omega_hz, f0, 'o-', label='f_0')
else:
    for i in range(f0.shape[1]):
        plt.plot(omega_hz, f0[:, i], 'o-', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('f_0 (Hz)')
plt.title('Campbell Diagram (f_0 vs Rotation Speed)')
plt.grid(True)
plt.legend()
plt.tight_layout()


zeta = model.q_of_interest['zeta_for_range_sorted'][:, :, :]

plt.figure()

if zeta.ndim == 1:
    plt.plot(omega_hz, zeta, 'o-', label='zeta')
else:
    for i in range(zeta.shape[1]):
        plt.plot(omega_hz, zeta[:, i], 'o-', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('Damping ratio')
plt.title('Damping Ratio vs Rotation Speed')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()