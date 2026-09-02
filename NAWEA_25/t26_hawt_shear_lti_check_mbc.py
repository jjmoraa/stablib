import numpy as np

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models import model5DOFs_mass_imbalance, model5DOFs_mass_imbalance_freq
from stablib.mbc import mbc
from stablib.floquet import  solve, test_periodic

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.dyn_sys_Param import dynSysParametricRange
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

omegas = np.linspace(0.2, 1.5, 10)

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
A_mbc_vector = [None] * len(omegas)

T1 = [None] * len(omegas)
T2 = [None] * len(omegas)
T3 = [None] * len(omegas)

for iom, omega in enumerate(omegas):
    A_vector[iom] = lambda t, omega=omega: stab.state_space.A_fromMCK(
    *stack_matrices(
        M_blade,
        C_blade,
        K_blade,
        aero_damping(modes, rho, chord, L, omega, t, U0, shear),
        aero_stiffness(modes, rho, chord, L, omega, t, U0, shear)
        ))
    A_mbc_vector [iom] , T1[iom] , T2[iom] , T3[iom] = mbc(A_vector[iom], omega, [0, 1, 2], debug = True)

    '''    
    # Pick a few azimuth positions over one revolution
    times = np.linspace(0, 2*np.pi/omega, 5)

    worked yay!!!!

    M_mbc = []
    M_mbc_list = []
    for t in times:

        T = T1[iom](t)
        M_matrix = Mt(t)
        perm = [3, 4, 0, 1, 2]
        M_perm = M_matrix[np.ix_(perm, perm)]
        M_matrix = M_perm
        # Your current multiplication
        M_right = M_matrix @ T

        # Full transformed mass matrix
        M_mbc = T.T @ M_matrix @ T

        print("\n======================")
        print(f"t = {t}")
        print("T1:")
        print(T)

        print("\nOriginal M:")
        print(M_matrix)

        print("\nM @ T:")
        print(M_right)

        print("\nT.T @ M @ T:")
        print(M_mbc)

        M_mbc_list.append(M_mbc)

    # Compare all transformed matrices against the first one
    for i in range(1, len(M_mbc)):
        error = np.max(np.abs(M_mbc[i] - M_mbc[0]))
        print(f"Time {times[i]:.3f}: error = {error}")
    '''

A_LTI = A_mbc_vector[iom](0.0)
eigvals, eigvecs = np.linalg.eig(A_LTI)

# Keep only one eigenvalue from each conjugate pair
eigvals = eigvals[np.imag(eigvals) >= 0]

sigma = np.real(eigvals)
wd = np.imag(eigvals)                 # damped angular frequency (rad/s)

wn = np.sqrt(sigma**2 + wd**2)        # undamped angular frequency (rad/s)
fn = wn / (2*np.pi)                   # natural frequency (Hz)

zeta = -sigma / wn                    # damping ratio

# Sort by frequency
idx = np.argsort(fn)
fn = fn[idx]
zeta = zeta[idx]
eigvals = eigvals[idx]

print(f"{'Mode':>4} {'Eigenvalue':>24} {'Freq (Hz)':>12} {'Damping':>12}")
print("-"*60)

for i, (lam, f, z) in enumerate(zip(eigvals, fn, zeta), start=1):
    print(f"{i:4d} {lam:24.6f} {f:12.4f} {z:12.5f}")

# PLot the matrices
Nt = 100
t_vec = np.linspace(0, 2*np.pi/omegas[-1], Nt, endpoint=False)
A_mbc_time = np.stack([A_mbc_vector[-1](t) for t in t_vec], axis=0)

import matplotlib.pyplot as plt

n = A_mbc_time[0].shape[1]

plt.figure(figsize=(10, 6))

for i in range(n):
    for j in range(n):
        plt.plot(
            t_vec,
            np.real(A_mbc_time[:, i, j]),
            alpha=0.4
        )

plt.xlabel("Azimuth / time (rad)")
plt.ylabel("Re(A_mbc entries)")
plt.title("All MBC matrix coefficients over one period")
plt.grid(True)
plt.show()

print('done\n')

