import numpy as np
import matplotlib.pyplot as plt

# locals

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models import model5DOFs_mass_imbalance
from stablib.state_space import mbc
from stablib.floquet import  solve, test_periodic

# --- Define constants
m = 1000
l = 100
M = 200000
imbalance = 0 #unit %
edgNatFreq_hz = 0.7  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 350000

# define rotation speeds of interest
omegas = np.linspace(0.1, 1, 10)

A_vector = [None] * len(omegas)
A_mbc_vector = [None] * len(omegas)

T1 = [None] * len(omegas)
T2 = [None] * len(omegas)
T3 = [None] * len(omegas)

for iom, omega in enumerate(omegas):

    Mt = lambda t, omega=omega: model5DOFs_mass_imbalance.mass(M, m, imbalance, l, omega, t)
    Kt = lambda t, omega=omega: model5DOFs_mass_imbalance.stiffness(edgNatFreq_rad, m, imbalance, l, kx, ky, omega, t)
    Ct = lambda t, omega=omega: model5DOFs_mass_imbalance.damping(m, imbalance, l, omega, t)

    A_vector[iom] = lambda t, Mt=Mt, Ct=Ct, Kt=Kt: stab.state_space.A_fromMCK(Mt(t), Ct(t), Kt(t))
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


# ---------------------------------------------------------
# Campbell diagram
# ---------------------------------------------------------

n_modes = None
campbell_freqs = []
campbell_damping = []

for iom, omega in enumerate(omegas):

    # Evaluate the (time-invariant) MBC matrix
    A_LTI = A_mbc_vector[iom](0.0)

    eigvals, eigvecs = np.linalg.eig(A_LTI)

    # Keep only one eigenvalue from each conjugate pair
    eigvals = eigvals[np.imag(eigvals) >= 0]

    sigma = np.real(eigvals)
    wd = np.imag(eigvals)

    wn = np.sqrt(sigma**2 + wd**2)
    fn = wn / (2*np.pi)          # Hz
    zeta = -sigma / wn

    # Sort by frequency
    idx = np.argsort(fn)

    fn = fn[idx]
    zeta = zeta[idx]

    if n_modes is None:
        n_modes = len(fn)

    campbell_freqs.append(fn)
    campbell_damping.append(zeta)

campbell_freqs = np.array(campbell_freqs)
campbell_damping = np.array(campbell_damping)

rotor_freq = omegas / (2*np.pi)

# ---------------------------------------------------------
# Campbell Diagram
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

for mode in range(n_modes):
    plt.plot(
        rotor_freq,
        campbell_freqs[:, mode],
        '-o',
        label=f'Mode {mode+1}'
    )

# 1P line
plt.plot(
    rotor_freq,
    rotor_freq,
    'k--',
    linewidth=2,
    label='1P'
)

plt.xlabel("Rotor Frequency (Hz)")
plt.ylabel("Natural Frequency (Hz)")
plt.title("Campbell Diagram")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Damping ratios
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

for mode in range(n_modes):
    plt.plot(
        rotor_freq,
        campbell_damping[:, mode],
        '-o',
        label=f'Mode {mode+1}'
    )

plt.xlabel("Rotor Frequency (Hz)")
plt.ylabel("Damping Ratio")
plt.title("Modal Damping")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()