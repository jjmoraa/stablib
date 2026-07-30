import numpy as np

# locals

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models import model5DOFs_mass_imbalance, model5DOFs_mass_imbalance_freq
from stablib.state_space import mbc
from stablib.floquet import  solve, test_periodic

# --- Define constants
m = 1000
l = 100
M = 200000
m_imbalance = 0.35 #unit %
f_imbalance = 0.05 #unit %
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

    Mt = lambda t, omega=omega: model5DOFs_mass_imbalance.mass(M, m, m_imbalance, l, omega, t)
    Kt = lambda t, omega=omega: model5DOFs_mass_imbalance_freq.stiffness(edgNatFreq_rad, m, m_imbalance,f_imbalance, l, kx, ky, omega, t)
    Ct = lambda t, omega=omega: model5DOFs_mass_imbalance.damping(m, m_imbalance, l, omega, t)

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

