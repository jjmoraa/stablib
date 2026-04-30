import numpy as np
import matplotlib.pyplot as plt
# locals

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models import model5DOFs_mass_imbalance_freq, model5DOFs_mass_imbalance, model5DOFs
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
omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)
'''
A_vector = [None]*len(omegas)

for iom, omega in enumerate(omegas):

    Mt = lambda t, omega=omega: model5DOFs.mass(M, m, l, omega, t)
    Kt = lambda t, omega=omega: model5DOFs.stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t, omega=omega: model5DOFs.damping(omega, t)

    A_vector[iom] = lambda t, Mt=Mt, Ct=Ct, Kt=Kt: stab.state_space.A_fromMCK(Mt(t), Ct(t), Kt(t))

model_5dof = floquetParametricRange(omegas, A_vector)
model_5dof.runAnalyses(out_spec_matrix = None, harmonics=3, rtol=1e-4)
model_5dof.sort_results()
'''
A_vector = [None]*len(omegas)

for iom, omega in enumerate(omegas):

    Mt = lambda t, omega=omega: model5DOFs_mass_imbalance.mass(M, m, m_imbalance, l, omega, t)
    Kt = lambda t, omega=omega: model5DOFs_mass_imbalance_freq.stiffness(edgNatFreq_rad, m, m_imbalance,f_imbalance, l, kx, ky, omega, t)
    Ct = lambda t, omega=omega: model5DOFs_mass_imbalance.damping(m, m_imbalance, l, omega, t)

    A_vector[iom] = lambda t, Mt=Mt, Ct=Ct, Kt=Kt: stab.state_space.A_fromMCK(Mt(t), Ct(t), Kt(t))

model_5dof_imb = floquetParametricRange(omegas, A_vector)
model_5dof_imb.runAnalyses(out_spec_matrix = None, harmonics=3, rtol=1e-4)
model_5dof_imb.sort_results()


f0 = model_5dof_imb.q_of_interest['vf_0_sorted'][:,:,:]
zeta = model_5dof_imb.q_of_interest['zeta_for_range_sorted'][:,:,:]

# --- styling (LaTeX-like) ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "legend.frameon": False,
})

omega_hz = omegas / (2 * np.pi)

blade0_idx = [0]
blade12_idx = [9]

colors = {
    "blade0": "#567EB4",   # desaturated blue-gray
    "blade12": "#375F43"   # desaturated green-gray
}

# better semantic mapping
linestyles = ['--', '-', ':']  # 0P, +1P, -1P

# =======================
# FREQUENCY PLOT
# =======================
plt.figure()

# --- Blade 0 ---
for h in range(f0.shape[1]):
    plt.plot(
        omega_hz,
        f0[:, h, blade0_idx[0]],
        linestyle=linestyles[h % len(linestyles)],
        color=colors["blade0"],
        linewidth=3,
        label=r'$\boldsymbol{\mathit{Blade\ 0}}$' if h == 1 else None
    )

# --- Blades 1 & 2 ---
for i, idx in enumerate(blade12_idx):
    for h in range(f0.shape[1]):
        plt.plot(
            omega_hz,
            f0[:, h, idx],
            linestyle=linestyles[h % len(linestyles)],
            color=colors["blade12"],
            linewidth=3,
            alpha=0.8,
            label=r'$\boldsymbol{\mathit{Blades\ 1\ &\ 2}}$' if (i == 0 and h == 1) else None
        )

plt.xlabel(r'$\boldsymbol{\mathit{Rotation\ speed\ \Omega\ (Hz)}}$')
plt.ylabel(r'$\boldsymbol{\mathit{Frequency\ f\ (Hz)}}$')
# plt.title(r'$\mathit{Blade\ Frequencies}$')
plt.tight_layout()

plt.legend()
# =======================
# DAMPING PLOT
# =======================
plt.figure()

# --- Blade 0 ---
for h in range(zeta.shape[1]):
    plt.plot(
        omega_hz,
        zeta[:, h, blade0_idx[0]],
        linestyle=linestyles[h % len(linestyles)],
        color=colors["blade0"],
        linewidth=3,
        label=r'$\boldsymbol{\mathit{Blade\ 0}}$' if h == 1 else None
    )

# --- Blades 1 & 2 ---
for i, idx in enumerate(blade12_idx):
    for h in range(zeta.shape[1]):
        plt.plot(
            omega_hz,
            zeta[:, h, idx],
            linestyle=linestyles[h % len(linestyles)],
            color=colors["blade12"],
            linewidth=3,
            alpha=0.8,
            label=r'$\boldsymbol{\mathit{Blades\ 1\ &\ 2}}$' if (i == 0 and h == 1) else None
        )

plt.xlabel(r'$\boldsymbol{\mathit{Rotation\ speed\ \Omega\ (Hz)}}$')
plt.ylabel(r'$\boldsymbol{\mathit{Damping\ ratio\ \zeta}}$')
# plt.title(r'$\mathit{Blade\ Damping}$')
plt.tight_layout()

plt.legend()
plt.show()