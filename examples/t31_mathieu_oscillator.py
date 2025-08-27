# 
# import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
 
from datetime import datetime

# Locals
# import welib.essentials
from stablib.state_space import A_fromMCK
from stablib.floquet import  solve,floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection
from stablib.PostProcessing import plot_freq_heatmap
# Super local
from t31_mathieu_oscillator_riva import mo_riva



# --- Script parameters 
sanityChecks = True



# 
def vectors_equal_up_to_sign(a, b, rtol=1e-5, atol=1e-8):
    return np.allclose(a, b, rtol=rtol, atol=atol) or np.allclose(a, -b, rtol=rtol, atol=atol)


# Define parameters for the Mathieu oscillator from Allen's paper.
m = 1.0
k0 = 1.0
k1 = 0.4
damp = 0.04
omega = 0.8  # [rad/s]

period = 2 * np.pi / omega  # [s]
w0 = k0 / m  # =omega_0^2
w1 = k1 / m  # =omega_1^2
cc = damp / m  # =2*zeta*omega_0

time=np.linspace(0.0, period, 2001)

# Define matrices for the Mathieu oscillator.
def mathieu_a(t, w0, w1, cc, W):
    return np.array([[0.0, 1.0], [-w0 - w1 * np.cos(W * t), -cc]])

def mathieu_b(m):
    return np.array([[0], [1 / m]])

riva = mo_riva()
monodromy_riva=riva['monodromy']
R = riva['R']
P = riva['P']
V = riva['V']
S = riva['S']
sol_stm = riva['sol_stm']
theta = riva['theta']
eta = riva['eta']
Xi_riva = riva['Xi']
n_principal = riva['n_principal']

# etc.
mathieu_c = np.array([[1, 0]])

At = lambda t:mathieu_a(t, w0, w1, cc, omega)
Ct = lambda t: mathieu_c

test_periodic(At, period, tol=1e-3)

# Integrate solution for linear system
sol=solve(At,time,plot=True)
print('Solution is finished')

if np.allclose(sol.y, sol_stm.y, atol=1e-3):
    print('[ OK ] Solution is close')
else:
    print('Solution is NOT close')

# Perform floquet analysis
[monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] = floquet_eigenanalysis(sol,time,omega, plot=True, sanityChecks=sanityChecks)

# CHECKS against riva
if np.allclose(monodromy, monodromy_riva, atol=1e-3):
    print('[ OK ] monodromy is close')
else:
    print('monodromy is NOT close')

if np.allclose(eigenvalues_mon, theta, atol=1e-3):
    print('[ OK ] mon eigenvalues is close')
else:
    print('mon eigenvalues is NOT close')

if np.allclose(eigenvectors_mon, V, atol=1e-3):
    print('[ OK ] mon eigenvectors is close')
else:
    print('mon eigenvectors is NOT close')

if np.allclose(eigenvalues_exp, eta[1000,:], atol=1e-1):
    print('[ OK ] exp eigenvalues is close')
else:
    print('WARNING: exp eigenvalues is NOT close')

if np.allclose(eigenvectors_exp, S, atol=1e-3):
    print('[ OK ] exp eigenvectors is close')
else:
    print('exp eigenvectors is NOT close')

# Define output specific factor

C = mathieu_c

# Perform modal projection on exponent matrix

[max_vals,max_index,participation_factor] = mode_projection(C, q_values, eigenvectors_mon, time, plot=True, sanityChecks=sanityChecks)

if vectors_equal_up_to_sign(max_index, n_principal-[1000,1000]):
    print('[ OK ] strongest harmonic is close')
else:
    print('strongest harmonic is NOT close')

    # eigenvalues=lambda_real+1j*(om_p + max_index*omega)
 
    # omega_d = np.imag(eigenvalues)
    # f_d = omega_d / (2*np.pi)
    # omega_0 = np.abs(eigenvalues)
    # f_0 = omega_0 / (2*np.pi)
    # zeta= -np.real(eigenvalues)/omega_0
    # print('f0',f_0)

#eigenvalues_for_range.append(eigenvalues)
# plot_freq_heatmap(participation_factor)
#participation_factor_for_range.append(participation_factor)
