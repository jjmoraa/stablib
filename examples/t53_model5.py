'''
x'=q'*y+q*y'
sub in
x'=ax
q'*y+q*y'=ax
also recall x=qy
q'*y+q*y'=a*q*y
q'*y=a*q*y-q*y'
q'*y=q*(a*y-y')
q'=q*(a-b)

Remember to use the monondromy matrix condition
Psi(t,0)'=[A^-1*B]Psi(t,0)

Initial condition must always be I for q
'''
# 
import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
from datetime import datetime

# Locals
from stablib.tictoc import Timer
from stablib.models.model5DOFs import mass, damping, stiffness
from stablib.state_space import A_fromMCK
from stablib.floquet import  solve, floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection
from stablib.PostProcessing import plot_freq_heatmap
# Define the mass, damping and stiffness matrices


# --- Script parameters
sanityChecks = False
plotIVP = False
plotFloquet = False
plotModeProj = False

# --- Define constants
m = 500
l = 30
M = 50000
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000


omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)


# Storage...
eigenvalues_for_range=[]
participation_factor_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]
f_d_for_range=[]
f_0_for_range=[]
zeta_for_range=[]

for iom, omega in enumerate(omegas):
    print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ------------------------')
    period=2*np.pi/omega
    # Figuring out Number of points per period
    base_points=5000
    min_points=1000
    #num_points = int(min_points + (base_points - min_points)  * (1-np.log(omega / omegas[0]) / np.log(omegas[-1] / omegas[0])))#check this oneliner
    num_points = 256
    time=np.linspace(0, period, num_points)
    # Functions of time
    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )
  
    if sanityChecks:
        test_periodic(At, period, tol=1e-3)

    with Timer('solve-ivp'):
        sol=solve(At,time,plot=plotIVP)
    C = np.zeros((1, int(np.sqrt(sol.y.shape[0]))))
    C[0, 4] = 1

    with Timer('floquet_eig'):
        [monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] = floquet_eigenanalysis(sol,time,omega, plot=plotFloquet, sanityChecks=sanityChecks)
    eigenvalues_for_range.append(eigenvalues_exp)
    
    with Timer('mode_proj'):
        [max_vals,max_index,participation_factor] = mode_projection(C, q_values, eigenvectors_mon, time, plot=plotModeProj, sanityChecks=sanityChecks)

    # plot_freq_heatmap(participation_factor)
    participation_factor_for_range.append(participation_factor)

    omega_d = np.imag(eigenvalues_exp)
    f_d = omega_d / (2*np.pi)
    omega_0 = np.abs(eigenvalues_exp)
    f_0 = omega_0 / (2*np.pi)
    print('f0',f_0)
    zeta= -np.real(eigenvalues_exp)/omega_0

    f_d_for_range.append(f_d)
    f_0_for_range.append(f_0)
    zeta_for_range.append(zeta)
    

#plot_freq_heatmap(participation_factor_for_range)

# After the loop, you can plot Campbell diagram
# Convert omegas to Hz for x-axis
freqs_Hz = omegas / (2 * np.pi)

fig, ax = plt.subplots(figsize=(8, 5))

# Iterate over the modes by using a list of lists (since we have appended modes)
for mode_idx in range(len(f_0_for_range[0])):  # number of modes in each result
    mode_freqs = [f_0_for_range[i][mode_idx] for i in range(len(f_0_for_range))]
    ax.plot(freqs_Hz, mode_freqs, 'o', label=f'Mode {mode_idx + 1}')

ax.set_xlabel('Rotor speed [Hz]')
ax.set_ylabel('Modal frequency [Hz]')
ax.set_title('Campbell Diagram (Floquet)')
ax.grid(True)
ax.legend(loc='best', fontsize='small')
plt.tight_layout()
# filename = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), f"t53_model5_Campbell.png")
scriptDir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(scriptDir, f"t53_model5_Campbell.png"))


plt.show()
plt.close()
