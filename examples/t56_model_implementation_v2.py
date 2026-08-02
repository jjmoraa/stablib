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

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import eig, inv
from scipy.signal import welch

# Locals
from stablib.tictoc import Timer
from stablib.models.model5DOFs import mass, damping, stiffness
from stablib.state_space import A_fromMCK, computeDamping, mac_sort_modes, reorder_parameters_by_assignment
from stablib.floquet import  solve, floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection, mode_projection_multiple_harmonics, mode_projection_multiple_harmonics_v2
from stablib.PostProcessing import plot_freq_heatmap, evaluateStabilityMonodromy, plotCampbellDiagram, plotCampbellDiagramAllModesSingleHarmonic, plotCampbellDiagramMultipleHarmonics
# Define the mass, damping and stiffness matrices

# --- Script parameters
RivaOnly=False
sanityChecks = True
sanityChecks = False
plotIVP = False
plotFloquet = False
plotModeProj = False
rtol=1e-4

# --- Define constants
m = 1000
l = 100
M = 200000
edgNatFreq_hz = 0.7  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 350000

# define rotation speeds of interest
omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)

# Storage...
eigenvalues_for_range=[]
mode_shapes = [] #corresponds to the eigenvectors of the exponent matrix
participation_factor_for_range=[]
max_index_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]
zeta_for_range=[]
vf_d = []
vf_0 = []
eigenvalues_exp_corrected_for_range = []

for iom, omega in enumerate(omegas): #rads

    print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ------------------------')
    period=2*np.pi/omega # seconds
    num_points=10001 #give odd number to get even number in fft (very important)
    time=np.linspace(0.0, period, num_points)
    
    # Functions of time
    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )
    
    # Time domain solution of the x_dot=Ax system
    with Timer('solve-ivp'):
        sol=solve(At,time,plot=plotIVP, rtol=rtol)
    print('Solution is finished')

    with Timer('floquet_eig'):
        [monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] = floquet_eigenanalysis(sol,time,omega, plot=plotFloquet, sanityChecks=sanityChecks)
        eigenvalues_for_range.append(eigenvalues_exp)
    print('Floquet eigenanalysis is finished')

    stabilityMon = evaluateStabilityMonodromy(eigenvalues_mon, doPlot=False)
    SS = eigenvectors_exp
    
        
    C = np.zeros((2, At(0).shape[0]))
    C = np.eye(10)
    C[0, 0] = 1.0
    # C[1, 4] = 1.0
    # with Timer('mode_proj'):
    #     [max_vals, max_index, participation_factor, basis, out_spec_basis, fourier_coefficients, participation_factor, freqs] = mode_projection(C, q_values, eigenvectors_exp, time, plot=plotModeProj, sanityChecks=sanityChecks)

    # # correct index with strongest frequency
    # eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)

    # # save participation factors
    # participation_factor_for_range.append(participation_factor)
    
    # # compute natural, damped frquencies and damping
    # f_d, f_0, zeta = computeDamping(eigenvalues_exp_corrected)

    # # save frequencies and damping
    # vf_d.append(f_d)
    # vf_0.append(f_0)
    # zeta_for_range.append(zeta)
        
    n_harmonics=3
    with Timer('mode_proj'):
        [max_vals, max_index, participation_factor, basis, out_spec_basis, fourier_coefficients, participation_factor, freqs, ifreq0] = mode_projection_multiple_harmonics_v2(C, q_values, eigenvectors_exp, time, n_harmonics, plot=False, sanityChecks=False)
    # correct index with strongest frequency
    eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)
    eigenvalues_exp_corrected_for_range.append(eigenvalues_exp_corrected)
    # save participation factors and max indices
    participation_factor_for_range.append(participation_factor)
    max_index_for_range.append(max_index)
    # compute natural, damped frquencies and damping
    # f_d, f_0, zeta = computeDamping(eigenvalues_exp_corrected)

    # save frequencies and damping AND MODES
    # ---- NO ----- do it after you found all the harmonics of interest
    mode_shapes.append(eigenvectors_exp)
    # vf_d.append(f_d)
    # vf_0.append(f_0)
    # zeta_for_range.append(zeta)

eigenvalues_exp_corrected_for_range = np.array(eigenvalues_exp_corrected_for_range)
participation_factor_for_range = np.array(participation_factor_for_range)
max_index_for_range = np.array(max_index_for_range)
off_indices = max_index_for_range - max_index_for_range[:, 0, :][:, None, :]

all_modes = off_indices.reshape(off_indices.shape[0], -1, order='F')
# unique_by_mode = [np.unique(all_modes[i]) for i in range(all_modes.shape[0])]
# Use np.unique with return_index
unique_indices, idx = np.unique(all_modes, return_index=True)
unique_indices = unique_indices[np.argsort(idx)]

eigenvalues_unique = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
f_d                 =np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
f_0=np.zeros_like(f_d)
zeta=np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
pf_index_unique=np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]))

for iom, omega in enumerate(omegas): #rads
    #eigenvalues_unique = eigenvalues_exp_corrected_for_range[0]+1j*(unique_by_mode)*omega
    for iind in range(len(unique_indices)):
        eigenvalues_unique [iom, iind, :] = eigenvalues_exp_corrected_for_range[iom, 0, :] + 1j * np.tile(unique_indices[None, :] * omega, (10, 1))[:,iind]
        pf_index_unique[iom, iind, :] = max_index_for_range[iom,0,:] + unique_indices[iind]+ifreq0
    f_d[iom], f_0[iom], zeta[iom] = computeDamping(eigenvalues_unique[iom,:,:])
#f_d2, f_02, zeta2 =  computeDamping(eigenvalues_unique)

n_omega, n_harmonics, n_modes = participation_factor_for_range.shape

# fancy indexing along axis=1 (harmonics)
pf_of_interest = participation_factor_for_range[
    np.arange(n_omega)[:, None, None],       # omega axis
    pf_index_unique.astype(int),             # harmonics axis
    np.arange(n_modes)[None, None, :]        # modes axis
]
# np.ix_
# pf_of_interest2 = participation_factor_for_range[np.ix(np.arange(n_omega) ,  pf_index_unique.astype(int),  np.arange(n_modes) ) ]

# pf_of_interest = participation_factor_for_range[:,np.int32(pf_index_unique[:,:])]
# select just first harmonic
# vf_0_plot_1 = [vf[0, :] for vf in vf_0]  # pick the first harmonic for each rotor speed
# vf_0_plot_2 = [vf[1, :] for vf in vf_0]  # pick the first harmonic for each rotor speed
# vf_0_plot_3 = [vf[2, :] for vf in vf_0]  # pick the first harmonic for each rotor speed

# vf_0, vf_d, zeta_for_range, participation_factor_for_range shapes: (n_omega, n_harmonics, n_modes)

'''
We are goiing to separate into two workflows:
1. First take all the participation factors
2. Also save all the unique indices for the strongest frequency contributions
3. Select only the unique frequencies, record the principal harmonic of course
4. On then! you construct everything else thats missing
'''
# convert into np arrays
# vf_0 = np.array(vf_0)
# vf_d =  np.array(vf_d)
# zeta_for_range = np.array(zeta_for_range)

# select the main frequency, principal or first harmonic
# vf_0_main=f_0[:,0,:]
# vf_d_main=f_d[:,0,:]
# zeta_for_range_main=zeta[:,0,:]
# participation_factor_for_range_main=pf_of_interest[:,0,:]

# sort the modes
mode_shapes_sorted, assignment_array = mac_sort_modes(mode_shapes, use_macx=False, debug=False)

# sort the first harmonic
vf_0_sorted = np.zeros_like(f_0[:, :, :])
vf_d_sorted = np.zeros_like(f_d[:, :, :])
zeta_for_range_sorted = np.zeros_like(zeta[:, :, :])
participation_factor_for_range_sorted = np.zeros_like(pf_of_interest[:, :, :])

for iind in range(len(unique_indices)):
    vf_0_sorted[:, iind, :] = reorder_parameters_by_assignment(f_0[:,iind,:], assignment_array)
    vf_d_sorted[:, iind, :] = reorder_parameters_by_assignment(f_d[:,iind,:], assignment_array)
    zeta_for_range_sorted[:, iind, :] = reorder_parameters_by_assignment(zeta[:,iind,:], assignment_array)
    participation_factor_for_range_sorted[:, iind, :] = reorder_parameters_by_assignment(pf_of_interest[:,iind,:], assignment_array)


# Convert omegas to Hz for x-axis
freqs_Hz = omegas / (2 * np.pi)
# build secondary modes

# # Select 2nd and 3rd harmonics
# vf_0_secondary = vf_0[:, 1:3, :]  # shape: (n_omega, 2, n_dof)
# vf_d_secondary = vf_d[:, 1:3, :]  # shape: (n_omega, 2, n_dof)
# zeta_for_range_secondary = zeta_for_range[:, 1:3, :]  # shape: (n_omega, 2, n_dof)
# participation_factor_for_range_secondary = participation_factor_for_range[:, 1:3, :]  # shape: (n_omega, 2, n_dof)

# # Collapse harmonics and DOFs into a single axis
# vf_0_secondary = vf_0_secondary.reshape(vf_0_secondary.shape[0], -1)
# vf_d_secondary = vf_d_secondary.reshape(vf_d_secondary.shape[0], -1)
# zeta_for_range_secondary = zeta_for_range_secondary.reshape(zeta_for_range_secondary.shape[0], -1)
# participation_factor_for_range_secondary = participation_factor_for_range_secondary.reshape(participation_factor_for_range_secondary.shape[0], -1)
# # Now shape: (n_omega, 2 * n_dof)

# check which modes start with the same zero frequency

'''
Remember:

The indexing convention is:
[omegas, harmonics, modes]

'''


'''original plots
plotCampbellDiagram(vf_0_sorted[:,0,:], freqs_Hz, 'natural frequency [Hz]', save_path=None)
plotCampbellDiagram(vf_d_sorted[:,0,:], freqs_Hz, 'natural frequency [Hz]', save_path=None)
plotCampbellDiagram(zeta_for_range_sorted[:,0,:], freqs_Hz, 'natural frequency [Hz]', save_path=None)

'''

'''
# Select the mode you want, e.g., mode index 1
pf_mode = participation_factor_for_range_sorted[:, :, 1]  # shape (n_omega, n_harmonics)

# Normalize each column (frequency) independently
pf_mode_norm = pf_mode / np.max(pf_mode, axis=1, keepdims=True)

# Now you can plot
plotCampbellDiagram(pf_mode_norm, freqs_Hz, 'natural frequency [Hz]', save_path=None)
'''


'''
YAAAY IT WORKED

'''

relevant_modes = int(len(mode_shapes)/2)

# All modes, principal harmonic (0)
plotCampbellDiagramAllModesSingleHarmonic(vf_0_sorted[:,:,0:(relevant_modes)],freqs_Hz,0,var_name='Frequency [Hz]',save_path=None)
plotCampbellDiagramAllModesSingleHarmonic(zeta_for_range_sorted[:,:,0:(relevant_modes)],freqs_Hz,0,var_name='Frequency [Hz]',save_path=None)

# All modes, 1st harmonic (1)
plotCampbellDiagramAllModesSingleHarmonic(vf_0_sorted[:,:,0:(relevant_modes)],freqs_Hz,1,var_name='Damping',save_path=None)
plotCampbellDiagramAllModesSingleHarmonic(zeta_for_range_sorted[:,:,0:(relevant_modes)],freqs_Hz,1,var_name='Damping',save_path=None)

#Fan of modes
mod_indices = [0, 1, 3]
for i in mod_indices:
    # --- Natural frequency Campbell plot ---
    plotCampbellDiagram(
        vf_0_sorted[:, :, i],
        freqs_Hz,
        f'natural frequency [Hz] Mode {i+1}',
        save_path=None
    )

    # --- Participation factor Campbell plot ---
    pf_mode = participation_factor_for_range_sorted[:, :, i]
    pf_mode_norm = pf_mode / np.sum(pf_mode, axis=1, keepdims=True)

    plotCampbellDiagram(
        pf_mode_norm,
        freqs_Hz,
        f'participation factor (normalized) Mode {i+1}',
        save_path=None
    )

print('Analysis done')