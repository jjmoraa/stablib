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
from stablib.modeProjection import mode_projection, mode_projection_multiple_harmonics
from stablib.PostProcessing import plot_freq_heatmap, evaluateStabilityMonodromy, plotCampbellDiagram, plotCampbellDiagramMultipleHarmonics
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
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]
zeta_for_range=[]
vf_d = []
vf_0 = []


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
    # C[0, 3] = 1.0
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
        [max_vals, max_index, max_participation_factor, basis, out_spec_basis, fourier_coefficients, participation_factor, freqs] = mode_projection_multiple_harmonics(C, q_values, eigenvectors_exp, time, n_harmonics, plot=False, sanityChecks=False)
    # correct index with strongest frequency
    eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)

    # save participation factors
    participation_factor_for_range.append(max_participation_factor)
    
    # compute natural, damped frquencies and damping
    f_d, f_0, zeta = computeDamping(eigenvalues_exp_corrected)

    # save frequencies and damping AND MODES
    mode_shapes.append(eigenvectors_exp)
    vf_d.append(f_d)
    vf_0.append(f_0)
    zeta_for_range.append(zeta)

# select just first harmonic
# vf_0_plot_1 = [vf[0, :] for vf in vf_0]  # pick the first harmonic for each rotor speed
# vf_0_plot_2 = [vf[1, :] for vf in vf_0]  # pick the first harmonic for each rotor speed
# vf_0_plot_3 = [vf[2, :] for vf in vf_0]  # pick the first harmonic for each rotor speed

# create an assignment list for modes after sorting them using MAC
sorted_modes, assignment_list = mac_sort_modes(mode_shapes)

# make np arrays
vf_d = np.array(vf_d)
vf_0 = np.array(vf_0)
zeta_for_range = np.array(zeta_for_range)
participation_factor_for_range = np.array(participation_factor_for_range)

for i in range(vf_0.shape[1]):
    print('Sorting quantities for mode ', i+1)
    # sort other quantities
    vf_d_sorted = reorder_parameters_by_assignment(vf_d[:, i, :], assignment_list)
    vf_0_sorted = reorder_parameters_by_assignment(vf_0[:, i, :], assignment_list)
    zeta_for_range_sorted = reorder_parameters_by_assignment(zeta_for_range[:, i, :], assignment_list)
    participation_factor_for_range_sorted = reorder_parameters_by_assignment(participation_factor_for_range[:, i, :], assignment_list)

    # After the loop, you can plot Campbell diagram
    # Convert omegas to Hz for x-axis
    freqs_Hz = omegas / (2 * np.pi)

    plotCampbellDiagram(vf_0_sorted, freqs_Hz, 'natural frequency [Hz]', save_path=None)
    # plotCampbellDiagram(vf_0_plot_1, freqs_Hz, save_path='Campbell1.pdf')
    # plotCampbellDiagram(vf_0_plot_2, freqs_Hz, save_path='Campbell2.pdf')
    # plotCampbellDiagram(vf_0_plot_3, freqs_Hz, save_path='Campbell3.pdf')

plotCampbellDiagramMultipleHarmonics(vf_d_sorted, freqs_Hz, y_label='Modal frequency [Hz]', save_path='Campbell1.pdf')
plotCampbellDiagramMultipleHarmonics(zeta_for_range_sorted, freqs_Hz, y_label='Damping ratio [-]', save_path='Campbell1.pdf')
plotCampbellDiagramMultipleHarmonics(participation_factor_for_range_sorted, freqs_Hz, y_label='Participation factor [-]', save_path='Campbell1.pdf')