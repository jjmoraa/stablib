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
from stablib.state_space import A_fromMCK
from stablib.floquet import  solve, floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection
from stablib.PostProcessing import plot_freq_heatmap

# Define the mass, damping and stiffness matrices

from t53_model5_riva import ro_riva


# --- Script parameters
RivaOnly=False
sanityChecks = True
sanityChecks = False
plotIVP = False
plotFloquet = False
plotModeProj = False
rtol=1e-4

def vectors_equal_up_to_sign(a, b, rtol=1e-5, atol=1e-8):
    return np.allclose(a, b, rtol=rtol, atol=atol) or np.allclose(a, -b, rtol=rtol, atol=atol)

def ro_riva_local(time_stm,At, C, rtol=1e-6):

    A0 = At(0)
    nx = A0.shape[0]
    tm0 = np.eye(nx)

    # Integrate Phi' = A @ Phi.
    sol_stm = solve_ivp(
        fun=lambda t, stm: (At(t) @ stm.reshape(nx, nx)).reshape(-1),
        t_span=(time_stm[0], time_stm[-1]),
        y0=tm0.reshape(-1),
        t_eval=time_stm,
        vectorized=True,
        rtol=rtol,
    )

    # Plot state transition matrix.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State transition matrix")
        ax.plot(sol_stm.t, sol_stm.y.T)

    # Reshape to 3D array.
    # The state transition matrix, stm, is ordered as:
    #  - axis 0, 1: state
    #  - axis 2: time
    stm = np.reshape(sol_stm.y, (nx, nx, sol_stm.t.size))

    # Get monodromy matrix.
    monodromy = stm[:, :, -1]

    # Compute characteristic multipliers.
    theta, S = eig(monodromy)

    # Plot characteristic multipliers.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_xlim(-1.1, +1.1)
        ax.set_ylim(-1.1, +1.1)
        ax.set_aspect("equal")
        ax.add_artist(plt.Circle(xy=(0.0, 0.0), radius=1.0, edgecolor="k", fill=False))
        for i in range(theta.size):
            ax.scatter(theta[i].real, theta[i].imag, label=f"Mode {i+1}")
        ax.legend(loc="center")

    # Define shift for characteristic exponents.
    if time_stm.size % 2 == 0:
        max_shift = int(time_stm.size / 2)

    else:
        max_shift = int((time_stm.size - 1) / 2)

    shift = np.arange(-max_shift, +max_shift)
    shift0 = max_shift  # n = 0.
    #print(f'T is of length({len(time_stm)})')
    #shift2 = np.fft.fftfreq(len(time_stm)-1)*(len(time_stm)-1)
    #shift3=shift2.copy()
    #shift3.sort()
    #print('')
    # Compute characteristic exponents.

    # eta is ordered as:
    #  - axis 0: harmonics.
    #  - axis 1: modes.
    abs_theta = np.abs(theta)
    ang_theta = np.angle(theta)
    eta = (np.log(abs_theta) + 1j * ang_theta)[np.newaxis, :] / period + 2j * np.pi / period * shift[:, np.newaxis]

    # Plot characteristic exponents.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Real [rad/s]")
        ax.set_ylabel("Imag [rad/s]")
        ax.set_xlim(-0.04, +0.02)
        ax.set_ylim(-3.2, +3.2)
        for i in range(eta.shape[1]):
            ax.scatter(eta[:, i].real, eta[:, i].imag, label=f"Mode {i}")
        ax.legend()

    # Compute frequency and damping.
    natural_frequency = np.abs(eta)  # [rad/s]
    damping_ratio = -eta.real / natural_frequency  # [-]
    natural_frequency /= 2 * np.pi  # [Hz]
    damped_frequency = eta.imag / (2 * np.pi)  # [Hz]. Can be positive or negative (useful later).

    # Set initial value of periodic transformation.
    P0 = np.eye(nx)

    # Compute eigenvectors of Floquet factor.
    V = S  # Trivial since P0 = I.

    # Compute Floquet factor.
    # R = V @ np.diag(eta[shift0, :]) @ inv(V)
    R = np.transpose(scipy.linalg.solve(V.T, np.diag(eta[shift0, :]) @ V.T))

    # Compute periodic transformation.
    # Time must be the first dimension to allow computing Xi using matmul.
    # P is periodic, therefore we skip the last time instant.
    P = np.zeros((time_stm.size - 1, nx, nx), dtype=complex)
    invS = inv(S)  # should use solve.

    for k in range(P.shape[0]):
        # P[k, :, :] = stm[:, :, k] @ P0 @ expm(- R * time_stm[k])
        # P[k, :, :] = stm[:, :, k] @ S @ np.diag(np.exp(-eta[shift0, :] * time_stm[k])) @ invS @ P0
        P[k, :, :] = stm[:, :, k] @ S * np.exp(-eta[shift0, :] * time_stm[k]) @ invS @ P0

    # Compute Xi.
    Xi = C[np.newaxis, :, :] @ P @ V[np.newaxis, :, :]

    # Expand Xi in Fourier series. psi contains the mode shapes, and is ordered as:
    # axis 0: harmonics.
    # axis 1: states or outputs if C is used.
    # axis 2: modes.
    psi = np.fft.fft(Xi, axis=0) / Xi.shape[0]

    # Sort harmonics from -n to +n.
    psi = np.fft.fftshift(psi, axes=0)

    #dt = time_stm[1] - time_stm[0]
    #freqs = np.fft.fftfreq(len(time_stm)-1, dt)
    #freqs = np.fft.fftshift(freqs) #shift the frequencies

    # Compute mode shapes norm.
    participation = np.linalg.norm(psi, ord=2, axis=1)

    # Normalize across the harmonics to get the output-specific participation factors.
    participation /= participation.sum(axis=0)

    # Find the principal harmonic for each mode.
    n_principal = np.argmax(participation, axis=0)



    f0_m1 = np.full(nx, np.nan)
    f0_principal = f0_m1.copy()
    f0_p1 = f0_m1.copy()
    for ix in range(nx):
        # We skip modes with negative damping frequency.
        if damped_frequency[n_principal[ix], ix] < 0.0:
            continue
        f0_m1[ix] = natural_frequency[n_principal[ix]-1, ix]
        f0_principal[ix] = natural_frequency[n_principal[ix], ix]
        f0_p1[ix] = natural_frequency[n_principal[ix]+1, ix]




    d = dict()
    d['R'] = R
    d['P'] = P
    d['V'] = V
    d['S'] = S
    d['theta'] = theta
    d['eta'] = eta
    d['monodromy'] = monodromy
    d['sol_stm'] = sol_stm
    #d['sol_free'] =sol_free
    d['Xi'] = Xi
    d['n_principal'] = n_principal
    d['damped_frequency'] = damped_frequency
    d['natural_frequency'] = natural_frequency
    d['psi'] = psi
    d['freqs'] = freqs
    d['participation'] = participation
    d['f_0_principal'] = f0_principal
    d['f_0_m1']        = f0_m1
    d['f_0_p1']        = f0_p1

    return d

# --- Define constants
m = 1000
l = 100
M = 200000
edgNatFreq_hz = 0.7  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 350000


# f_expected=[np.sqrt(kx/(M+3*m)) /(2*np.pi), edgNatFreq_hz]
f_expected=[edgNatFreq_hz]

omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)
# omegas = np.array([0.325]) #should be (0.1, 1, 100)


# Storage...
eigenvalues_for_range=[]
participation_factor_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]
zeta_for_range=[]
vf_d = []
vf_0 = []

vf_d_riva = []
vf_0_riva = []
vf_0_m1_riva = []
vf_0_p1_riva = []

for iom, omega in enumerate(omegas): #rads
    print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ------------------------')
    period=2*np.pi/omega # seconds
    # Figuring out Number of points per period
    base_points=5000
    min_points=1000
    #num_points = int(min_points + (base_points - min_points)  * (1-np.log(omega / omegas[0]) / np.log(omegas[-1] / omegas[0])))#check this oneliner
    time=np.linspace(0.0, period, 10001)
    # mass_matrix=mass(m, l, omega, time)
    # damping_matrix=damping(omega, time)
    # stiffness_matrix=stiffness(edgNatFreq_rad, m, l, kx, ky, omega, time)
    # Functions of time
    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )
    
    C = np.zeros((2, At(0).shape[0]))
    C = np.eye(10)
    #C[0, 3] = 1.0
    #C[1, 4] = 1.0
    with Timer('riva'):
        riva = ro_riva(time, At, C, rtol=rtol, period=period)
    # d['R'] = R
    # d['P'] = P
    # d['V'] = V
    # d['S'] = S
    # d['theta'] = theta
    # d['eta'] = eta
    # d['monodromy'] = monodromy
    # d['sol_stm'] = sol_stm
    # #d['sol_free'] =sol_free
    # d['Xi'] = Xi
    # d['n_principal'] = n_principal
    # d['damped_frequency'] = damped_frequency
    # d['natural_frequency'] = natural_frequency

    # get all Riva important comparisson parameters
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
    psi = riva['psi']
    participation = riva['participation']
    f_d_riva = riva['damped_frequency']
    f_0_riva = riva['natural_frequency']

    # Store
    vf_d_riva.append(f_d_riva)
    vf_0_riva.append(riva['f_0_principal'])
    vf_0_m1_riva.append(riva['f_0_m1'])
    vf_0_p1_riva.append(riva['f_0_p1'])
    #print('f0',f_0_riva, '(riva)')
    #import pdb; pdb.set_trace()


    if not RivaOnly:

        if sanityChecks:
            test_periodic(At, period, tol=1e-3)
        
        with Timer('solve-ivp'):
            sol=solve(At,time,plot=plotIVP, rtol=rtol)
        print('Solution is finished')

        if np.allclose(sol.y, sol_stm.y, atol=1e-3):
            print('Solution is close')
        else:
            print('Solution is NOT close')


        with Timer('floquet_eig'):
            [monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] = floquet_eigenanalysis(sol,time,omega, plot=plotFloquet, sanityChecks=sanityChecks)
        eigenvalues_for_range.append(eigenvalues_exp)
        
        # CHECKS against riva
        if np.allclose(monodromy, monodromy_riva, atol=1e-3):
            print('[ OK ] monodromy is close')
        else:
            print('[ FAIL ] monodromy is NOT close')

        if np.allclose(eigenvalues_mon, theta, atol=1e-3):
            print('[ OK ] mon eigenvalues is close')
        else:
            print('[ FAIL ] mon eigenvalues is NOT close')

        if np.allclose(eigenvectors_mon, V, atol=1e-3):
            print('[ OK ] mon eigenvectors is close')
        else:
            print('[ FAIL ] mon eigenvectors is NOT close')

        if np.allclose(eigenvalues_exp, eta[5000,:], atol=1e-1):
            print('[ OK ] exp eigenvalues is close')
        else:
            print('[ FAIL ] exp eigenvalues is NOT close')

        SS = eigenvectors_exp

        # Code snippet to test Q components against B (From GPT)
        # import matplotlib.pyplot as plt, numpy as np; 
        # T=min(q_values.shape[0],P.shape[0]); t=np.arange(T); i,j=0,0
        # plt.plot(t,q_values[:T,i,j].real,'b-',label="q real"); 
        # plt.plot(t,q_values[:T,i,j].imag,'b--',label="q imag"); 
        # plt.plot(t,P[:T,i,j].real,'r-',label="P real"); 
        # plt.plot(t,P[:T,i,j].imag,'r--',label="P imag"); 
        # plt.xlabel("time"); plt.ylabel("value"); plt.legend(); plt.grid(True); plt.show()


        if np.allclose(q_values[:-1,:,:], P, atol=1e-3):
            print('[ OK ] q values is close - P for RIVA - ')
        else:
            print('[ FAIL ] exp q values is NOT close - P for RIVA -')

        with Timer('mode_proj'):
            [max_vals, max_index, participation_factor, basis, out_spec_basis, fourier_coefficients, participation_factor, freqs] = mode_projection(C, q_values, eigenvectors_exp, time, plot=plotModeProj, sanityChecks=sanityChecks)


        # iy=0; ix=0; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=0; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=1; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=1; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=2; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=2; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=3; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=3; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=4; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=4; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=5; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=5; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=6; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=6; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=7; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=7; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=8; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=8; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=0; ix=9; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()
        # iy=1; ix=9; plt.figure(); plt.plot(freqs, fourier_coefficients[:,iy, ix]); plt.plot(freqs_riva, psi[:,iy, ix], '--'); plt.show()

        if vectors_equal_up_to_sign(out_spec_basis, Xi_riva):
            print('[ OK ] Output specific basis is close')
        else:
            print('[ FAIL ] Output specific basis is NOT close')

        if vectors_equal_up_to_sign(fourier_coefficients, psi): #this has an issue
            print('[ OK ] fourier coefficients is close')
        else:
            print('[ FAIL ] fourier coefficients is NOT close')

        if vectors_equal_up_to_sign(participation_factor, participation):
            print('[ OK ] participation factor is close')
        else:
            print('[ FAIL ] participation factor is NOT close')

        if vectors_equal_up_to_sign(max_index, n_principal-5000):
            print('[ OK ] max index is close')
        else:
            print('[ FAIL ] max index is NOT close')

        eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)

        # if vectors_equal_up_to_sign(eigenvalues_exp_corrected, n_principal-5000):
        #     print('[ OK ] strongest harmonic is close')
        # else:
        #     print('strongest harmonic is NOT close')

        # Compute frequency and damping.
        # natural_frequency = np.abs(eigenvalues_exp_corrected)  # [rad/s]
        # damping_ratio = -eigenvalues_exp_corrected.real / natural_frequency  # [-]
        # natural_frequency_hz = natural_frequency * 2 * np.pi  # [Hz]
        # damped_frequency = np.abs(eigenvalues_exp_corrected.imag) / (2 * np.pi)  # [Hz]

        # plot_freq_heatmap(participation_factor)
        participation_factor_for_range.append(participation_factor)

        omega_d = np.imag(eigenvalues_exp_corrected)
        f_d = omega_d / (2*np.pi)
        omega_0 = np.abs(eigenvalues_exp_corrected)
        f_0 = omega_0 / (2*np.pi)
        print('f0',f_0)
        zeta= -np.real(eigenvalues_exp_corrected)/omega_0

        vf_d.append(f_d)
        vf_0.append(f_0)
        zeta_for_range.append(zeta)
        

#plot_freq_heatmap(participation_factor_for_range)

# After the loop, you can plot Campbell diagram
# Convert omegas to Hz for x-axis
freqs_Hz = omegas / (2 * np.pi)


# Iterate over the modes by using a list of lists (since we have appended modes)
fig, ax = plt.subplots(figsize=(8, 5))
for mode_idx in range(len(vf_0_riva[0])):  # number of modes in each result
    mode_freqs_riva = [vf_0_riva[i][mode_idx] for i in range(len(vf_0_riva))]
    ax.plot(freqs_Hz, mode_freqs_riva, 'o', label=f'Mode {mode_idx + 1}')
    ax.set_xlabel('Rotor speed [Hz]')
    ax.set_ylabel('Modal frequency [Hz]')
    ax.set_title('Campbell Diagram (Floquet)')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
for f0 in f_expected:
    ax.plot(freqs_Hz, f0+freqs_Hz, 'k--')
    ax.plot(freqs_Hz, f0-freqs_Hz, 'k--')
plt.tight_layout()

if not RivaOnly:

    # --- Plot Campbell
    fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)

    COLRS = plt.rcParams['axes.prop_cycle'].by_key()['color']


    #fig, ax = plt.subplots(figsize=(8, 5))
    for mode_idx in range(len(vf_0[0])):  # number of modes in each result
        mode_freqs = np.array([vf_0[i][mode_idx] for i in range(len(vf_0))])
        b1 = np.logical_and(mode_freqs<0.17, mode_freqs>0.1 )
        b2 = np.logical_and(mode_freqs<0.23, mode_freqs>0.18 )
        b3 = np.logical_and(mode_freqs<1.0, mode_freqs>0.6 )
        ax.plot(freqs_Hz[b1], mode_freqs[b1], 'o', c=COLRS[0]) #, label=f'Mode {mode_idx + 1}')
        ax.plot(freqs_Hz[b2], mode_freqs[b2], 'o', c=COLRS[1])#, label=f'Mode {mode_idx + 1}')
        ax.plot(freqs_Hz[b3], mode_freqs[b3], 'o', c=COLRS[2])# label=f'Mode {mode_idx + 1}')
    ax.set_xlabel('Rotor speed [Hz]')
    ax.set_ylabel('Modal frequency [Hz]')
    ax.set_title('Campbell Diagram (Floquet)')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    for f0 in f_expected:
        ax.plot(freqs_Hz, f0+freqs_Hz, 'k--')
        ax.plot(freqs_Hz, f0-freqs_Hz, 'k--')
    #plt.tight_layout()
    ax.set_ylim([-0.05,0.9])
    fig.savefig('Floquet.pdf')
# filename = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), f"t53_model5_Campbell.png")
# scriptDir = os.path.dirname(os.path.abspath(__file__))
# plt.savefig(os.path.join(scriptDir, f"t53_model5_Campbell_ltest.png"))

plt.show()
plt.close()
