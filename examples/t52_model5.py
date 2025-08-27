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
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

from datetime import datetime
# Locals
from stablib.models.model5DOFs import mass, damping, stiffness
from state_space import A_fromMCK
from floquet import  solve,floquet_eigenanalysis, test_periodic
from modeProjection import mode_projection
# Define the mass, damping and stiffness matrices

# Define constants
m = 500
l = 30
M = 50000
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000
omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)
eigenvalues_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]
f_d_for_range=[]
f_0_for_range=[]
zeta_for_range=[]

#fA(t, omega)
#fnt(omega)

for iom, omega in enumerate(omegas):
    period=2*np.pi/omega
    base_points=5000
    min_points=1000
    #num_points = int(min_points + (base_points - min_points)  * (1-np.log(omega / omegas[0]) / np.log(omegas[-1] / omegas[0])))#check this oneliner
    time=np.linspace(0,period,2048)
    # mass_matrix=mass(m, l, omega, time)
    # damping_matrix=damping(omega, time)
    # stiffness_matrix=stiffness(edgNatFreq_rad, m, l, kx, ky, omega, time)
    # Functions of time
    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )
  
    test_periodic(At, period, tol=1e-3)
    sol=solve(At,time,plot=False)
    print('Solution is finished')
    eigenvalues = floquet_eigenanalysis(sol,time,omega, plot=False)
    eigenvalues_for_range.append(eigenvalues)

    omega_d = np.imag(eigenvalues)
    f_d = omega_d / (2*np.pi)
    omega_0 = np.abs(eigenvalues)
    f_0 = omega_0 / (2*np.pi)
    print('f0',f_0)
    zeta= -np.real(eigenvalues)/omega_0

    f_d_for_range.append(f_d)
    f_0_for_range.append(f_0)
    zeta_for_range.append(zeta)
    

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
plt.show()
# filename = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), f"campbell_ronnie.png")
# plt.savefig(filename)
plt.close()