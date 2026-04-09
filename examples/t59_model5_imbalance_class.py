import numpy as np

# locals

import stablib as stab
from stablib.floquetParam import floquetParametricRange
from stablib.models import model5DOFs_mass_imbalance
# --- Define constants
m = 1000
l = 100
M = 200000
imbalance = 0.01 #unit %
edgNatFreq_hz = 0.7  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 350000

# define rotation speeds of interest
omegas = np.linspace(0.1, 1, 10) #should be (0.1, 1, 100)
A_vector = [None]*len(omegas)

for iom, omega in enumerate(omegas):

    Mt = lambda t, omega=omega: model5DOFs_mass_imbalance.mass(M, m, imbalance, l, omega, t)
    Kt = lambda t, omega=omega: model5DOFs_mass_imbalance.stiffness(edgNatFreq_rad, m, imbalance, l, kx, ky, omega, t)
    Ct = lambda t, omega=omega: model5DOFs_mass_imbalance.damping(m, imbalance, l, omega, t)

    A_vector[iom] = lambda t, Mt=Mt, Ct=Ct, Kt=Kt: stab.state_space.A_fromMCK(Mt(t), Ct(t), Kt(t))

model_5dof = floquetParametricRange(omegas, A_vector)
model_5dof.runAnalyses(out_spec_matrix = None, harmonics=3, rtol=1e-4)

#model_5dof.index_off - results
print(f'\ndone')

import matplotlib.pyplot as plt
import numpy as np

# Convert omega from rad/s to Hz
omega_hz = omegas / (2 * np.pi)

# Extract results
f0 = model_5dof.results['f_0'][:,2,:]  # assuming shape (n_omegas, ...)

# If f0 is 1D → simple plot
if f0.ndim == 1:
    plt.plot(omega_hz, f0, 'o-', label='f_0')

# If f0 is 2D (e.g., multiple DOFs or modes)
else:
    for i in range(f0.shape[1]):
        plt.plot(omega_hz, f0[:, i], 'o-', label=f'mode {i}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('f_0 (Hz)')
plt.title('Campbell Diagram (f_0 vs Rotation Speed)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()