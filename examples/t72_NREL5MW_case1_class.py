from pathlib import Path
from stablib import openfast
from stablib.openfast import turbine
foldername = Path('/home/jjmoraa/work/python_libs/stablib/stablib/models/Land NREL 5MW 8DOF no wind')

nrel5MW = turbine(foldername)
nrel5MW.runAnalyses(out_spec_matrix = None, harmonics=3, rtol=1e-2)

omegas = nrel5MW.omegas
#model_5dof.index_off - results
print(f'\ndone')

import matplotlib.pyplot as plt
import numpy as np

# Convert omega from rad/s to Hz
omega_RPM = 60 * omegas / (2 * np.pi)

# Extract results
physical_dofs = int(nrel5MW.results['f_0'].shape[-1]/2);
zeroth_frequency = int(round(nrel5MW.results['f_0'].shape[1]/2));
f0 = nrel5MW.results['f_0'][:,zeroth_frequency,:]  # assuming shape (n_omegas, n_harmonics, n_dof)
f0_flat = f0.reshape(f0.shape[0], -1)  # shape = (n_omegas, n_harmonics * n_dof)

'''

Discuss with Branlard

'''
plt.figure(figsize=(8, 6))
for i in range(f0_flat.shape[1]):
    plt.plot(omega_RPM, f0_flat[:, i], 'o-', label=f'mode {i+1}')

plt.xlabel('Rotation speed (Hz)')
plt.ylabel('f_0 (Hz)')
plt.title('Campbell Diagram (f_0 vs Rotation Speed)')
plt.grid(True)
plt.legend(ncol=2, fontsize=8)  # Adjust if too many modes
plt.tight_layout()
plt.show()