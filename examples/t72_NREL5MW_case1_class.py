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
omega_hz = omegas / (2 * np.pi)

# Extract results
f0 = nrel5MW.results['f_0'][:,2,:]  # assuming shape (n_omegas, ...)

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