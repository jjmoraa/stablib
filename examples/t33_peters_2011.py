import numpy as np
import matplotlib.pyplot as plt

# locals
from stablib.floquetParam import floquetParametricRange

# define epsilons of interest
epsilons = np.linspace(0, 3.5, 50) #should be (0.1, 1, 100)
A_vector = [None]*len(epsilons)
omega_0 = 1
omega = 1
for iep, epsilon in enumerate(epsilons):
    A_vector[iep] = lambda t, eps=epsilon: np.array([
        [0.0, 1.0],
        [-(omega_0**2 + eps * np.cos(omega * t)), 0.0]
    ])

peters_2011 = floquetParametricRange([omega]*len(epsilons), A_vector, param = epsilons, param_label=epsilons)
peters_2011.runAnalyses(out_spec_matrix = None, harmonics=10, rtol=1e-4)
peters_2011.sort_results()

import matplotlib.pyplot as plt

vf = peters_2011.q_of_interest["vf_0_sorted"]
pf = peters_2011.q_of_interest["participation_factor_for_range_sorted"]
x = peters_2011.param

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# --- Left: vf (dot plot) ---
for i in range(vf.shape[1]):
    ax1.scatter(x, vf[:, i, 0], s=12, alpha=0.7)

ax1.set_xlabel("Parameter")
ax1.set_ylabel("vf_0_sorted")
ax1.set_title("Modal frequencies (vf)")
ax1.grid(True)

# --- Right: pf (dot plot) ---
for i in range(pf.shape[1]):
    ax2.scatter(x, pf[:, i, 0], s=12, alpha=0.7)

ax2.set_xlabel("Parameter")
ax2.set_ylabel("Participation factor")
ax2.set_title("Participation factors")
ax2.grid(True)

plt.show()

print("done\n")