# import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
# Locals
# import welib.essentials
from stablib.state_space import A_fromMCK
from stablib.floquet import  solve, test_periodic
from stablib.modeProjection import mode_projection

# Define time period
period=2*np.pi
time=np.linspace(0,period,1000)

# Define a matrix
At = lambda t: np.array([[np.cos(t), -np.sin(t)], 
                         [np.sin(t), np.cos(t)]])

test_periodic(At, period, tol=1e-3)

# get the eigenvalues using floquet solve
sol=solve(At,time,plot=True)
print('Solution is finished')
# there's no eigenvalues in this problem
# eigenvalues = floquetsolve(At,time)



''' for conference plot

import matplotlib.pyplot as plt

# solver output
t = sol.t          # (N,)
y = sol.y          # (4, N)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.plot(t, y[i, :], linewidth=3)   # thick blue lines (default color)
    ax.set_title(fr"$x_{i+1}$", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5)

# axis labels
for ax in axes[2:]:
    ax.set_xlabel("Time [s]", fontsize=11)

for ax in axes[::2]:
    ax.set_ylabel("Amplitude", fontsize=11)

fig.suptitle("State Transition matrix time histories", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

'''