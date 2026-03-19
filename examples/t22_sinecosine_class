import numpy as np
from stablib.floquetParam import floquetParametric

# Define a matrix
At = lambda t: np.array([[np.cos(t), -np.sin(t)], 
                         [np.sin(t), np.cos(t)]])

A = At
omega = 1
sine_cosine = floquetParametric(omega, A)
sine_cosine.run_floquet_analysis(
    plotIVP=True,
    rtol=1e-4
)
print('done\n')