# import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
# Locals
# import welib.essentials
from stablib.state_space import A_fromMCK
from stablib.floquet import  solve, test_periodic
from stablib.modeProjection import mode_projection
from stablib.ronnie import mass, damping, stiffness

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
