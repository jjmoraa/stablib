import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
### --- Functions for A(t) with state vector taken as a matrix and flattened out
def ode_system_Ax_flat(x, A):
    """
    Defines the ODE system x' = A x.

    Parameters:
    - t (float): Time (unused because A is time-independent).
    - x (numpy.ndarray): Flattened state vector.
    - a_matrix (numpy.ndarray): Precomputed A matrix.

    Returns:
    - numpy.ndarray: Flattened derivative of x at time t.
    """
    size = int(np.sqrt(x.size))  # Determine the size of the square matrix
    x_matrix = x.reshape(size, size)  # Reshape the state into a matrix
    dxdt_matrix = A @ x_matrix  # Compute A * x(t), where x is a matrix
    return dxdt_matrix.flatten()  # Flatten to return as a vector

def solve_ode_At_flat(At, x0, t_values):
    """
    Solves the ODE system x'(t) = A(t) x(t) using solve_ivp.

    Parameters:
    - a_matrix (numpy.ndarray): Precomputed A matrix.
    - x0 (numpy.ndarray): Initial state vector (flattened).
    - t_values (numpy.ndarray): Time points at which to evaluate the solution.

    Returns:
    - sol: Solution object returned by solve_ivp.
    """
    sol = solve_ivp(lambda t, x: ode_system_Ax_flat(x, At(t)), [t_values[0], t_values[-1]], x0, t_eval=t_values)
    return sol


def A_fromMCK(M, C, K):
    mass_inv = np.linalg.inv(M)
    
    return np.block([
        [np.zeros_like(M), np.eye(M.shape[0])],
        [-mass_inv @ K, -mass_inv @ C]])

