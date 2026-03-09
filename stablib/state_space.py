import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

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

def solve_ode_At_flat(At, x0, t_values, rtol=1e-6):
    """
    Solves the ODE system x'(t) = A(t) x(t) using solve_ivp.

    Parameters:
    - a_matrix (numpy.ndarray): Precomputed A matrix.
    - x0 (numpy.ndarray): Initial state vector (flattened).
    - t_values (numpy.ndarray): Time points at which to evaluate the solution.

    Returns:
    - sol: Solution object returned by solve_ivp.
    """
    sol = solve_ivp(lambda t, x: ode_system_Ax_flat(x, At(t)), [t_values[0], t_values[-1]], x0, t_eval=t_values, vectorized=True, rtol=rtol)

    # sol_stm = solve_ivp(
    #     fun=lambda t, stm: (At(t) @ stm.reshape(nx, nx)).reshape(-1),
    #     t_span=(time_stm[0], time_stm[-1]),
    #     y0=tm0.reshape(-1),
    #     t_eval=time_stm,
    #     vectorized=True,
    #     rtol=1e-6,
    # )


    return sol


def A_fromMCK(M, C, K):
    # Original approach with explicit inversion:
    # mass_inv = np.linalg.inv(M)
    # return np.block([
    #     [np.zeros_like(M), np.eye(M.shape[0])],
    #     [-mass_inv @ K, -mass_inv @ C]])

    # Direct solve approach (preferred):
    minus_massinv_K = -np.linalg.solve(M, K)
    minus_massinv_C = -np.linalg.solve(M, C)
    return np.block([
        [np.zeros_like(M), np.eye(M.shape[0])],
        [minus_massinv_K, minus_massinv_C]])

def computeDamping(eigenvalues):
    """
    Computes damped/undamped frequencies and damping ratios from Floquet exponents.

    Parameters
    ----------
    eigenvalues_exp : array-like
        Floquet exponents (corrected)

    Returns
    -------
    f_d : ndarray
        Damped frequencies (Hz)
    f_0 : ndarray
        Undamped frequencies (Hz)
    zeta : ndarray
        Damping ratios
    """
    eigenvalues = np.array(eigenvalues)
    omega_d = np.imag(eigenvalues)
    f_d = omega_d / (2*np.pi)
    omega_0 = np.abs(eigenvalues)
    f_0 = omega_0 / (2*np.pi)
    zeta = -np.real(eigenvalues) / omega_0

    print("f0:", f_0)
    return f_d, f_0, zeta

def calculate_mac_matrix(mode_prev, mode_next):
    """
    This is a function to calculate the MAC matrix for two modes

    The matrix is like              abs(mode1^H*mode2)^2
                                    --------------------
                                (mode1^H*mode2)*(mode2^H*mode2)

    The closest to 1 it is the more "similar are both modes    
    
    Parameters
    ----------
    mode_prev : array-like
    mode_next : array-like

    Returns
    -------
    MAC = float
    """
    
    # phi_prev, phi_curr shape: (m, n)
    n_modes = mode_prev.shape[1]
    MAC = np.zeros((n_modes, n_modes))

    for i in range(n_modes):
        for j in range(n_modes):
            num = np.abs(np.vdot(mode_prev[:, i], mode_next[:, j]))**2
            denom = (np.vdot(mode_prev[:, i], mode_prev[:, i]) *
                    np.vdot(mode_next[:, j], mode_next[:, j]))  # FIXED
            MAC[i, j] = num / denom
    return MAC



def mac_sort_modes(mode_shapes, use_macx=False, debug=False):
    """
    MAC-based sorting of mode shapes across frequencies using MACX or standard MAC.
    Hungarian assignment ensures unique mode pairing.

    Parameters
    ----------
    mode_shapes : list of np.ndarray
        List of mode shapes at each frequency. Each element has shape (m, n),
        where m = number of state variables, n = number of modes.
    use_macx : bool
        If True, use MACX for complex modes. Otherwise, use standard MAC.
    debug : bool
        If True, plot MAC matrix for each frequency step.

    Returns
    -------
    sorted_modes_array : np.ndarray
        Mode shapes sorted across frequencies, shape (n_freqs, m, n_modes).
    assignment_array : np.ndarray
        Assignment indices used at each frequency, shape (n_freqs, n_modes)
    """
    n_freqs = len(mode_shapes)
    m, n_modes = mode_shapes[0].shape

    # initialize
    sorted_modes_list = [mode_shapes[0]]  # first frequency stays
    assignment_array = np.zeros((n_freqs, n_modes), dtype=int)
    assignment_array[0, :] = np.arange(n_modes)  # first frequency assignment

    # loop over frequencies
    for i in range(1, n_freqs):
        phi_prev = sorted_modes_list[-1]  # previous frequency
        phi_curr = mode_shapes[i]         # current frequency

        # Compute MAC matrix
        if use_macx:
            MAC = calculate_macx_matrix(phi_prev, phi_curr)
        else:
            MAC = calculate_mac_matrix(phi_prev, phi_curr)

        # Hungarian assignment (maximize MAC)
        cost = 1 - MAC  # maximize similarity -> minimize cost
        row_ind, col_ind = linear_sum_assignment(cost)

        # Reorder current modes
        phi_curr_sorted = phi_curr[:, col_ind]

        # store
        sorted_modes_list.append(phi_curr_sorted)
        assignment_array[i, :] = col_ind

        # debug plot
        if debug:
            plt.figure(figsize=(6,5))
            sns.heatmap(MAC, annot=True, fmt=".2f", cmap='viridis')
            plt.title(f'MAC Matrix between freq {i-1} and {i}')
            plt.xlabel('Current Modes')
            plt.ylabel('Previous Modes')
            plt.show()

    # Stack all frequencies into a single 3D array: (n_freqs, m, n_modes)
    sorted_modes_array = np.stack(sorted_modes_list, axis=0)

    return sorted_modes_array, assignment_array

def reorder_parameters_by_assignment(parameters_list, assignment_list):
    """
    Reorder mode-dependent parameters using a MAC assignment list.

    Parameters
    ----------
    parameters_list : list of np.ndarray
        Each element is a parameter array at a given frequency, shape (n_modes,) or (m, n_modes)
    assignment_list : list of np.ndarray
        Each element is the assignment array for that frequency, as returned by MAC sorting.

    Returns
    -------
    sorted_parameters : list of np.ndarray
        Parameters reordered according to mode tracking.
    """
    # Convert first element to array if needed
    first_param = np.array(parameters_list[0])
    sorted_parameters = [first_param]  # first frequency stays

    for i in range(1, len(parameters_list)):
        # Convert to array if it's a list
        param = np.array(parameters_list[i])
        assign = np.array(assignment_list[i], dtype=int)  # ensure integer indexing

        if param.ndim == 1:
            sorted_parameters.append(param[assign])
        elif param.ndim == 2:
            sorted_parameters.append(param[:, assign])
        else:
            raise ValueError(f"Unsupported parameter shape: {param.shape}")

    sorted_parameters = np.array(sorted_parameters)
    return sorted_parameters

def calculate_macx_matrix(mode_prev, mode_next):
    """
    Computes the MACX matrix between two mode shape sets.
    MACX penalizes cross-orthogonality, giving a stronger criterion than MAC.

    Parameters
    ----------
    mode_prev : ndarray (ndof × nmodes)
        Modes at previous condition
    mode_next : ndarray (ndof × nmodes)
        Modes at current condition

    Returns
    -------
    MACX : ndarray (nmodes × nmodes)
    """

    n_modes = mode_prev.shape[1]
    MACX = np.zeros((n_modes, n_modes))

    # Precompute all auto/inner products
    A = mode_prev.T @ mode_prev        # previous mode Gram matrix
    B = mode_next.T @ mode_next        # next mode Gram matrix
    C = mode_prev.T @ mode_next        # cross-terms

    # Standard MAC numerator and denominator
    MAC_num = np.abs(C)**2
    MAC_den = np.outer(np.diag(A), np.diag(B))

    MAC = MAC_num / MAC_den            # base MAC

    # MACX adds orthogonality correction
    for i in range(n_modes):
        for j in range(n_modes):
            cross_prev = np.sum(MAC[i, :]) - MAC[i, j]   # coupling of prev-i to remaining
            cross_next = np.sum(MAC[:, j]) - MAC[i, j]   # coupling of next-j to remaining
            MACX[i, j] = MAC[i, j] / (1 + cross_prev + cross_next)

    return MACX