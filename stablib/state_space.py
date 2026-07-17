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

def mac_sort_modes_pf(mode_shapes, participation_factor, use_macx=False, debug=False):
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

def calculate_macx_matrix_pf(mode_prev, mode_next):
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

def make_matrix_interpolator(matrices, period=None, positions=None, kind='cubic'):
    """
    Create a lambda function to interpolate between a series of matrices.
    
    If only one matrix is given, returns a constant function.

    Parameters
    ----------
    matrices : np.ndarray
        Array of shape (nMatrices, nStates, nStates) containing the matrices to interpolate.
    positions : np.ndarray, optional
        Positions corresponding to each matrix. If None, uses np.linspace(0, 1, nMatrices).
    kind : str
        Type of interpolation ('linear', 'cubic', etc.)
    period : float, optional
        Period in seconds. If given, the returned lambda accepts time in seconds.

    Returns
    -------
    interp_lambda : function
        Lambda function interp_lambda(pos_or_time) that returns an interpolated matrix.
        If period is given, input is in seconds; otherwise input is normalized [0,1].
    """
    nMatrices, nStates, _ = matrices.shape

    if nMatrices == 1:
        return lambda pos: matrices[0]

    if positions is None:
        positions = np.linspace(0, 1, nMatrices)
    
    matrices_flat = matrices.reshape(nMatrices, -1)
    interp_func = interp1d(positions, matrices_flat, axis=0, kind=kind, fill_value="extrapolate")
    
    if period is None:
        # Input is normalized position
        return lambda pos: interp_func(pos).reshape(nStates, nStates)
    else:
        # Input is physical time; scale and wrap
        return lambda t: interp_func((t / period) % 1.0).reshape(nStates, nStates)

def mbc(
    A,
    psi,
    Omega,
    rotating_groups,
):
    """
    Generalized Multi-Blade Coordinate (MBC) operator.

    Parameters
    ----------
    A : (n,n) ndarray
        Periodic system matrix.

    psi : float
        Rotor azimuth [rad].

    Omega : float
        Rotor speed [rad/s].

    rotating_groups : list[list[int]]
        Lists of equivalent cyclic DOFs.

        Example:
        rotating_groups = [
            [4,17,33],   # blade copies of same coordinate
            [5,18,34],
            [6,19,35],
        ]

    Returns
    -------
    A_mbc : (n,n) ndarray
        MBC transformed system matrix.

    T : (n,n) ndarray
        Coordinate transformation matrix.

    Tdot : (n,n) ndarray
        Time derivative of T.
    """

    n = A.shape[0]

    T = np.eye(n, dtype=complex)
    Tdot = np.zeros((n, n), dtype=complex)

    for group in rotating_groups:

        nb = len(group)

        blade_angles = (
            psi
            + 2*np.pi*np.arange(nb)/nb
        )

        # ---------------------------------
        # Fourier / cyclic symmetry block
        # ---------------------------------

        F = np.zeros((nb, nb), dtype=complex)
        dF = np.zeros((nb, nb), dtype=complex)

        for k in range(nb):

            phase = np.exp(
                -1j * k * blade_angles
            )

            F[k, :] = phase / np.sqrt(nb)

            dF[k, :] = (
                -1j * k * Omega * phase
                / np.sqrt(nb)
            )

        idx = np.ix_(group, group)

        T[idx] = F
        Tdot[idx] = dF

    # ---------------------------------
    # MBC transformed system
    # ---------------------------------

    A_mbc = (
        (T @ A + Tdot)
        @ np.linalg.solve(T, np.eye(n))
    )

    return A_mbc, T, Tdot

def mbc_state_space(A, psi, Omega, num_structural_dofs, blade_dofs):
    """
    Applies real-valued MBC transformation to a 3-blade state-space matrix.
    Assumes state vector layout is x = [q, q_dot]
    
    Parameters:
    -----------
    A : (n, n) ndarray - State space matrix at current time slice
    psi : float - Rotor azimuth angle (rad)
    Omega : float - Rotor speed (rad/s)
    num_structural_dofs : int - Number of DOFs in original M, C, K matrices (e.g., 5)
    blade_dofs : list of int - 0-indexed positions of the 3 blades in M, C, K (e.g., [1, 2, 3])
    """
    n = A.shape[0]
    N = num_structural_dofs
    
    # 1. Build the building block 3x3 Coleman Transformation matrix (MBC to Physical)
    # psi_b = psi + 2*pi*(b-1)/3
    psi_b = psi + 2 * np.pi * np.array([0, 1, 2]) / 3
    
    T3 = np.zeros((3, 3))
    T3[:, 0] = 1.0                          # Collective/Multi-rotational
    T3[:, 1] = np.cos(psi_b)                # Cosine component
    T3[:, 2] = np.sin(psi_b)                # Sine component
    
    # Derivative of T3 with respect to time (dpsi/dt = Omega)
    dT3 = np.zeros((3, 3))
    dT3[:, 0] = 0.0
    dT3[:, 1] = -Omega * np.sin(psi_b)
    dT3[:, 2] =  Omega * np.cos(psi_b)
    
    # 2. Embed into structural size identity matrices (System size N)
    T_q = np.eye(N)
    dT_q = np.zeros((N, N))
    
    # Map the blade DOFs explicitly
    for i, src_idx in enumerate(blade_dofs):
        for j, dest_idx in enumerate(blade_dofs):
            T_q[src_idx, dest_idx] = T3[i, j]
            dT_q[src_idx, dest_idx] = dT3[i, j]
            
    # 3. Construct full state-space transformation matrix T_ss (Size 2N x 2N)
    # For x = [q; q_dot] and y = [q_mbc; q_mbc_dot]:
    # x_dot = T_q * y_dot + dT_q * y
    T_ss = np.zeros((n, n))
    T_ss[0:N, 0:N] = T_q
    T_ss[N:n, N:n] = T_q       # Velocity block uses T_q
    T_ss[N:n, 0:N] = dT_q      # Cross coupling due to derivative of displacement mapping
    
    # Time derivative of state-space transformation matrix
    dT_ss = np.zeros((n, n))
    dT_ss[0:N, 0:N] = dT_q
    dT_ss[N:n, N:n] = dT_q
    # (Assuming Omega is constant, d^2(T_q)/dt^2 would go in the bottom left block if needed,
    # but the explicit state equation handles it via the identity relation substitution)
    
    # 4. Apply explicit state-space coordinate change: A_mbc = T^-1 * A * T - T^-1 * T_dot
    T_inv = np.linalg.inv(T_ss)
    A_mbc = T_inv @ A @ T_ss - T_inv @ dT_ss
    
    return A_mbc