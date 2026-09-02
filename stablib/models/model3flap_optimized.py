import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag


'''
This model is based on a 3 bladed flapping rotor (no edgewise dofs)
for n modes. Sheared inflow is included so that the system becomes

M q_ddot + (C + Ca(t)) q_dot + (K + Ka(t)) q = 0

The aerodynamic matrices are precomputed over an azimuth grid and
interpolated during the time integration.
'''


def cantilever_modes(n, L):
    """
    Return the first n Euler-Bernoulli cantilever beam mode shapes.
    """

    betaL = np.array([
        1.87510407,
        4.69409113,
        7.85475744,
        10.99554073,
        14.13716839,
        17.27875953,
        20.42035225,
        23.56194490
    ])

    if n > len(betaL):
        raise ValueError("Only the first 8 modes are defined.")

    modes = []

    for b in betaL[:n]:

        sigma = (
            np.cosh(b) + np.cos(b)
        ) / (
            np.sinh(b) + np.sin(b)
        )

        def mode(r, b=b, sigma=sigma):

            xi = r / L

            return (
                np.cosh(b * xi)
                - np.cos(b * xi)
                - sigma * (
                    np.sinh(b * xi)
                    - np.sin(b * xi)
                )
            )

        modes.append(mode)

    return modes


def phi(omega, t):
    """
    Azimuth of each blade at time t.
    """

    return np.array([
        omega * t,
        omega * t + 2 * np.pi / 3,
        omega * t + 4 * np.pi / 3
    ])


def U(r, phi_val, U0, shear):
    """
    Sheared inflow velocity at radial position r and azimuth phi.
    """

    z = r * np.sin(phi_val)

    return U0 + shear * z


def CL(r, U):
    """
    Simplified lift coefficient model.
    """

    span_factor = np.minimum(r / 0.25, 1.0)
    speed_factor = U

    return span_factor * speed_factor


# ============================================================
# STRUCTURAL MATRICES
# ============================================================

def mass(modes, unit_mass, L):

    n = len(modes)

    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            M[i, j] = quad(
                lambda r:
                    unit_mass(r)
                    * modes[i](r)
                    * modes[j](r),
                0,
                L
            )[0]

    return M


def damping(modes, c, L):

    n = len(modes)

    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            C[i, j] = quad(
                lambda r:
                    c(r)
                    * modes[i](r)
                    * modes[j](r),
                0,
                L
            )[0]

    return C


def stiffness(modes_2, EI, L):

    n = len(modes_2)

    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            K[i, j] = quad(
                lambda r:
                    EI(r)
                    * modes_2[i](r)
                    * modes_2[j](r),
                0,
                L
            )[0]

    return K


# ============================================================
# AERODYNAMIC MATRICES AT A SINGLE AZIMUTH
# ============================================================

def aero_damping_at_phi(
    modes,
    rho,
    chord,
    L,
    phi_val,
    U0,
    shear
):

    n = len(modes)

    Ca = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            Ca[i, j] = quad(
                lambda r:
                    rho
                    * chord(r)
                    * CL(
                        r,
                        U(r, phi_val, U0, shear)
                    )
                    * U(r, phi_val, U0, shear)
                    * modes[i](r)
                    * modes[j](r),
                0,
                L
            )[0]

    return Ca


def aero_stiffness_at_phi(
    modes,
    rho,
    chord,
    L,
    phi_val,
    U0,
    shear
):

    n = len(modes)

    Ka = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            Ka[i, j] = quad(
                lambda r:
                    -rho
                    * chord(r)
                    * CL(
                        r,
                        U(r, phi_val, U0, shear)
                    )
                    * U(r, phi_val, U0, shear)
                    * shear
                    * modes[i](r)
                    * modes[j](r),
                0,
                L
            )[0]

    return Ka


# ============================================================
# PRECOMPUTE AERODYNAMIC MATRICES
# ============================================================

def precompute_aero(
    modes,
    rho,
    chord,
    L,
    U0,
    shear,
    n_phi=361
):
    """
    Precompute Ca and Ka over one complete rotor revolution.

    Returns
    -------
    phi_grid : ndarray
        Azimuth grid [rad].

    Ca_grid : ndarray
        Aerodynamic damping matrices.
        Shape = (n_phi, n_modes, n_modes)

    Ka_grid : ndarray
        Aerodynamic stiffness matrices.
        Shape = (n_phi, n_modes, n_modes)
    """

    phi_grid = np.linspace(
        0,
        2 * np.pi,
        n_phi
    )

    Ca_grid = np.zeros(
        (n_phi, len(modes), len(modes))
    )

    Ka_grid = np.zeros(
        (n_phi, len(modes), len(modes))
    )

    for k, phi_val in enumerate(phi_grid):

        Ca_grid[k] = aero_damping_at_phi(
            modes,
            rho,
            chord,
            L,
            phi_val,
            U0,
            shear
        )

        Ka_grid[k] = aero_stiffness_at_phi(
            modes,
            rho,
            chord,
            L,
            phi_val,
            U0,
            shear
        )

    return phi_grid, Ca_grid, Ka_grid


# ============================================================
# INTERPOLATE AERODYNAMIC MATRICES
# ============================================================

def aero_matrices_at_phi(
    phi_val,
    phi_grid,
    Ca_grid,
    Ka_grid
):
    """
    Return Ca and Ka at a given azimuth using interpolation.
    """

    # Wrap azimuth into [0, 2*pi)
    phi_val = phi_val % (2 * np.pi)

    n = Ca_grid.shape[1]

    Ca = np.zeros((n, n))
    Ka = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            Ca[i, j] = np.interp(
                phi_val,
                phi_grid,
                Ca_grid[:, i, j]
            )

            Ka[i, j] = np.interp(
                phi_val,
                phi_grid,
                Ka_grid[:, i, j]
            )

    return Ca, Ka


# ============================================================
# THREE-BLADE MATRIX ASSEMBLY
# ============================================================

def stack_matrices(
    M,
    C,
    K,
    phi_vals,
    phi_grid,
    Ca_grid,
    Ka_grid
):
    """
    Stack the three blade systems into global matrices.
    """

    Ca = []
    Ka = []

    for phi_val in phi_vals:

        Ca_blade, Ka_blade = aero_matrices_at_phi(
            phi_val,
            phi_grid,
            Ca_grid,
            Ka_grid
        )

        Ca.append(Ca_blade)
        Ka.append(Ka_blade)

    M_stack = block_diag(
        M,
        M,
        M
    )

    C_stack = block_diag(
        C + Ca[0],
        C + Ca[1],
        C + Ca[2]
    )

    K_stack = block_diag(
        K + Ka[0],
        K + Ka[1],
        K + Ka[2]
    )

    return M_stack, C_stack, K_stack