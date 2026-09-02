import numpy as np
from scipy.integrate import quad

'''
This model is based on a 3 bladed flapping rotor (no edgewise dofs) for n modes. I added sheared inflow to it so that the system becomes

M q_dot_dot + ( C + Ca(t) ) q_dot + (K + Ka(t)) q = 0
'''

def cantilever_modes(n, L):
    """
    Return the first n Euler-Bernoulli cantilever beam mode shapes.

    Parameters
    ----------
    n : int
        Number of modes.
    L : float
        Beam length.

    Returns
    -------
    modes : list
        List of callable mode-shape functions.
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

def phi(omega,t):
    return np.array([omega*t,omega*t+2*np.pi/3,omega*t+4*np.pi/3])

def U(r, phi_val, U0, shear):
    z = r * np.sin(phi_val)
    return U0 + shear * z

def CL(r, U):
    span_factor = np.minimum(r / 0.25, 1.0)
    speed_factor = U

    return span_factor * speed_factor

def mass(modes, unit_mass, L):
    n = len(modes)

    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            M[i, j] = quad(lambda r: (unit_mass(r) *  modes[i](r) * modes[j](r)),0,L)[0]

    return M


def damping(modes, c, L):
    n = len(modes)

    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            C[i, j] = quad(
                lambda r: (c(r) * modes[i](r) * modes[j](r)), 0, L)[0]

    return C


def stiffness(modes_2, EI, L):
    n = len(modes_2)

    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = quad(lambda r: ( EI(r) * modes_2[i](r) * modes_2[j](r)),0,L)[0]

    return K

def aero_damping(modes, rho, chord, L, omega, t, U0, shear):
    n = len(modes)
    phi_vals = phi(omega, t)
    Ca = []

    for phi_val in phi_vals:
        Ca_blade = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                Ca_blade[i, j] = quad(lambda r: rho * chord(r) * CL(r, U(r, phi_val, U0, shear)) * U(r, phi_val, U0, shear) * modes[i](r) * modes[j](r), 0, L)[0]

        Ca.append(Ca_blade)

    return Ca


def aero_stiffness(modes, rho, chord, L, omega, t, U0, shear):
    n = len(modes)
    phi_vals = phi(omega, t)
    Ka = []

    for phi_val in phi_vals:
        Ka_blade = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                Ka_blade[i, j] = quad(lambda r: -rho * chord(r) * CL(r, U(r, phi_val, U0, shear)) * U(r, phi_val, U0, shear) * shear * modes[i](r) * modes[j](r), 0, L)[0]

        Ka.append(Ka_blade)

    return Ka

from scipy.linalg import block_diag

def stack_matrices(M, C, K, Ca, Ka):
    M_stack = block_diag(M, M, M)
    C_stack = block_diag(C + Ca[0], C + Ca[1], C + Ca[2])
    K_stack = block_diag(K + Ka[0], K + Ka[1], K + Ka[2])

    return M_stack, C_stack, K_stack