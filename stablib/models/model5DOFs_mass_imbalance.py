
import numpy as np

# Define the mass, damping and stiffness matrices
# I'm taking m1 to be the imbalanced mass
# Define the mass matrix
def phi(omega,t):
    return np.array([omega*t,omega*t+2*np.pi/3,omega*t+4*np.pi/3])

def mass(M, m, imbalance, l, omega, t):
    m_imb = m * (1 + imbalance)
    phi_vals = phi(omega, t)
    return np.array([
        [m_imb * l**2, 0, 0, m_imb * l * np.cos(phi_vals[0]), -m_imb * l * np.sin(phi_vals[0])],
        [0, m * l**2, 0, m * l * np.cos(phi_vals[1]), -m * l * np.sin(phi_vals[1])],
        [0, 0, m * l**2, m * l * np.cos(phi_vals[2]), -m * l * np.sin(phi_vals[2])],
        [m_imb * l * np.cos(phi_vals[0]), -m * l * np.cos(phi_vals[1]), -m * l * np.cos(phi_vals[2]), M + 2 * m + m_imb, 0],
        [-m_imb * l * np.sin(phi_vals[0]), -m * l * np.sin(phi_vals[1]), -m * l * np.sin(phi_vals[2]), 0, M + 2 * m + m_imb]
    ])

def damping(m, imbalance, l, omega, t):
    m_imb = m * (1 + imbalance)
    phi_vals = phi(omega, t)
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-2 * m_imb * l * omega**2 * np.sin(phi_vals[0]), -2 * m * l * omega**2 * np.sin(phi_vals[1]), -2 * m * l * omega**2 * np.sin(phi_vals[2]), 0, 0],
        [-2 * m_imb * l * omega**2 * np.cos(phi_vals[0]), -2 * m * l * omega**2 * np.cos(phi_vals[1]), -2 * m * l * omega**2 * np.cos(phi_vals[2]), 0, 0]
    ])

def stiffness(edgNatFreq_rad, m, imbalance, l, kx, ky, omega, t):
    m_imb = m * (1 + imbalance)
    phi_vals = phi(omega, t)
    return np.array([
        [m_imb * (l**2) * (edgNatFreq_rad**2), 0, 0, 0, 0],
        [0, m * (l**2) * (edgNatFreq_rad**2), 0, 0, 0],
        [0, 0, m * (l**2) * (edgNatFreq_rad**2), 0, 0],
        [-m_imb * l * (omega**2) * np.cos(phi_vals[0]), -m * l * (omega**2) * np.cos(phi_vals[1]), -m * l * (omega**2) * np.cos(phi_vals[2]), kx, 0],
        [m_imb * l * (omega**2) * np.sin(phi_vals[0]), m * l * (omega**2) * np.sin(phi_vals[1]), m * l * (omega**2) * np.sin(phi_vals[2]), 0, ky]
    ])
