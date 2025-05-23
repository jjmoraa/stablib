
import numpy as np

# Define the mass, damping and stiffness matrices

# Define the mass matrix
def phi(omega,t):
    return np.array([omega*t,omega*t+2*np.pi/3,omega*t+4*np.pi/3])

def mass(M, m, l, omega, t):
    phi_vals = phi(omega, t)
    return np.array([
        [m * l**2, 0, 0, m * l * np.cos(phi_vals[0]), -m * l * np.sin(phi_vals[0])],
        [0, m * l**2, 0, m * l * np.cos(phi_vals[1]), -m * l * np.sin(phi_vals[1])],
        [0, 0, m * l**2, m * l * np.cos(phi_vals[2]), -m * l * np.sin(phi_vals[2])],
        [m * l * np.cos(phi_vals[0]), -m * l * np.cos(phi_vals[1]), -m * l * np.cos(phi_vals[2]), M + 3 * m, 0],
        [-m * l * np.sin(phi_vals[0]), -m * l * np.sin(phi_vals[1]), -m * l * np.sin(phi_vals[2]), 0, M + 3 * m]
    ])

def damping(omega, t):
    phi_vals = phi(omega, t)
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-np.sin(phi_vals[0]), -np.sin(phi_vals[1]), -np.sin(phi_vals[2]), 0, 0],
        [-np.cos(phi_vals[0]), -np.cos(phi_vals[1]), -np.cos(phi_vals[2]), 0, 0]
    ])

def stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t):
    phi_vals = phi(omega, t)
    return np.array([
        [m * (l**2) * (edgNatFreq_rad**2), 0, 0, 0, 0],
        [0, m * (l**2) * (edgNatFreq_rad**2), 0, 0, 0],
        [0, 0, m * (l**2) * (edgNatFreq_rad**2), 0, 0],
        [-m * l * (omega**2) * np.cos(phi_vals[0]), -m * l * (omega**2) * np.cos(phi_vals[1]), -m * l * (omega**2) * np.cos(phi_vals[2]), kx, 0],
        [m * l * (omega**2) * np.sin(phi_vals[0]), m * l * (omega**2) * np.sin(phi_vals[1]), m * l * (omega**2) * np.sin(phi_vals[2]), 0, ky]
    ])
