import numpy as np

# Function declaration must come before any calls to it
def spaceStateTransformation(mass, damping, stiffness):
    A = np.concatenate(
        (np.concatenate((np.zeros(np.shape(mass)), np.eye(5)), axis=1),
        np.concatenate((-np.linalg.inv(mass) @ damping, np.linalg.inv(mass) @ stiffness), axis=1)),
        axis=0
    )
    return A

# Define constants
m = 500
l = 30
M = 50000
omega = 0
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000

# Define the mass matrix
mass = [[m * l**2, 0, 0, 0, 0],
        [0, m * l**2, 0, m * l, 0],
        [0, 0, m * l**2, 0, -m * l],
        [0, 3 * m * l / 2, 0, M + 3 * m, 0],
        [0, 0, -3 * m * l / 2, 0, M + 3 * m]]

# Define the damping matrix
damping = [[0, 0, 0, 0, 0],
           [0, 0, 0, 2 * m * (l**2) * omega, 0],
           [0, 0, -2 * m * (l**2) * omega, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]

# Define the stiffness matrix
stiffness = [[m * (l**2) * (edgNatFreq_rad**2), 0, 0, 0, 0],
             [0, m * (l**2) * (edgNatFreq_rad**2) - m * (l**2) * (omega**2), 0, 0, 0],
             [0, 0, m * (l**2) * (edgNatFreq_rad**2) - m * (l**2) * (omega**2), 0, 0],
             [0, 0, 0, kx, 0],
             [0, 0, 0, 0, ky]]

# Call the function after it has been defined
A = spaceStateTransformation(mass, damping, stiffness)
eigenvalues, eigenvectors = np.linalg.eig(A)
# Uncomment to print matrices
# print(mass)
# print(damping)
# print(stiffness)
# print(A)
# naturalFrequencies = np.sqrt(np.real(eigenvaluesA))

natural_frequencies = np.sqrt(np.real(eigenvalues))

print(eigenvalues)
print("Natural Frequencies (rad/s):", natural_frequencies)
print("Natural Frequencies (Hz):", natural_frequencies/(2*np.pi))
print("Mode Shapes:\n", eigenvectors)
# print(naturalFrequencies)


