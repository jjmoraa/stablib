import numpy as np
import matplotlib.pyplot as plt

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
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000

# Range of omegas
omegas = np.linspace(0, 1, 10)

# Store frequencies
all_freqs = []
f_expected=[np.sqrt(kx/(M+3*m)) /(2*np.pi), edgNatFreq_hz]

for omega in omegas:
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

    # Compute natural frequencies (rad/s → Hz)
    nat_freqs = np.sqrt(np.real(eigenvalues))
    all_freqs.append(nat_freqs)

    # print(eigenvalues)
    # print("Natural Frequencies (rad/s):", nat_freqs)

    print("Omega (Hz):", omega)
    print("Natural Frequencies (Hz):", nat_freqs/(2*np.pi))

# Convert to array (shape: len(omegas) x num_modes)
all_freqs = np.array(all_freqs)

# Plot natural frequencies vs omega
fig, ax = plt.subplots(figsize=(8, 5))

# Loop over modes
for mode_idx in range(all_freqs.shape[1]):  
    mode_freqs = all_freqs[:, mode_idx] / (2*np.pi)  # Convert to Hz
    ax.plot(omegas/(2*np.pi), mode_freqs, 'o', label=f'Mode {mode_idx + 1}')

# Labels, grid, legend
ax.set_xlabel('Rotor speed [Hz]')
ax.set_ylabel('Modal frequency [Hz]')
ax.set_title('Campbell Diagram')
ax.grid(True)
ax.legend(loc='best', fontsize='small')

# Offset lines (±f0 around operating line)
for f0 in f_expected:
    ax.plot(omegas/(2*np.pi), f0 + omegas/(2*np.pi), 'k--')
    ax.plot(omegas/(2*np.pi), f0 - omegas/(2*np.pi), 'k--')

plt.tight_layout()
plt.show()

# plt.xlabel("Omega [rad/s]")
# plt.ylabel("Natural Frequency [Hz]")
# plt.title("Campbell Diagram (Natural Frequencies vs Omega)")
# plt.legend()
# plt.grid(True)
# plt.show()
