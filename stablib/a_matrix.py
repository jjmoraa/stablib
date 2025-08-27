import numpy as np

def a(mass_matrix_3d, stiffness_matrix_3d, damping_matrix_3d):
    """
    Computes the time-dependent A(t) matrix as a 3D array.

    Parameters:
    - mass_matrix_3d (numpy.ndarray): 3D array of mass matrices (N, M, M).
    - stiffness_matrix_3d (numpy.ndarray): 3D array of stiffness matrices (N, M, M).
    - damping_matrix_3d (numpy.ndarray): 3D array of damping matrices (N, M, M).

    Returns:
    - numpy.ndarray: 3D array of A(t) matrices (N, 2M, 2M).
    """
    num_time_steps = mass_matrix_3d.shape[0]
    matrix_size = mass_matrix_3d.shape[1]
    a_matrix_3d = np.zeros((num_time_steps, 2 * matrix_size, 2 * matrix_size))

    for t in range(num_time_steps):
        mass_matrix = mass_matrix_3d[t]
        stiffness_matrix = stiffness_matrix_3d[t]
        damping_matrix = damping_matrix_3d[t]

        # Check if the mass matrix is invertible (optional, can be omitted for performance)
        if np.linalg.det(mass_matrix) == 0:
            raise ValueError(f"Mass matrix at time index {t} is singular and cannot be inverted.")

        # Compute the inverse of the mass matrix (original approach)
        # mass_inv = np.linalg.inv(mass_matrix)
        # a_matrix_3d[t] = np.block([
        #     [np.zeros_like(mass_matrix), np.eye(matrix_size)],
        #     [-mass_inv @ stiffness_matrix, -mass_inv @ damping_matrix]
        # ])

        # Directly solve for mass^{-1} * stiffness and mass^{-1} * damping
        minus_massinv_stiffness = -np.linalg.solve(mass_matrix, stiffness_matrix)
        minus_massinv_damping = -np.linalg.solve(mass_matrix, damping_matrix)

        a_matrix_3d[t] = np.block([
            [np.zeros_like(mass_matrix), np.eye(matrix_size)],
            [minus_massinv_stiffness, minus_massinv_damping]
        ])

    return a_matrix_3d