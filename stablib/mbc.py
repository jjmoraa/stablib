import numpy as np
from stablib.state_space import test_periodic
from scipy.linalg import block_diag

def mbc(At, omega, rotating_dofs, debug = False):
    
    # Get matrix size by evaluating once
    n = At(0.0).shape[0]
    n_q = n // 2

    # rotating_dofs are indices in the q block only
    q_rot = list(rotating_dofs)

    # non-rotating coordinates in q block
    q_nr = [i for i in range(n_q) if i not in q_rot]

    # Build MBC ordering:
    # [q_nonrot, q_rot, qdot_nonrot, qdot_rot]
    perm = (
        q_nr +
        q_rot +
        [i + n_q for i in q_nr] +
        [i + n_q for i in q_rot]
    )

    # Number of q coordinates
    n_rot_q = len(q_rot)
    n_nr_q = len(q_nr)

    # For the full state
    n_rot_dofs = 2 * n_rot_q
    n_nr_dofs = 2 * n_nr_q

    # Permuted matrix
    A_perm = lambda t: At(t)[np.ix_(perm, perm)]

    # define the transformation matrices for the degrees of freedom

    def t_curly_1(t):
        psi = omega*t
        psi_b = psi + np.array([0, 2*np.pi/3, 4*np.pi/3])

        return np.column_stack((
                    np.ones(3),
                    np.cos(psi_b),
                    np.sin(psi_b)
                ))

    def t_curly_2(t):
        psi = omega*t
        psi_b = psi + np.array([0, 2*np.pi/3, 4*np.pi/3])

        return np.column_stack((
                    np.zeros(3),
                    -np.sin(psi_b),
                    np.cos(psi_b)
                ))

    def t_curly_3(t):
        psi = omega*t
        psi_b = psi + np.array([0, 2*np.pi/3, 4*np.pi/3])

        return np.column_stack((
                    np.zeros(3),
                    -np.cos(psi_b),
                    -np.sin(psi_b)
                ))
    
    def T1(t):
        T_blocks = [np.eye(n_nr_q,n_nr_q)]

        n_triplets = n_rot_q // 3

        for _ in range(n_triplets):
            T_blocks.append(t_curly_1(t))

        return block_diag(*T_blocks)

    def T2(t):
        T_blocks = [np.zeros((n_nr_q, n_nr_q))]

        n_triplets = n_rot_q // 3

        for _ in range(n_triplets):
            T_blocks.append(t_curly_2(t))

        return block_diag(*T_blocks)
    
    def T3(t):
        T_blocks = [np.zeros((n_nr_q, n_nr_q))]

        n_triplets = n_rot_q // 3

        for _ in range(n_triplets):
            T_blocks.append(t_curly_3(t))

        return block_diag(*T_blocks)
    
    def L(t):

        T1m = T1(t)
        T2m = T2(t)

        return np.block([
            [T1m,               np.zeros_like(T1m)],
            [omega*T2m,         T1m]
        ])


    def R(t):

        T2m = T2(t)
        T3m = T3(t)

        return np.block([
            [omega*T2m,                     np.zeros_like(T2m)],
            [omega**2*T3m,                  2*omega*T2m]
        ])


    A_mbc = lambda t: (
        block_diag(
            np.linalg.inv(T1(t)),
            np.linalg.inv(T1(t))
        )
        @
        (
            A_perm(t) @ L(t)
            - R(t)
        )
    )

    test_periodic(A_mbc,2*np.pi/omega,tol = 0.001)

    if debug:
        return A_mbc, T1, T2, T3

    return A_mbc