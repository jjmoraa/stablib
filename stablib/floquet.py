import numpy as np
from scipy.linalg import logm, expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Local
from stablib.state_space import solve_ode_At_flat
from stablib.modeProjection import mode_projection
from stablib.PostProcessing import plot_matrix

def compute_B(C,period):

    problem_size = int(np.sqrt(C.shape[0]))
    print('proble size B', problem_size)
    # C = sol.y[:, -1].reshape(problem_size,problem_size)
    # df =pd.DataFrame(Mon)
    # df.to_csv('monodromy_matrix_{omega}.csv', index=False)
    # Compute the matrix logarithm of M to get Floquet exponent matrix B
    # C is not the monondromy matrix
    # B = logm(C) / (2 * np.pi)
    B = logm(C) / period
    return B

def compute_q(size, sol, B, t_values, method='manual'):
    """
    Computes the Floquet transformation matrix Q(t).

    Parameters:
    - size (int): Dimension of the square matrix.
    - sol (object): Solution object from an ODE solver.
    - B (numpy.ndarray): Time-invariant matrix from the Floquet transform.
    - t_values (numpy.ndarray): Time values for the solution.

    Returns:
    - list of numpy.ndarray: List of Q(t) matrices for each time step.
    """
    if method=='manual':
        q_values = []
        for i, t in enumerate(t_values):
            x_t = sol.y[:, i].reshape((size, size))  # Get x(t) at time t
            exp_Bt = expm(B * t)  # Compute e^(Bt)
            q_t = x_t @ np.linalg.inv(exp_Bt)  # Compute q(t) = x(t) * e^(-Bt)
            q_values.append(q_t)
    else:
        raise NotImplementedError(method)
    return q_values

def compute_y(t_values, B, y0):
    """
    Computes y(t) = e^(Bt) * y0 for all time steps.

    Parameters:
    - t_values (numpy.ndarray): Time values for the solution.
    - B (numpy.ndarray): Time-invariant matrix from the Floquet transform.
    - y0 (numpy.ndarray): Initial condition for y(t).

    Returns:
    - list of numpy.ndarray: List of y(t) for each time step.
    """
    y_t_values = []
    for t in t_values:
        #import pdb ; pdb.set_trace()
        y_t = expm(B * t) @ y0
        y_t_values.append(y_t)
    return y_t_values

def compute_x(q_values, y_t_values):
    """
    Computes x(t) = Q(t) * y(t) for all time steps.

    Parameters:
    - q_values (list of numpy.ndarray): List of Q(t) matrices for each time step.
    - y_t_values (list of numpy.ndarray): List of y(t) vectors for each time step.

    Returns:
    - list of numpy.ndarray: List of x(t) matrices for each time step.
    """
    x_t_values = []
    for i in range(len(q_values)):
        q_t = q_values[i]  # Get Q(t) (Floquet transformation matrix at time t)
        y_t = y_t_values[i]  # Get y(t) (transformed solution at time t)
        x_t = q_t @ y_t  # Compute x(t) = Q(t) * y(t)
        x_t_values.append(x_t)
    return x_t_values

def solve(At,time,plot=True,folder_name=None):
    print('Solve IVP for At')
    problem_size=At(0).shape[0]
    # Initial condition: identity matrix (flattened)
    
    x0 = np.eye(problem_size)
    # Solve the system numerically over one period [0, 2*pi]
    
    # Solve the ODE system
    sol = solve_ode_At_flat(At, x0.flatten(), time)
    
    # Reshape solution to (len(time), n, n) so each time step is a matrix
    y_vals = sol.y.reshape(problem_size, problem_size, len(time)).transpose(2, 0, 1)
    # y_vals = sol.y.reshape(len(time), problem_size, problem_size)

    if plot:
        plot_matrix(y_vals, time, folder_name=None)
    return sol

def floquet_eigenanalysis(sol,time,omega,plot=False):
    print('computing eigenvalues for solution usingn floquet analysis')
    # Compute the required transformations

    period=2*np.pi/omega
    problem_size=int(np.sqrt(sol.y.shape[0]))
    x0 = np.eye(problem_size)
    # Solve the system numerically over one period [0, 2*pi]

    # Find monodromy matrix (Last instance of fundamental matrix)
    monodromy = sol.y[:, -1].reshape(problem_size,problem_size)

    # Find floquet exponent matrix
    # B = logm(C) / (2 * np.pi) where C is monodromy
    exponent_matrix = compute_B(monodromy, period)
    
    # Get state transformation matrix
    q_values = compute_q(problem_size, sol, exponent_matrix, time) #plot the time series and fft from mode_projection add debug flags
    q_values = np.array(q_values)

    # Perform periodicity test on Q
    test_periodic_matrix(q_values, period, tol=1e-3)
    #Hey justin, what do you mean you can guess(?) the modes from just looking at this

    # Get solution in floquet space (y) by employing exponent matrix, initial conditions
    y_t_values = compute_y(time, exponent_matrix, x0)
    y_t_values = np.array(y_t_values)

    # get solution x from floquet solution and transformation matrix
    x_t_values = compute_x(q_values, y_t_values)
    x_t_values = np.array(x_t_values)

    #if np.allclose(x_t_values.reshape(x_t_values.shape[0], 4), sol.y.T, atol=1e-3):
    #    print('calculated solution from floquet theory is close to integrated solution')

    
    if plot:
        #plot_X_one_period(time, sol)
        plot_matrix(q_values, time, folder_name='q_values')
    
    #The floquet multipliers are the eigenvalues of the monondromy matrix
    [eigenvalues_mon,eigenvectors_mon] = np.linalg.eig(monodromy)

    #The floquet multipliers are the eigenvalues of the monondromy matrix
    [eigenvalues_exp,eigenvectors_exp] = np.linalg.eig(exponent_matrix)

    # lambda_real=np.real(eigenvalues)
    # lambda_imag=np.imag(eigenvalues)
    # lambda_matrix=np.diag(lambda_real)
    # om_p =[]
    # for i, lam in enumerate(lambda_imag):
    #     if lam>omega/2 or lam <-omega/2:
    #         print('[WARN] omega is not between -om/2 and om/2')
    #     om = np.mod(lam , omega  ) # principal value between 0 and omega
    #     if om>omega/2:
    #         om -=omega # now om is betweeen -omega/2 and omega/2
    #     om_p.append(om)
    # print('Omega _p', om_p)
    # om_p = lambda_imag # principal value

    # [max_vals,max_index,participation_factor] = mode_projection(C, q_values, V, time, plot=True) #####REALLY CHECK

    # eigenvalues=lambda_real+1j*(om_p + max_index*omega)
 
    # omega_d = np.imag(eigenvalues)
    # f_d = omega_d / (2*np.pi)
    # omega_0 = np.abs(eigenvalues)
    # f_0 = omega_0 / (2*np.pi)
    # zeta= -np.real(eigenvalues)/omega_0
    # print('f0',f_0)

    return monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values

'''
def floquet_eigenanalysis_old(sol,time,omega,C,plot=False):
    print('computing eigenvalues for solution usingn floquet analysis')
    # Compute the required transformations

    period=2*np.pi/omega
    problem_size=int(np.sqrt(sol.y.shape[0]))
    x0 = np.eye(problem_size)
    # Solve the system numerically over one period [0, 2*pi]
    B = compute_B(sol,period)
    q_values = compute_q(problem_size, sol, B, time) #plot the time series and fft from mode_projection add debug flags
    q_values = np.array(q_values)
    test_periodic_matrix(q_values, period, tol=1e-3)
    #Hey justin, what do you mean you can guess(?) the modes from just looking at this
    y_t_values = compute_y(time, B, x0)
    y_t_values = np.array(y_t_values)
    x_t_values = compute_x(q_values, y_t_values)
    
    if plot:
        #plot_X_one_period(time, sol)
        plot_matrix(q_values, time, folder_name='q_values')

    #Mon = sol.y[:, -1].reshape(problem_size,problem_size)
    
    [D,V] = np.linalg.eig(B) #The floquet multipliers are the eigenvalues of the monondromy matrix
    
    lambda_real=np.real(D)
    lambda_imag=np.imag(D)
    lambda_matrix=np.diag(lambda_real)
    om_p =[]
    for i, lam in enumerate(lambda_imag):
        if lam>omega/2 or lam <-omega/2:
            print('[WARN] omega is not between -om/2 and om/2')
        om = np.mod(lam , omega  ) # principal value between 0 and omega
        if om>omega/2:
            om -=omega # now om is betweeen -omega/2 and omega/2
        om_p.append(om)
    print('Omega _p', om_p - lambda_imag)
    om_p = lambda_imag # principal value

    [max_vals,max_index] = mode_projection(C, q_values, V, time, plot=False) #####REALLY CHECK

    eigenvalues=lambda_real+1j*(om_p + max_index*omega)
 
    omega_d = np.imag(eigenvalues)
    f_d = omega_d / (2*np.pi)
    omega_0 = np.abs(eigenvalues)
    f_0 = omega_0 / (2*np.pi)
    zeta= -np.real(eigenvalues)/omega_0
    print('f0',f_0)

    return eigenvalues
'''
def test_periodic(At, period, tol=1e-3):
    """Tests if At is periodic with the given period."""
    t0 = 0  # Initial time
    t1 = period  # One full period later
    
    # Compute matrices at t0 and t1
    A0 = At(t0)
    A1 = At(t1)
    
    # Check if they are approximately equal
    if np.allclose(A0, A1, atol=tol):
        print("Matrix is periodic")
    else:
        print("Matrix is NOT periodic")
        raise ValueError("Matrix is not periodic")

def test_periodic_matrix(matrix, period, tol=1e-3):
    """Tests if At is periodic with the given period."""
    # t0 = 0  # Initial time
    # t1 = period  # One full period later
    
    # Compute matrices at t0 and t1
    matrix0 = matrix[0,:,:]
    matrix1 = matrix[-1,:,:]
    
    # Check if they are approximately equal
    if np.allclose(matrix0, matrix1, atol=tol):
        print("Matrix is periodic")
    else:
        print("Matrix is NOT periodic")
        raise ValueError("Matrix is not periodic")
#make for other type of variables too


def floquet_from_monodromy(monodromy, time_stm, period):
    """ 



    """
    from scipy.linalg import eig
    # Compute characteristic multipliers.
    theta, S = eig(monodromy)

    # Define shift for characteristic exponents.
    if time_stm.size % 2 == 0:
        max_shift = int(time_stm.size / 2)
    else:
        max_shift = int((time_stm.size - 1) / 2)
    shift = np.arange(-max_shift, +max_shift)
    shift0 = max_shift  # n = 0.

    # Compute characteristic exponents.
    # eta is ordered as:
    #  - axis 0: harmonics.
    #  - axis 1: modes.
    abs_theta = np.abs(theta)
    ang_theta = np.angle(theta)
    eta = (np.log(abs_theta) + 1j * ang_theta)[np.newaxis, :] / period   + 2j * np.pi / period * shift[:, np.newaxis]


    # Compute frequency and damping.
    natural_frequency = np.abs(eta)  # [rad/s]
    damping_ratio = -eta.real / natural_frequency  # [-]
    natural_frequency /= 2 * np.pi  # [Hz]
    damped_frequency = np.abs(eta.imag) / (2 * np.pi)  # [Hz]


    # # Plot state at every period.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Displacement [m]")
    # ax.set_xlim(0.0, 100.0)
    # ax.plot(sol_free.t, sol_free.y[0, :])
    # ax.scatter(time_sampled, x_sampled[0, :], color="C1")
    # fig.savefig(
    #     "./mathieu_oscillator/free_response_time_with_monodromy.svg",
    #     bbox_inches="tight",
    # )


    # # Plot characteristic multipliers.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Real")
    # ax.set_ylabel("Imag")
    # ax.set_xlim(-1.1, +1.1)
    # ax.set_ylim(-1.1, +1.1)
    # ax.set_aspect("equal")
    # ax.add_artist(plt.Circle(xy=(0.0, 0.0), radius=1.0, edgecolor="k", fill=False))
    # for i in range(theta.size):
    #     ax.scatter(theta[i].real, theta[i].imag, label=f"Mode {i+1}")
    # ax.legend(loc="center")
    # fig.savefig(
    #     "./mathieu_oscillator/characteristic_multiplier.svg", bbox_inches="tight"
    # )
       # Plot characteristic exponents.
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Real [rad/s]")
    # ax.set_ylabel("Imag [rad/s]")
    # ax.set_xlim(-0.04, +0.02)
    # ax.set_ylim(-3.2, +3.2)
    # for i in range(eta.shape[1]):
    #     ax.scatter(eta[:, i].real, eta[:, i].imag, label=f"Mode {i+1}")
    # ax.legend()

    # fig.savefig(
    #     "./mathieu_oscillator/characteristic_exponent.svg", bbox_inches="tight"
    # )
