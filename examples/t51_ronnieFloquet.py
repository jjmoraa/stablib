'''
x'=q'*y+q*y'
sub in
x'=ax
q'*y+q*y'=ax
also recall x=qy
q'*y+q*y'=a*q*y
q'*y=a*q*y-q*y'
q'*y=q*(a*y-y')
q'=q*(a-b)

Remember to use the monondromy matrix condition
Psi(t,0)'=[A^-1*B]Psi(t,0)

Initial condition must always be I for q
'''
# 
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
# Locals
from ronnie import mass, damping, stiffness
import welib.essentials
from state_space import A_fromMCK, solve_ode_At_flat
from floquet import compute_q, compute_y, compute_x, compute_B
from a_matrix import a
from modeProjection import mode_projection

# Define the mass, damping and stiffness matrices


# Define constants
m = 500
l = 30
M = 50000
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000
omegas = np.linspace(0.1, 1, 100)
eigenvalues_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]
multipliers_for_range=[]

for iom, omega in enumerate(omegas):
    period=2*np.pi/omega
    time=np.linspace(0,period,1000)
    # mass_matrix=mass(m, l, omega, time)
    # damping_matrix=damping(omega, time)
    # stiffness_matrix=stiffness(edgNatFreq_rad, m, l, kx, ky, omega, time)
    # Functions of time
    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )
  
    problem_size=At(0).shape[0]
    # Initial condition: identity matrix (flattened)
    x0 = np.eye(problem_size)
    # Solve the system numerically over one period [0, 2*pi]
    
    # Solve the ODE system
    sol = solve_ode_At_flat(At, x0.flatten(), time)
    print('OK')
    print('proble size A', problem_size)
    # Compute the required transformations
    B = compute_B(sol)
    q_values = compute_q(problem_size, sol, B, time)
    y_t_values = compute_y(time, B, x0)
    x_t_values = compute_x(q_values, y_t_values)
    #sol = solve_ivp(ode_system, [0, period], x0, t_eval=time, method='RK45')

    #print(sol.y.T)

    #print(f"Time shape: {sol.t.shape}")
    #print(f"Solution shape: {sol.y.shape}")

    data = np.column_stack([sol.t, sol.y.T])
    #cols = ['Time'] + [f'vars{i+1}' for i in range(problem_size*problem_size)]
    #df = pd.DataFrame(data, columns=cols)
    #df.to_csv(f'Documents/Independent_Study/data/ronnie_solution_{omega}.csv', index=False)

    # print("monondromy matrix")
    # print(sol.t[-1])
    # Reshape the final solution to get the monodromy matrix M

    # Mon = sol.y.T
    # printMon = np.column_stack((sol.t, M))
    # cols = ['Time'] + [ 'x'+str(i) for i in range(n)]  + [ 'xdot'+str(i) for i in range(n)]
    # df =pd.dataframe(data=M, columns=cols)
    # df.to_csv('filename.csv', index=False)

    Mon = sol.y[:, -1].reshape(problem_size,problem_size)
    #df =pd.DataFrame(Mon)
    #filename= 'monodromy_matrix_{:.4f}'.format(omega)
    #df.to_csv(f'Documents/Independent_Study/data/monodromy_matrix_{omega}.csv', index=False)


    # Compute the matrix logarithm of M to get Floquet exponent matrix B
    B = logm(Mon) / (2 * np.pi)
    #df =pd.DataFrame(B)
    #df.to_csv('Documents/Independent_Study/data/floquet_exponents_{omega}.csv', index=False)

    # Time values at which q(t) is computed
    t_values = sol.t

    # Compute q(t) for all time steps
    q_values = compute_q(problem_size,sol, B, t_values)

    data = np.column_stack([t_values, [q.flatten() for q in q_values]])
    #cols = ['Time'] + [f'q{i+1}' for i in range(data.shape[1] - 1)]
    #df = pd.DataFrame(data, columns=cols)
    #df.to_csv('Documents/Independent_Study/data/q_values_ronnie_{omega}.csv', index=False)

    [D,V] = np.linalg.eig(Mon) #The floquet multipliers are the eigenvalues of the monondromy matrix
    lambda_k=np.real(D)
    lambda_mat=np.diag(lambda_k)
    #Lets try to get mode projejction info
    fourier_coeffs = mode_projection(q_values, V, lambda_mat, t_values)

    # # Sort indices based on eigenvalues (ascending order)
    # sorted_indices = np.argsort(D)

    # # Reorder eigenvalues and eigenvectors
    # sorted_eigenvalues = D[sorted_indices]
    # sorted_eigenvectors = V[:, sorted_indices]

    # D=sorted_eigenvalues
    # V=sorted_eigenvectors

    data = {
        'Eigenvalue': D,}

    for i in range(problem_size):
        data[f'Eigenvector_{i+1}'] = V[:, i]

    df = pd.DataFrame(data)
    df.to_csv('Documents/Independent_Study/data/monodromy_eigenvalues_eigenvectors_{omega}.csv', index=False)

    print("Eigenvalues")
    print(D)
    print(np.abs(D)/(2*np.pi))
    lk = D
    pk=(1/period)*np.log(D)
    re_lk = np.real(lk)
    im_lk = np.imag(lk)
    omk = 1/period * np.arctan(im_lk/re_lk)
    Multipliers = np.abs(lk) #/(2*np.pi)
    multipliers_for_range.append(Multipliers)

# Find the number of multipliers
num_multipliers = len(multipliers_for_range[0])  # Assumes non-empty list and equal-length arrays

# Create the DataFrame for the Campbell diagram (mode frequencies)
data_multipliers = np.column_stack(
    [omegas,  # Omega values (assuming same scale as the loop)
     np.array(multipliers_for_range)]  # Convert list of arrays to a 2D array
)
cols_multipliers = ['Omega'] + [f'multipliers({i+1})' for i in range(num_multipliers)]
df_multipliers = pd.DataFrame(data_multipliers, columns=cols_multipliers)
df_multipliers.to_csv('Documents/Independent_Study/data/multipliers.csv', index=False)


# # Create the DataFrame for the Campbell diagram (mode frequencies)
# data_frequencies = np.column_stack(
#     [omegas,  # Omega values (assuming same scale as the loop)
#      np.array(mode_frequencies)]  # Convert list of arrays to a 2D array
# )
# cols_frequencies = ['Omega'] + [f'mode_frequencies({i+1})' for i in range(data_frequencies.shape[1] - 1)]
# df_frequencies = pd.DataFrame(data_frequencies, columns=cols_frequencies)
# df_frequencies.to_csv('Campbell_diagram.csv', index=False)

# # Create the DataFrame for the stability diagram (mode decay)
# data_decay = np.column_stack(
#     [omegas,  # Omega values
#      np.array(mode_decay)]  # Convert list of arrays to a 2D array
# )
# cols_decay = ['Omega'] + [f'mode_decay({i+1})' for i in range(data_decay.shape[1] - 1)]
# df_decay = pd.DataFrame(data_decay, columns=cols_decay)
# df_decay.to_csv('stability_diagram.csv', index=False)

# data_damping_ratio = np.column_stack(
#     [omegas,  # Omega values
#      np.array(damping_ratio)]  # Convert list of arrays to a 2D array
# )
# cols_damping_ratio = ['Omega'] + [f'damping_ratio({i+1})' for i in range(data_damping_ratio.shape[1] - 1)]
# df_damping_ratio = pd.DataFrame(data_damping_ratio, columns=cols_damping_ratio)
# df_damping_ratio.to_csv('damping_ratio.csv', index=False)

'''
# Plot q(t)
plot_q(t_values, q_values)

# Plot the solution over one period
plot_X_one_period(t_values, sol)

plt.show(block=True) 
#plt.ioff()
'''