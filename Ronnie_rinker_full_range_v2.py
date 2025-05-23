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
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import welib.essentials

# Define the mass, damping and stiffness matrices

# Define the mass matrix
def phi(omega,t):
    return np.array([omega*t,omega*t+2*np.pi/3,omega*t+4*np.pi/3])

def mass(m, l, omega, t):
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

# Define the time-dependent matrix A(t)
def a(edgNatFreq_rad, m, l, kx, ky, omega, t):
    mass_matrix = mass(m, l, omega, t)
    stiffness_matrix = stiffness(edgNatFreq_rad, m, l, kx, ky, omega, t)
    damping_matrix = damping(omega, t)
    
    mass_inv = np.linalg.inv(mass_matrix)
    
    return np.block([
        [np.zeros_like(mass_matrix), np.eye(mass_matrix.shape[0])],
        [-mass_inv @ stiffness_matrix, -mass_inv @ damping_matrix]
    ])
# ODE system definition
def ode_system(t, x):
    size = int(np.sqrt(x.size))  # Determine the size of the square matrix
    x_matrix = x.reshape(size, size)  # Reshape the state into a matrix
    a_matrix = a(edgNatFreq_rad,m,l,kx,ky,omega,t)  # Get A(t) matrix at time t
    dxdt_matrix = a_matrix @ x_matrix   # Compute A(t) * x(t), where x is a matrix
    return dxdt_matrix.flatten()  # Flatten to return as a vector

def compute_q(size, sol, B, t_values):
    q_values = []
    for i, t in enumerate(t_values):
        x_t = sol.y[:, i].reshape((size, size))  # Get x(t) at time t
        exp_Bt = expm(B * t)               # Compute e^(Bt)
        #print("this is an inner check for exp_Bt")
        #print("t value",t)
        #print("exp value",exp_Bt)
        q_t = x_t @ np.linalg.inv(exp_Bt)  # Compute q(t) = x(t) * e^(-Bt)
        #print("q value",q_t)
        q_values.append(q_t)
    return q_values

# Compute y(t) for all time steps
def compute_y(t_values, B, y0):
    y_t_values = []
    for t in t_values:
        y_t = expm(B * t) @ y0  # Compute y(t) = e^(Bt) * y0 for each time t
        y_t_values.append(y_t)
    return y_t_values

def compute_x(q_values, y_t_values):
    x_t_values = []
    for i in range(len(q_values)):
        q_t = q_values[i]  # Get Q(t) (Floquet transformation matrix at time t)
        y_t = y_t_values[i]  # Get y(t) (transformed solution at time t)
        x_t = q_t @ y_t  # Compute x(t) = Q(t) * y(t)
        x_t_values.append(x_t)
    return x_t_values


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
    problem_size=a(edgNatFreq_rad,m,l,kx,ky,omega,0).shape[0]
    # Initial condition: identity matrix (flattened)
    x0 = np.eye(problem_size).flatten()
    period=2*np.pi/omega
    time=np.linspace(0,period,1000)
    # Solve the system numerically over one period [0, 2*pi]
    sol = solve_ivp(ode_system, [0, period], x0, t_eval=time, method='RK45')

    #print(sol.y.T)

    #print(f"Time shape: {sol.t.shape}")
    #print(f"Solution shape: {sol.y.shape}")

    data = np.column_stack([sol.t, sol.y.T])
    cols = ['Time'] + [f'vars{i+1}' for i in range(problem_size*problem_size)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv('ronnie_solution_{omega}.csv', index=False)

    # print("monondromy matrix")
    # print(sol.t[-1])
    # Reshape the final solution to get the monodromy matrix M

    # Mon = sol.y.T
    # printMon = np.column_stack((sol.t, M))
    # cols = ['Time'] + [ 'x'+str(i) for i in range(n)]  + [ 'xdot'+str(i) for i in range(n)]
    # df =pd.dataframe(data=M, columns=cols)
    # df.to_csv('filename.csv', index=False)

    Mon = sol.y[:, -1].reshape(problem_size,problem_size)
    df =pd.DataFrame(Mon)
    df.to_csv('monodromy_matrix_{omega}.csv', index=False)


    # Compute the matrix logarithm of M to get Floquet exponent matrix B
    B = logm(Mon) / (2 * np.pi)
    df =pd.DataFrame(B)
    df.to_csv('floquet_exponents_{omega}.csv', index=False)

    # Time values at which q(t) is computed
    t_values = sol.t

    # Compute q(t) for all time steps
    q_values = compute_q(problem_size,sol, B, t_values)

    data = np.column_stack([t_values, [q.flatten() for q in q_values]])
    cols = ['Time'] + [f'q{i+1}' for i in range(data.shape[1] - 1)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv('q_values_ronnie_{omega}.csv', index=False)

    [D,V] = np.linalg.eig(Mon) #The floquet multipliers are the eigenvalues of the monondromy matrix

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
    df.to_csv('monodromy_eigenvalues_eigenvectors_{omega}.csv', index=False)

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
df_multipliers.to_csv('multipliers.csv', index=False)


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