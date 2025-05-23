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

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm

# Define the time-dependent matrix A(t)
def a(t):
    return np.array([[np.cos(t), -np.sin(t)], 
                     [np.sin(t), np.cos(t)]])

# ODE system definition
def ode_system(t, x):
    size = int(np.sqrt(x.size))  # Determine the size of the square matrix
    x_matrix = x.reshape(size, size)  # Reshape the state into a matrix
    a_matrix = a(t)  # Get A(t) matrix at time t
    dxdt_matrix = a_matrix @ x_matrix   # Compute A(t) * x(t), where x is a matrix
    return dxdt_matrix.flatten()  # Flatten to return as a vector

def compute_q(sol, B, t_values):
    q_values = []
    for i, t in enumerate(t_values):
        x_t = sol.y[:, i].reshape((2, 2))  # Get x(t) at time t
        exp_Bt = expm(B * t)               # Compute e^(Bt)
        q_t = x_t @ np.linalg.inv(exp_Bt)  # Compute q(t) = x(t) * e^(-Bt)
        q_values.append(q_t)
    return q_values

# Plotting function for one period
def plot_X_one_period(t_values, sol):
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, sol.y[0, :], label='X1(t)')
    plt.plot(t_values, sol.y[1, :], label='X2(t)')
    plt.title('Solution X(t) over One Period')
    plt.xlabel('Time (t)')
    plt.ylabel('Solution X(t)')
    plt.legend()
    plt.grid()
    plt.show()

# Plotting function for multiple periods
def plot_X_multiple_periods(periods, sol, M):
    # Extend time range over multiple periods
    t_values = np.linspace(0, periods * 2 * np.pi, 1000)
    x_values = np.zeros((2, len(t_values)))  # Prepare array for storing x(t)

    # Loop over the extended time values
    for i, t in enumerate(t_values):
        period_index = int(t // (2 * np.pi))  # Which period
        time_in_period = t % (2 * np.pi)      # Time within the current period
        
        # Compute X(t) based on the current period
        if period_index == 0:
            # For the first period, use the original solution directly from sol.y
            index_in_sol = np.searchsorted(sol.t, time_in_period)
            x_values[:, i] = sol.y[:, index_in_sol][:2]  # Original solution
        else:
            # For subsequent periods, use the monodromy matrix
            previous_x = sol.y[:, index_in_sol][:2]  # Get state at the current time in period
            x_values[:, i] = M @ previous_x  # Apply the monodromy matrix

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, x_values[0, :], label='X1(t)')
    plt.plot(t_values, x_values[1, :], label='X2(t)')
    plt.title('Solution X(t) over Multiple Periods')
    plt.xlabel('Time (t)')
    plt.ylabel('Solution X(t)')
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot q(t) elements
def plot_q(t_values, q_values):
    q11 = [q[0, 0] for q in q_values]
    q12 = [q[0, 1] for q in q_values]
    q21 = [q[1, 0] for q in q_values]
    q22 = [q[1, 1] for q in q_values]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].plot(t_values, q11, label='q11(t)')
    axs[0, 0].set_title('q11(t)')
    
    axs[0, 1].plot(t_values, q12, label='q12(t)')
    axs[0, 1].set_title('q12(t)')
    
    axs[1, 0].plot(t_values, q21, label='q21(t)')
    axs[1, 0].set_title('q21(t)')
    
    axs[1, 1].plot(t_values, q22, label='q22(t)')
    axs[1, 1].set_title('q22(t)')
    
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='q(t)')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

plt.ion()

# Initial condition: identity matrix (flattened)
x0 = np.eye(a(0).shape[0]).flatten()

# Solve the system numerically over one period [0, 2*pi]
sol = solve_ivp(ode_system, [0, 2*np.pi], x0, method='RK45')

# Reshape the final solution to get the monodromy matrix M
M = sol.y[:, -1].reshape((2, 2))

# Compute the matrix logarithm of M to get Floquet exponent matrix B
B = logm(M) / (2 * np.pi)

# Time values at which q(t) is computed
t_values = sol.t

# Compute q(t) for all time steps
q_values = compute_q(sol, B, t_values)

# Output the monodromy matrix
print("Monodromy matrix M:")
print(M)

print("Floquet exponent matrix R:")
print(B)

# Plot q(t)
plot_q(t_values, q_values)

# Plot the solution over one period
plot_X_one_period(t_values, sol)

# Plot the solution over multiple periods
plot_X_multiple_periods(3, sol, M)  # Change the number of periods as needed

plt.show(block=True) 
#plt.ioff()
