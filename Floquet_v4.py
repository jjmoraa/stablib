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
        #print("this is an inner check for exp_Bt")
        #print("t value",t)
        #print("exp value",exp_Bt)
        q_t = x_t @ np.linalg.inv(exp_Bt)  # Compute q(t) = x(t) * e^(-Bt)
        #print("q value",q_t)
        q_values.append(q_t)
    return q_values

# Plotting function for one period
def plot_X_one_period(t_values, sol):
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, sol.y[0, :], label='X1(t)')
    plt.plot(t_values, sol.y[1, :], label='X2(t)')
    plt.plot(t_values, sol.y[2, :], label='X3(t)')
    plt.plot(t_values, sol.y[3, :], label='X4(t)')
    plt.title('Solution X(t) over One Period')
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

# Optionally, you can plot x(t) similarly to how you plotted q(t):
def plot_x(t_values, x_t_values):
    x11 = [x[0, 0] for x in x_t_values]
    x12 = [x[0, 1] for x in x_t_values]
    x21 = [x[1, 0] for x in x_t_values]
    x22 = [x[1, 1] for x in x_t_values]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].plot(t_values, x11, label='x11(t)')
    axs[0, 0].set_title('x11(t)')
    
    axs[0, 1].plot(t_values, x12, label='x12(t)')
    axs[0, 1].set_title('x12(t)')
    
    axs[1, 0].plot(t_values, x21, label='x21(t)')
    axs[1, 0].set_title('x21(t)')
    
    axs[1, 1].plot(t_values, x22, label='x22(t)')
    axs[1, 1].set_title('x22(t)')
    
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='x(t)')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

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

plt.ion()

# Initial condition: identity matrix (flattened)
x0 = np.eye(a(0).shape[0]).flatten()

t=np.linspace(0,2*np.pi,5000)

# Solve the system numerically over one period [0, 2*pi]
sol = solve_ivp(ode_system, [0, 2*np.pi], x0, t_eval=t, method='RK45')

print("monondromy matrix calculated at")
print(sol.t[-1])
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

[D,V] = np.linalg.eig(B)

print("Eigenvalues")
print(D)

print("Eigenvectors")
print(V)

invV = np.eye(a(0).shape[0])/V
Exponents = np.diag(D)

print("Floquet exponents")
print(Exponents)

Multipliers = np.exp(np.diag(D)*2*np.pi)

print("Multipliers")
print(Multipliers)

# Adjust q(t) values for plotting and extend to multiple time steps
q_adjusted = []
for i in range(10):
    q_adjusted.extend(q_values[1:len(q_values)])  # Extend the array with q_values without first element

# Generate an array of time values for plotting (0 to 10*pi)
time_range = np.linspace(0, 20*np.pi, len(q_adjusted))

# Compute y_t for all time steps in time_range
y0 = q_values[0] * np.array([[1, 0], [0, 1]])  # Initial condition
y_t_values = compute_y(time_range, B, y0)  # Compute y(t) for all time steps

print("y values")
print(y_t_values[0:len(q_values)])

x_t_values = compute_x(q_adjusted, y_t_values)

# Solution given by folkers

# Initialize the folk_sol array
folk_sol = np.zeros((2, 2, len(time_range)))  # Shape: (2, 2, number of time points)

# Populate folk_sol with the correct values for each time point


# Preallocate the array for the folk solution
folk_sol = np.zeros((len(time_range), 2, 2))

# Fill in the folk_sol array with calculated values
for t in range(len(time_range)):
    folk_sol[t] = np.array([
        [np.exp(np.sin(time_range[t])) * np.cos(1-np.cos(time_range[t])),
         -np.exp(np.sin(time_range[t])) * np.sin(1-np.cos(time_range[t]))],
        
        [np.exp(np.sin(time_range[t])) * np.sin(1-np.cos(time_range[t])),
         np.exp(np.sin(time_range[t])) * np.cos(1-np.cos(time_range[t]))]
    ])

print("folkers solution")
print(folk_sol)

plot_X_one_period(t_values, sol)

# Now you can plot q_adjusted for the time_range
plot_q(t_values, q_values)
plot_q(time_range, q_adjusted)

# Plot the solution x(t) over the time range
plot_x(time_range, x_t_values)

plot_x(time_range, y_t_values)
plot_x(time_range, folk_sol)
plt.show(block=True) 



'''
# Plot q(t)
plot_q(t_values, q_values)

# Plot the solution over one period
plot_X_one_period(t_values, sol)

plt.show(block=True) 
#plt.ioff()
'''