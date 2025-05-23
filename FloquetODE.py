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


REDO against analytical solution (try to plot/code the result)

x'=Ax
experiment with x_0

test the solution
step back and see if x(0)=x(T)

you know B! for this case
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def a(t):
    return np.array([[np.cos(t),-np.sin(t)], 
                     [np.sin(t), np.cos(t)]])

def ode_system(t,q,b):
    size=a(0).shape[0]
    q_matrix = q.reshape(size,size)  # Reshape q into a 2x2 matrix
    dqdt_matrix = np.dot(a(t), q_matrix) - np.dot(q_matrix, b)
    #print(dqdt_matrix)
    return dqdt_matrix.flatten()  # Return as a flattened array for the solver

def ODESolve(b_flat):
    size=a(0).shape[0]
    b = b_flat.reshape((size, size))

    #assume initial condition for q
    q0= np.eye(size).flatten()

    # Time span from 0 to 2*pi (for periodicity)
    t_span = (0, 2 * np.pi)
    t_eval = np.linspace(0, 2 * np.pi, 100)

    # Solve the system
    # print("this is the ode system\n")
    sol = solve_ivp(ode_system, t_span, q0, t_eval=t_eval, args=(b,))

    # Reshape the solution to a 2x2 matrix at each time point
    q_sol = sol.y.T.reshape(-1, size, size)

    # Check periodic boundary condition
    q0 = q_sol[0]  # q(0)
    q_final = q_sol[-1]  # q(2pi)
    
    return np.linalg.norm(q0-q_sol), q_sol, sol.t

# Objective function for the optimizer
def objective_function(b_flat):
    return ODESolve(b_flat)[0]  # Return only the error

print("a(0):", a(0))
print("Shape of a(0):", a(0).shape)

#b_init = np.eye(a(0).shape[0]).flatten()  # Identity matrix flattened for larger sizes
#b_init = np.zeros((2, 2)).flatten()
b_init = np.array([[0,1], 
                     [0, 1]]).flatten()

result = minimize(objective_function, b_init, method='BFGS')

if result.success:
    
    b_final = result.x.reshape(a(0).shape)  # Reshape the result back into 2x2 matrix
    print(f"Optimized b matrix:\n{b_final}")
    final_error, q_solution, time_points = ODESolve(b_final.flatten())
    print(f"Final error for periodic condition: {final_error}")
else:
    print("Optimization did not succeed.")

# Output the results
print(f"Solution shape: {q_solution.shape}")  # Should be (100, 2, 2) for 100 time points
print("q(t) at initial time point:\n", q_solution[0])
print("q(t) at final time point:\n", q_solution[-1])

#print("q solution\n",q_solution)

# Plot q(t) over time
time_points = np.linspace(0, 2 * np.pi, 100)
for i in range(q_solution.shape[1]):  # Loop over rows of q (0, 1 for 2x2 matrix)
    for j in range(q_solution.shape[2]):  # Loop over columns of q (0, 1 for 2x2 matrix)
        plt.plot(time_points, q_solution[:, i, j], label=f'q[{i+1},{j+1}]')

plt.title('Evolution of q(t) over time')
plt.xlabel('Time')
plt.ylabel('q(t)')
plt.legend()
plt.grid()
plt.show()