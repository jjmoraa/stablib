import numpy as np
import pandas as pd

# Function declaration must come before any calls to it
def spaceStateTransformation(mass, damping, stiffness):
    A = np.concatenate(
        (np.concatenate((np.zeros(np.shape(mass)), np.eye(len(mass))), axis=1),
        np.concatenate((-np.linalg.inv(mass) @ stiffness, -np.linalg.inv(mass) @ damping), axis=1)),
        axis=0
    )
    return A

# Define constants
m = 500
l = 30
M = 50000
# omega = 0
edgNatFreq_hz = 0.8  # Edgewise frequency in Hz
edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi  # Convert to rad/s
kx = 200000
ky = 200000

omegas = np.linspace(0.0, 1, 100)
eigenvalues_for_range=[]
mode_frequencies=[]
mode_decay=[]
damping_ratio=[]

for iom, omega in enumerate(omegas):
    # Define the mass matrix
    mass = [[m * l**2, 0, 0, 0, 0],
            [0, m * l**2, 0, m * l, 0],
            [0, 0, m * l**2, 0, -m * l],
            [0, 3 * m * l / 2, 0, M + 3 * m, 0],
            [0, 0, -3 * m * l / 2, 0, M + 3 * m]]


    # Define the stiffness matrix
    stiffness = [[m * (l**2) * (edgNatFreq_rad**2), 0, 0, 0, 0],
            [0, m * (l**2) * (edgNatFreq_rad**2) - m * (l**2) * (omega**2), 0, 0, 0],
            [0, 0, m * (l**2) * (edgNatFreq_rad**2) - m * (l**2) * (omega**2), 0, 0],
            [0, 0, 0, kx, 0],
            [0, 0, 0, 0, ky]]
    
    stiffness=np.array(stiffness)
    mass=np.array(mass)

    kdiag=np.diag(stiffness)
    mdiag=np.diag(mass)
    zeta_diag=np.array([0.1, 0.1,  0.1, 0.4, 0.5])
    slope=np.array([2 ,0, 0, 0, 0])
    ca = 2*zeta_diag * np.sqrt(kdiag*mdiag)
    c_diag = ca*(1  -  slope*omega) 
        
    # Define the damping matrix
    damping = [[0, 0, 0, 0, 0],
            [0, 0, 2 * m * (l**2) * omega, 0, 0],
            [0, -2 * m * (l**2) * omega, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
    damping = np.array(damping)+ np.diag(c_diag)
    
    I=[0, 1, 2 ,3, 4]
    stiffness=np.array(stiffness)[np.ix_(I,I)]
    damping=np.array(damping)[np.ix_(I,I)]
    mass=np.array(mass)[np.ix_(I,I)]

    #print('M\n', mass)
    #print('C\n', damping)
    
    # Call the function after it has been defined
    A = spaceStateTransformation(mass, damping, stiffness)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Uncomment to print matrices
    # print(mass)
    # print(damping)
    # print(stiffness)
    # print(A)
    # naturalFrequencies = np.sqrt(np.real(eigenvaluesA))

    # natural_frequencies = np.sqrt(np.real(eigenvalues))

    # print(eigenvalues)
    # print("Natural Frequencies (rad/s):", natural_frequencies)
    # print("Natural Frequencies (Hz):", natural_frequencies/(2*np.pi))
    # print("Mode Shapes:\n", eigenvectors)
    # print(naturalFrequencies)

    data = {
        'Eigenvalue': eigenvalues,}

    for i in range(len(mass)):
        data[f'Eigenvector_{i+1}'] = eigenvectors[:, i]

    df = pd.DataFrame(data)
    df.to_csv('monodromy_eigenvalues_eigenvectors_{omega}.csv', index=False)



    omega_n = np.sqrt(np.real(eigenvalues)**2+np.imag(eigenvalues)**2)
    fd = np.imag(eigenvalues)/(2*np.pi) # damped frequencies in Hz
    fn = omega_n/(2*np.pi) # natural frequencies
    zeta = -np.real(eigenvalues)/omega_n  # damping ratio
    #print('fn:', fn.sort()[0:3])
    #print('ze:', fn.sort()[0:3])
    eigenvalues_for_range.append(eigenvalues)
    mode_frequencies.append(fn)  #Normally print fn instead of fd
    mode_decay.append(np.real(eigenvalues))
    damping_ratio.append(zeta)
    #print('zeta')
    #print(zeta)
    #print('fd\n', fd)
    
#print('eigenvectors')
#print(eigenvectors)

#print('eigenvalues for range')
#print(eigenvalues_for_range)
# Create the DataFrame for the Campbell diagram (mode frequencies)
data_frequencies = np.column_stack(
    [omegas,  # Omega values (assuming same scale as the loop)
     np.array(mode_frequencies)]  # Convert list of arrays to a 2D array
)
cols_frequencies = ['Omega'] + [f'mode_frequencies({i+1})' for i in range(data_frequencies.shape[1] - 1)]
df_frequencies = pd.DataFrame(data_frequencies, columns=cols_frequencies)
df_frequencies.to_csv('Campbell_diagram_rinker.csv', index=False)

# Create the DataFrame for the stability diagram (mode decay)
data_decay = np.column_stack(
    [omegas,  # Omega values
     np.array(mode_decay)]  # Convert list of arrays to a 2D array
)
cols_decay = ['Omega'] + [f'mode_decay({i+1})' for i in range(data_decay.shape[1] - 1)]
df_decay = pd.DataFrame(data_decay, columns=cols_decay)
df_decay.to_csv('stability_diagram_rinker.csv', index=False)

data_damping_ratio = np.column_stack(
    [omegas,  # Omega values
     np.array(damping_ratio)]  # Convert list of arrays to a 2D array
)
cols_damping_ratio = ['Omega'] + [f'damping_ratio({i+1})' for i in range(data_damping_ratio.shape[1] - 1)]
df_damping_ratio = pd.DataFrame(data_damping_ratio, columns=cols_damping_ratio)
df_damping_ratio.to_csv('damping_ratio_rinker.csv', index=False)
