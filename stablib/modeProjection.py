import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft

#from stablib.PostProcessing import plot_fft_norms, plot_control_panel, plot_peters

def ensure_3d(mat, T):
    """
    Convert mat to 3D by:
    - If 1D: treat as (length,) and reshape to (1, length, 1), then repeat over T
    - If 2D: treat as (n, m) and reshape to (1, n, m), then repeat over T
    - If 3D: check first dim == T
    """
    if mat.ndim == 1:
        mat3d = np.repeat(mat[np.newaxis, :, np.newaxis], T, axis=0)  # (T, length, 1)
    elif mat.ndim == 2:
        mat3d = np.repeat(mat[np.newaxis, :, :], T, axis=0)           # (T, n, m)
    elif mat.ndim == 3:
        if mat.shape[0] != T:
            raise ValueError(f"3D input first dimension {mat.shape[0]} != expected T={T}")
        mat3d = mat
    else:
        raise ValueError(f"Unsupported number of dims: {mat.ndim}")
    return mat3d

def time_multiply(mat1, mat2, method="forloop"):
    '''
    For this multiplication, mat1 and mat2 must have 3 dimensions:
        matx.shape[0]=timesteps
        matx.shape[1]=rows
        matx.shape[2]=columns

    multiplication between matrices is only possible when mat1.shape[2]=mat2.shape[1]
    '''
    # if mat1.shape[2]!=mat2.shape[1]:
    #     print("Matrices have incompatible size")
    #     raise ValueError("Matrices have incompatible size")
    # elif mat1.shape[0]!=mat2.shape[0]:
    #     print("Matrices hace different time size")
    #     raise ValueError("Matrices hace different time size")
    # else:
    #     pass


    mat1_is_3d = (mat1.ndim == 3)
    mat2_is_3d = (mat2.ndim == 3)

    if mat1_is_3d and mat2_is_3d:
        T1 = mat1.shape[0]
        T2 = mat2.shape[0]
        if T1 != T2:
            raise ValueError(f"Time dimension mismatch: mat1 has {T1}, mat2 has {T2}")
        T = T1
    elif mat1_is_3d:
        T = mat1.shape[0]
    elif mat2_is_3d:
        T = mat2.shape[0]
    else:
        T = 1

    mat1 = ensure_3d(mat1, T)
    mat2 = ensure_3d(mat2, T)

    if mat1.shape[2]!=mat2.shape[1]:
        raise ValueError(f"dimension mismatch: mat1 has {mat1.shape[2]}, mat2 has {mat2.shape[1]}")
    
    n_rows = mat1.shape[1]     # A
    n_inner = mat1.shape[2]    # B
    n_cols = mat2.shape[2]     # C

    num_timesteps = mat1.shape[0]

    if method =="forloop":
        result = np.zeros((num_timesteps, n_rows, n_cols), dtype=np.complex128)

        for time in range(num_timesteps):  # Iterate over time steps
            if time % 50 == 0:
                print(time, end=" ")
            for j in range(n_cols):  # Output columns
                for i in range(n_rows):  # Output rows
                    sum_val = 0
                    for k in range(n_inner):  # Inner dimension
                        sum_val += mat1[time, i, k] * mat2[time,k, j]
                    result[time, i, j] = sum_val

        print('')

    elif method =="einsum":
        result=0
        pass

    elif method =="@":
        result=0
        pass

    return result


def mode_projection(C, Q, V, t, plot=False, debug_check=True):

    #Ct_array = np.zeros((num_timesteps, n, n), dtype=np.complex128) 
    #Ct_arrar[it, :, :] = Ct(t[i])

    num_timesteps=len(t)
    n = V.shape[0]  # Number of modes
    P = np.zeros((num_timesteps, n, n), dtype=np.complex128)
    for time in range(num_timesteps):
        P[time,:,:]=np.linalg.inv(Q[time,:,:])

    basis = np.zeros((num_timesteps, n, n), dtype=np.complex128)  # Initialize the result matrix

    print('Getting "sandwich :)" (basis of eigenvectors)')
    # M1 = time_multiply(method="einsum")
    # if debug_check:
    #     M2 = time_multiply(method="forloop")
    # np.ttesting.assert_almost_equal(M1, M2, 8)

    # for itime, time in enumerate(range(num_timesteps)):  # Iterate over time steps
    #     if np.mod(itime,50)==0:
    #         print(time, end=" ")
    #     for m in range(n):  # Iterate over columns
    #         for i in range(n):  # Iterate over rows
    #             sum_val = 0  # Accumulate the sum for the element (time, i, m)
    #             for j in range(n):  # Iterate over the intermediate dimension
    #                 sum_val += P[time, i, j] * V[j, m]
    #             basis[time, i, m] = sum_val  # Store the result
    # print('')

    # # sanitybasis=time_multiply(P, V, method="forloop")
    basis=time_multiply(P, V, method="forloop")

    # if np.allclose(basis, sanitybasis, atol=1e-3):
    #    print("Multiplication is good")

    # Fourier coefficients storage (make sure this works correctly)
    freqs = np.fft.fftfreq(len(t), t[1]-t[0])
    freqs = np.fft.fftshift(freqs) #shift the frequencies
    ncomp= len(freqs)
    fourier_coefficients = np.zeros((ncomp, n, n), dtype=complex)  # (n, n, t)

    out_spec_basis=time_multiply(C, basis, method="forloop")
    out_spec_basis_sanity=C @ basis 

    if np.allclose(out_spec_basis, out_spec_basis_sanity, atol=1e-3):
        print("Multiplication is good")

    #each column is a mode, so each column has n elements that change with time (time series)
    for j in range(n):  # Iterate over columns (modes)
        for i in range(out_spec_basis.shape[1]):  # Iterate over rows
            phij = out_spec_basis[:, i, j]  # Extract time series for each mode
            #fft_coeffs = fft(phij) / num_timesteps  # Compute FFT and normalize
            fft_coeffs = np.fft.fft(phij)
            fft_coeffs = np.fft.fftshift(fft_coeffs)
            # fft_coeffs = np.fft.rfft(phij)is it cheating to keep it without the real because im still using max()
            
            fourier_coefficients[:, i, j] = fft_coeffs  # Store Fourier coefficients

    '''
    #after you get the fft of every element, you're supposed to get the norm of each mode.
    # So according to Peters, that is for each frequency: the norms of the state variables of each mode
    '''
    norms=np.zeros((ncomp, n))
    total_norms=np.zeros((ncomp)) #this is epsilon according to peters
    participation_factor=np.zeros((ncomp,n))
    max_vals=np.zeros(ncomp) #one value per frequency
    max_index=np.zeros(n)
    max_index_full=np.zeros(ncomp, dtype=int)
    ifreq0 = np.where(freqs==0)[0][0]

    for freq in range(ncomp):
        for j in range(n):
            norms[freq,j]=np.linalg.norm(fourier_coefficients[freq,:,j]) #heres all the fourier coeficient norms for one frequency 

        total_norms[freq]=np.sum(norms[freq,:])    #This is what Peters calls epsilon
        for j in range(n):
            participation_factor[freq,j]=norms[freq,j]/total_norms[freq]

    '''
    if plot:
        plot_peters(freqs,participation_factor,total_norms)
        #add plot function here

    
    
    for freq in range(ncomp):
        max_vals=np.max(participation_factor[freq,:])
        max_index_full[freq]=np.argmax(np.max(participation_factor[freq, :]))
        max_index[freq] =  max_index_full[freq] - ifreq0
    '''
    
    start_row = int(np.ceil(ncomp / 2))
    # Slice the 2D array (rows only)
    # slice_pf = participation_factor[start_row:-1, :]
    slice_pf = participation_factor

    
    max_col_indices=np.zeros(n)
    for deg_free in range(n):
        max_col_indices[deg_free] = np.argmax(slice_pf[:,deg_free])-ifreq0
    
    max_values = np.max(participation_factor, axis=1)
    print('max indices are:',max_col_indices)
    #if plot:
        #plot_fft_norms(normalized, freqs, folder_name='Freq_cont')
        #plot_control_panel(t, Q, normalized, freqs, max_index, max_index_full, folder_name=None)
    return max_values, max_col_indices, participation_factor

def check_real(vector, name="vector"):
    if not np.isrealobj(vector):
        raise ValueError(f"{name} contains complex values.")
    
