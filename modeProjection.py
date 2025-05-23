import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.fft import fft
import seaborn as sns
from PostProcessing import plot_fft_norms, plot_control_panel, plot_peters

def mode_projection(Q, V, t, plot=False):
    num_timesteps=len(t)
    n = V.shape[0]  # Number of modes
    P = np.zeros((num_timesteps, n, n), dtype=np.complex128)
    for time in range(num_timesteps):
        P[time,:,:]=np.linalg.inv(Q[time,:,:])

    basis = np.zeros((num_timesteps, n, n), dtype=np.complex128)  # Initialize the result matrix

    print('Getting "sandwich :)" (basis of eigenvectors)')
    for itime, time in enumerate(range(num_timesteps)):  # Iterate over time steps
        if np.mod(itime,50)==0:
            print(time, end=" ")
        for m in range(n):  # Iterate over columns
            for i in range(n):  # Iterate over rows
                sum_val = 0  # Accumulate the sum for the element (time, i, m)
                for j in range(n):  # Iterate over the intermediate dimension
                    sum_val += P[time, i, j] * V[j, m]
                basis[time, i, m] = sum_val  # Store the result
    print('')
    # Fourier coefficients storage (make sure this works correctly)
    freqs = np.fft.fftfreq(len(t), t[1]-t[0])
    freqs = np.fft.fftshift(freqs) #shift the frequencies
    ncomp= len(freqs)
    fourier_coefficients = np.zeros((ncomp, n, n), dtype=complex)  # (n, n, t)

    #each column is a mode, so each column has n elements that change with time (time series)
    for j in range(n):  # Iterate over columns (modes)
        for i in range(n):  # Iterate over rows
            phij = basis[:, i, j]  # Extract time series for each mode
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
    
    #if plot:
        #plot_fft_norms(normalized, freqs, folder_name='Freq_cont')
        #plot_control_panel(t, Q, normalized, freqs, max_index, max_index_full, folder_name=None)
    return max_vals, max_index, participation_factor

def mode_projection_old(Q, V, t, plot=True):
    num_timesteps=len(t)
    n = V.shape[0]  # Number of modes
    P = np.zeros((num_timesteps, n, n), dtype=np.complex128)
    for time in range(num_timesteps):
        P[time,:,:]=np.linalg.inv(Q[time,:,:])

    basis = np.zeros((num_timesteps, n, n), dtype=np.complex128)  # Initialize the result matrix

    print('Getting "sandwich :)" (basis of eigenvectors)')
    for itime, time in enumerate(range(num_timesteps)):  # Iterate over time steps
        if np.mod(itime,50)==0:
            print(time, end=" ")
        for m in range(n):  # Iterate over columns
            for i in range(n):  # Iterate over rows
                sum_val = 0  # Accumulate the sum for the element (time, i, m)
                for j in range(n):  # Iterate over the intermediate dimension
                    sum_val += P[time, i, j] * V[j, m]
                basis[time, i, m] = sum_val  # Store the result
    print('')
    # Fourier coefficients storage (make sure this works correctly)
    freqs = np.fft.fftfreq(len(t), t[1]-t[0])
    freqs = np.fft.fftshift(freqs) #shift the frequencies
    ncomp= len(freqs)
    fourier_coefficients = np.zeros((ncomp, n, n), dtype=complex)  # (n, n, t)

    #each column is a mode, so each column has n elements that change with time (time series)
    for j in range(n):  # Iterate over columns (modes)
        for i in range(n):  # Iterate over rows
            phij = basis[:, i, j]  # Extract time series for each mode
            #fft_coeffs = fft(phij) / num_timesteps  # Compute FFT and normalize
            fft_coeffs = np.fft.fft(phij)
            fft_coeffs = np.fft.fftshift(fft_coeffs)
            # fft_coeffs = np.fft.rfft(phij)is it cheating to keep it without the real because im still using max()
            
            fourier_coefficients[:, i, j] = fft_coeffs  # Store Fourier coefficients

    #after you get the fft of every element, you're supposed to get the norm of each mode. this means, every element of the vector!
    #Something like: Ci=[c1,c2,c3,..,cn]. vector=[C1,C2,C3,C4..,CN], norm(C)

    norms=np.zeros((ncomp,n, n))
    total_norms=np.zeros(n)
    normalized=np.zeros((ncomp,n, n))
    max_vals=np.zeros(n) #one value per mode
    max_index=np.zeros(n)
    max_index_full=np.zeros(n, dtype=int)
    ifreq0 = np.where(freqs==0)[0][0]

    for j in range(n):
        for i in range(n):
            norms[:,i,j]=np.abs(fourier_coefficients[:,i,j]) #heres all the fourier coeficient norms for all frequencies
            
        total_norms[j]=np.sum(norms[:,:,j])    #Now we want to normalize them [Because Riva/Bottasso/Caccaiola say so :)]
        normalized[:,:,j]=norms[:,:,j]/total_norms[j]
        max_vals[j]=np.max(normalized[:,:,j])
        max_index_full[j]=np.argmax(np.max(normalized[:, :, j], axis=1))
        max_index[j] =  max_index_full[j] - ifreq0
    
    #if plot:
        #plot_fft_norms(normalized, freqs, folder_name='Freq_cont')
        #plot_control_panel(t, Q, normalized, freqs, max_index, max_index_full, folder_name=None)

    

    return max_vals, max_index

def check_real(vector, name="vector"):
    if not np.isrealobj(vector):
        raise ValueError(f"{name} contains complex values.")
    