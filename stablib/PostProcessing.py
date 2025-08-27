# PostProcessing
# Plotting function for one period
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_freq_heatmap(participation_factors):
    """heatmap of participation factors """
    participation_factors=np.array(participation_factors)
    plt.imshow(participation_factors.T, cmap='viridis', aspect='auto')  # 'hot', 'gray', 'plasma', etc.
    plt.colorbar()
    plt.title("Heatmap of Matrix")
    plt.show()

def plot_peters(frequencies,participation_factor,epsilon,folder_name=None):
    """
    This function gets the participation factors for all the modes and plots them
    
    Parameters:

    - frequencies: all the frequencies evaluated in the fft. This is a number of magnitude ncomp
    - participation_factor: the normalized norm of each mode for each frequeny. this is a ncomp x n matrix
    - epsilon: is the total magnitude of all the modes. this is a len(ncomp) vector
    """

    # Generate timestamped folder name if not provided
    if folder_name is None:
        folder_name = '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder in the current working directory
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Get plot frequency size and problem ammount of participation factors
    ncomp, n = len(frequencies), participation_factor.shape[2]

    # Plot each participation factor component over time and save it
    
    plt.figure(figsize=(8, 6))
    for i in range(10): #Do we need all of the frequency content?
        for j in range(n):
            plt.plot(epsilon[i], participation_factor[i, j], label=f"PF {i},{j}")
            plt.xlabel("epsilon")
            plt.ylabel(f"participation factors")
            plt.title("Participation factors")
            plt.legend()
            
    # Save the figure to the folder without displaying it
    filename = os.path.join(folder_path, f"A_{i+1}_{j+1}.png")
    plt.savefig(filename)
    plt.close()  # Close the figure to avoid showing it

    print(f"Plots saved in folder: {folder_path}")

def plot_matrix(matrix, time, folder_name=None):
    """
    Plots each component of the solution against time and saves the images in a timestamped folder.
    
    Parameters:
    - solution: np.array of shape (len(time), n, m) where n x m is the matrix size at each time step.
    - time: np.array of time values.
    - folder_name: (Optional) Custom folder name. Defaults to current date and time if None.
    """
    # Generate timestamped folder name if not provided
    if folder_name is None:
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder in the current working directory
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Get matrix shape at each time step
    n, m = matrix.shape[1], matrix.shape[2]

    # Plot each matrix component over time and save it
    for i in range(n):
        for j in range(m):
            plt.figure(figsize=(8, 6))
            plt.plot(time, matrix[:, i, j], label=f"Element ({i+1},{j+1})")
            plt.xlabel("Time")
            plt.ylabel(f"A[{i+1},{j+1}]")
            plt.title(f"Solution Component A[{i+1},{j+1}] Over Time")
            plt.legend()
            
            # Save the figure to the folder without displaying it
            filename = os.path.join(folder_path, f"A_{i+1}_{j+1}.png")
            plt.savefig(filename)
            plt.close()  # Close the figure to avoid showing it

    print(f"Plots saved in folder: {folder_path}")

def plot_fft_norms(norm_fourier_coeffs, freqs, folder_name=None):
    """
    Plots a matrix of norms of fourier coefficients over a range of frequencies.

    Parameters:
    - norm_fourier_coeffs: np.array of shape (len(freqs), n, m) where n x m is the matrix size at each time step.
    - freqs: np.array of frequency values.
    - folder_name: (Optional) Custom folder name. Defaults to current date and time if None.
    """
    # Generate timestamped folder name if not provided
    if folder_name is None:
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder in the current working directory
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)

    n = norm_fourier_coeffs.shape[1]
    for i in range(n):  # loop over modes
        plt.figure(figsize=(8, 6))  # new figure for each mode
        for j in range(n):  # loop over state variables
            plt.plot(freqs, norm_fourier_coeffs[:, j, i], label=f"state variable ({j+1})")

        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.title(f"Mode [{i+1}] Frequency Decomposition")
        plt.legend()
        plt.tight_layout()
        plt.show()
            
            # Save the figure to the folder without displaying it
        filename = os.path.join(folder_path, f"Mode [{i+1}].png")
        plt.savefig(filename)
        plt.close()  # Close the figure to avoid showing it

def plot_control_panel(time, sandwich, norm_fourier_coeffs, freqs, max_index, max_index_full, folder_name=None):

        # Generate timestamped folder name if not provided
    if folder_name is None:
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder in the current working directory
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path, exist_ok=True)

    n_states = norm_fourier_coeffs.shape[1]
    n_modes = norm_fourier_coeffs.shape[2]

    for i in range(n_modes):  # loop over modes
        fig, axs = plt.subplots(n_states, 2, figsize=(8, 2.5 * n_states), sharex=False)

        for j in range(n_states):  # loop over state variables
            # Time domain signal
            axs[j, 0].plot(time, sandwich[:,j,i])
            axs[j, 0].set_ylabel(f"state {j+1}")
            axs[j, 0].set_title("Time Domain" if j == 0 else "")
            axs[j, 0].grid(True)

            #Frequency plots
            axs[j, 1].plot(freqs, norm_fourier_coeffs[:, j,  i])
            I = max_index_full[i]
            axs[j, 1].plot(freqs[I], norm_fourier_coeffs[I, j,  i], 'o')
            axs[j, 1].set_ylabel(f"State {j+1}")
            axs[j, 1].set_title("Frequency Domain" if j == 0 else "")
            axs[j, 1].grid(True)
            axs[j, 1].set_xlim([-3,3])
            axs[j, 1].set_yscale('log')


        axs[-1, 0].set_xlabel("Time")
        axs[-1, 1].set_xlabel("Frequency")
        
        fig.suptitle(f"Mode [{i+1}] Signal + Frequency Decomposition", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for title
        filename = os.path.join(folder_path, f"Mode_{i+1}_with_time_and_freq.png")
        plt.savefig(filename)
        plt.close()
