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
        #folder_name = '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = '_' + datetime.now().strftime("%Y-%m-%d")
    else:
        #folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = folder_name + "_" + datetime.now().strftime("%Y-%m-%d")

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
        folder_name = '_' +datetime.now().strftime("%Y-%m-%d")
    else:
        folder_name = '_' +folder_name + "_" + datetime.now().strftime("%Y-%m-%d")

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
        folder_name = '_' + datetime.now().strftime("%Y-%m-%d")
    else:
        folder_name = '_' + folder_name + "_" + datetime.now().strftime("%Y-%m-%d")

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
        folder_name = '_' + datetime.now().strftime("%Y-%m-%d")
    else:
        folder_name = '_' + folder_name + "_" + datetime.now().strftime("%Y-%m-%d")

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

def evaluateStabilityMonodromy(eigenvalues_mon, doPlot=True):
    """
    Evaluates Floquet stability from monodromy eigenvalues.

    Parameters
    ----------
    eigenvalues_mon : array-like
        Monodromy matrix eigenvalues
    doPlot : bool
        If True, plots eigenvalues on the complex plane with unit circle

    Returns
    -------
    stabilityReport : dict
        Dictionary with:
            'isStable' : True if all eigenvalues inside unit circle
            'maxModulus' : maximum |lambda|
            'unstableEigenvalues' : eigenvalues outside the unit circle
    """
    eigenvalues_mon = np.array(eigenvalues_mon)
    modEigen = np.abs(eigenvalues_mon)
    maxMod = np.max(modEigen)
    unstableEigenvalues = eigenvalues_mon[modEigen > 1]
    isStable = maxMod <= 1

    print(f"Maximum eigenvalue modulus: {maxMod:.4f}")
    if isStable:
        print("System is stable: all eigenvalues inside the unit circle.")
    else:
        print(f"System is unstable: {len(unstableEigenvalues)} eigenvalue(s) outside the unit circle.")
        print("Unstable eigenvalues:", unstableEigenvalues)

    if doPlot:
        theta = np.linspace(0, 2*np.pi, 500)
        plt.figure()
        plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1.5)
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.gca().set_aspect('equal', 'box')
        plt.grid(True)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Floquet Eigenvalues on the Complex Plane')

        stable_idx = modEigen <= 1
        unstable_idx = ~stable_idx

        plt.plot(eigenvalues_mon[stable_idx].real, eigenvalues_mon[stable_idx].imag, 'go', markersize=8, label='Stable Eigenvalues')
        plt.plot(eigenvalues_mon[unstable_idx].real, eigenvalues_mon[unstable_idx].imag, 'ro', markersize=8, label='Unstable Eigenvalues')
        plt.legend()
        plt.show()

    stabilityReport = {
        'isStable': isStable,
        'maxModulus': maxMod,
        'unstableEigenvalues': unstableEigenvalues
    }

    return stabilityReport

def plotCampbellDiagram(parameter, freqs_Hz, y_label, save_path=None):

    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11)

    parameter = np.array(parameter)                # Shape = (cases, modes)
    n_cases, n_modes = parameter.shape
    COLRS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ---- plot only ONE branch across cases ---- #
    for mode in range(n_modes):
        mode_freqs = parameter[:, mode]
        ax.plot(freqs_Hz, mode_freqs, '-o',
                color=COLRS[mode % len(COLRS)],
                label=f"Mode {mode+1}")

    ax.set_xlabel('Rotor speed [Hz]')
    ax.set_ylabel(y_label)
    ax.set_title(f'Campbell Diagram — {y_label}')
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    # --- try to save safely ---
    if save_path:
        try:
            fig.savefig(save_path, dpi=300)
            print(f"Saved to {save_path}")
        except Exception as e:
            print(f"⚠ Could not save figure — {e}")

    plt.show()



def plotCampbellDiagramMultipleHarmonics(vf_0, freqs_Hz, var_name='Frequency [Hz]', save_path='Campbell.pdf'):
    """
    Plots all harmonic quantities (e.g., frequencies, damping, participation) from Floquet modal analysis.

    Parameters
    ----------
    vf_0 : list of 2D arrays
        Quantity for each operating point, shape: [n_cases][n_harmonics, n_modes]
    freqs_Hz : array-like
        Rotor speeds (Hz) corresponding to vf_0
    var_name : str, optional
        Label for the plotted variable (used for y-axis label and title)
    save_path : str, optional
        File path to save the figure
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.11)

    COLRS = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_cases = len(vf_0)
    n_harmonics = vf_0[0].shape[0]

    # Plot each harmonic with its own color
    for h_idx in range(n_harmonics):
        color = COLRS[h_idx % len(COLRS)]
        x_points, y_points = [], []

        # Collect data across all operating points
        for i_case in range(n_cases):
            freqs_this = np.full(vf_0[i_case][h_idx, :].shape, freqs_Hz[i_case])
            x_points.append(freqs_this)
            y_points.append(vf_0[i_case][h_idx, :])

        x_points = np.concatenate(x_points)
        y_points = np.concatenate(y_points)

        ax.plot(
            x_points,
            y_points,
            marker='o',       # dots at each point
            linestyle='-',    # connect dots with lines
            markersize=5,     # size of the dots
            alpha=0.8,
            color=color,
            label=f'Harmonic {h_idx + 1}'
        )


    ax.set_xlabel('Rotor speed [Hz]')
    ax.set_ylabel(var_name)
    ax.set_title(f'Campbell Diagram ({var_name})')
    ax.grid(True, ls='--', alpha=0.5)

    # Autoscale limits
    ax.set_xlim([0, np.max(freqs_Hz) * 1.05])
    all_values = np.concatenate([arr.ravel() for arr in vf_0])
    ax.set_ylim([0, np.max(all_values) * 1.1])

    ax.legend(title='Harmonics', fontsize='small', loc='best')
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    # plt.close(fig)

def plotCampbellDiagramAllModesSingleHarmonic(
    vf_0,
    freqs_Hz,
    harmonic_idx,
    var_name='Frequency [Hz]',
    save_path=None
):
    """
    Plots all modes for a single harmonic across operating points as dots connected by lines.

    Parameters
    ----------
    vf_0 : list of 2D arrays
        Shape: [n_cases][n_harmonics, n_modes]
    freqs_Hz : array-like
        Rotor speeds (Hz)
    harmonic_idx : int
        Harmonic index to plot (0-based)
    var_name : str
        Y-axis label
    save_path : str or None, optional
        If provided, save figure to this path. If None, figure is not saved.
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.11)

    n_cases = len(vf_0)
    n_modes = vf_0[0].shape[1]

    COLRS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for mode_idx in range(n_modes):
        x_points = [freqs_Hz[i_case] for i_case in range(n_cases)]
        y_points = [vf_0[i_case][harmonic_idx, mode_idx] for i_case in range(n_cases)]

        # Plot line with dots
        ax.plot(
            x_points,
            y_points,
            marker='o',       # show dots
            linestyle='-',    # connect with lines
            markersize=5,
            alpha=0.8,
            color=COLRS[mode_idx % len(COLRS)],
            label=f'Mode {mode_idx + 1}'
        )

    ax.set_xlabel('Rotor speed [Hz]')
    ax.set_ylabel(var_name)
    ax.set_title(f'Campbell Diagram – Harmonic {harmonic_idx}')
    ax.grid(True, ls='--', alpha=0.5)

    ax.set_xlim([0, max(freqs_Hz) * 1.05])

    all_vals = np.concatenate([arr[harmonic_idx, :] for arr in vf_0])
    ax.set_ylim([0, 1.1 * np.max(all_vals)])

    ax.legend(title='Modes', fontsize='small', loc='best')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()
