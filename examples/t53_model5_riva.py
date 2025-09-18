# -*- coding: utf-8 -*-
# %% Import.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import eig, inv, solve, expm
from scipy.signal import welch

plt.close("all")

# %% Analytical model.

def phi(omega,t):
    return np.array([omega*t,omega*t+2*np.pi/3,omega*t+4*np.pi/3])

def mass(M, m, l, omega, t):
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
    cb = 0.0
    ct = 0.0
    return np.array([
        [cb, 0, 0, 0, 0],
        [0, cb, 0, 0, 0],
        [0, 0, cb, 0, 0],
        [-np.sin(phi_vals[0]), -np.sin(phi_vals[1]), -np.sin(phi_vals[2]), ct, 0],
        [-np.cos(phi_vals[0]), -np.cos(phi_vals[1]), -np.cos(phi_vals[2]), 0, ct]
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

def A_fromMCK(M, C, K):
    mass_inv = np.linalg.inv(M)
    return np.block([
        [np.zeros_like(M), np.eye(M.shape[0])],
        [-mass_inv @ K, -mass_inv @ C]]) 

def ro_riva(time_stm,At, C, rtol=1e-6, period=1):

    A0 = At(0)
    nx = A0.shape[0]
    tm0 = np.eye(nx)

    # Integrate Phi' = A @ Phi.
    sol_stm = solve_ivp(
        fun=lambda t, stm: (At(t) @ stm.reshape(nx, nx)).reshape(-1),
        t_span=(time_stm[0], time_stm[-1]),
        y0=tm0.reshape(-1),
        t_eval=time_stm,
        vectorized=True,
        rtol=rtol,
    )

    # Plot state transition matrix.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State transition matrix")
        ax.plot(sol_stm.t, sol_stm.y.T)

    # Reshape to 3D array.
    # The state transition matrix, stm, is ordered as:
    #  - axis 0, 1: state
    #  - axis 2: time
    stm = np.reshape(sol_stm.y, (nx, nx, sol_stm.t.size))

    # Get monodromy matrix.
    monodromy = stm[:, :, -1]

    # Compute characteristic multipliers.
    theta, S = eig(monodromy)

    # Plot characteristic multipliers.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_xlim(-1.1, +1.1)
        ax.set_ylim(-1.1, +1.1)
        ax.set_aspect("equal")
        ax.add_artist(plt.Circle(xy=(0.0, 0.0), radius=1.0, edgecolor="k", fill=False))
        for i in range(theta.size):
            ax.scatter(theta[i].real, theta[i].imag, label=f"Mode {i+1}")
        ax.legend(loc="center")

    # Define shift for characteristic exponents.
    if time_stm.size % 2 == 0:
        max_shift = int(time_stm.size / 2)
        shift = np.arange(-max_shift, +max_shift)
    else:
        max_shift = int((time_stm.size - 1) / 2)
        #min_shift = -int((time_stm.size - 1) / 2)
        #max_shift =  int((time_stm.size - 1) / 2)-1
        
        shift = np.arange(-max_shift, +max_shift)
    shift0 = max_shift  # n = 0.
    #print(f'T is of length({len(time_stm)})')
    #shift2 = np.fft.fftfreq(len(time_stm)-1)*(len(time_stm)-1)
    #shift3=shift2.copy()
    #shift3.sort()
    #print('')
    # Compute characteristic exponents.

    # eta is ordered as:
    #  - axis 0: harmonics.
    #  - axis 1: modes.
    abs_theta = np.abs(theta)
    ang_theta = np.angle(theta)
    eta = (np.log(abs_theta) + 1j * ang_theta)[np.newaxis, :] / period + 2j * np.pi / period * shift[:, np.newaxis]
    # Plot characteristic exponents.
    if False:
        fig, ax = plt.subplots()
        ax.set_xlabel("Real [rad/s]")
        ax.set_ylabel("Imag [rad/s]")
        ax.set_xlim(-0.04, +0.02)
        ax.set_ylim(-3.2, +3.2)
        for i in range(eta.shape[1]):
            ax.scatter(eta[:, i].real, eta[:, i].imag, label=f"Mode {i}")
        ax.legend()

    # Compute frequency and damping.
    natural_frequency = np.abs(eta)  # [rad/s]
    damping_ratio = -eta.real / natural_frequency  # [-]
    natural_frequency /= 2 * np.pi  # [Hz]
    damped_frequency = eta.imag / (2 * np.pi)  # [Hz]. Can be positive or negative (useful later).

    # Set initial value of periodic transformation.
    P0 = np.eye(nx)

    # Compute eigenvectors of Floquet factor.
    V = S  # Trivial since P0 = I.

    # Compute Floquet factor.
    # R = V @ np.diag(eta[shift0, :]) @ inv(V)
    R = np.transpose(solve(V.T, np.diag(eta[shift0, :]) @ V.T))

    # Compute periodic transformation.
    # Time must be the first dimension to allow computing Xi using matmul.
    # P is periodic, therefore we skip the last time instant.
    P = np.zeros((time_stm.size - 1, nx, nx), dtype=complex)
    invS = inv(S)  # should use solve.

    for k in range(P.shape[0]):
        # P[k, :, :] = stm[:, :, k] @ P0 @ expm(- R * time_stm[k])
        # P[k, :, :] = stm[:, :, k] @ S @ np.diag(np.exp(-eta[shift0, :] * time_stm[k])) @ invS @ P0
        P[k, :, :] = stm[:, :, k] @ S * np.exp(-eta[shift0, :] * time_stm[k]) @ invS @ P0

    # Compute Xi.
    Xi = C[np.newaxis, :, :] @ P @ V[np.newaxis, :, :]

    # Expand Xi in Fourier series. psi contains the mode shapes, and is ordered as:
    # axis 0: harmonics.
    # axis 1: states or outputs if C is used.
    # axis 2: modes.
    psi = np.fft.fft(Xi, axis=0) / Xi.shape[0]

    # Sort harmonics from -n to +n.
    psi = np.fft.fftshift(psi, axes=0)

    #dt = time_stm[1] - time_stm[0]
#     freqs = np.fft.fftfreq(len(t), dt)
#     freqs = np.fft.fftshift(freqs) #shift the frequencies

    # Compute mode shapes norm.
    participation = np.linalg.norm(psi, ord=2, axis=1)

    # Normalize across the harmonics to get the output-specific participation factors.
    participation /= participation.sum(axis=0)

    # Find the principal harmonic for each mode.
    n_principal = np.argmax(participation, axis=0)

    f0_m1 = np.full(nx, np.nan)
    f0_principal = f0_m1.copy()
    f0_p1 = f0_m1.copy()
    for ix in range(nx):
        # We skip modes with negative damping frequency.
        if damped_frequency[n_principal[ix], ix] < 0.0:
            continue
        f0_m1[ix] = natural_frequency[n_principal[ix]-1, ix]
        f0_principal[ix] = natural_frequency[n_principal[ix], ix]
        f0_p1[ix] = natural_frequency[n_principal[ix]+1, ix]



    d = dict()
    d['R'] = R
    d['P'] = P
    d['V'] = V
    d['S'] = S
    d['theta'] = theta
    d['eta'] = eta
    d['monodromy'] = monodromy
    d['sol_stm'] = sol_stm
    #d['sol_free'] =sol_free
    d['Xi'] = Xi
    d['n_principal'] = n_principal
    d['damped_frequency'] = damped_frequency
    d['natural_frequency'] = natural_frequency
    d['psi'] = psi
    #d['freqs'] = freqs
    d['participation'] = participation
    d['f_0_principal'] = f0_principal
    d['f_0_m1']        = f0_m1
    d['f_0_p1']        = f0_p1
    return d
    
    
if __name__ == "__main__":

    # --- Parameters
    m = 500   # Blade mass
    l = 30    # Blade length
    M = 50000 # Nacelle mass
    edgNatFreq = 0.8 * 2 *np.pi  # Edgewise frequency in rad
    kx = 200000 # Support stiffness
    ky = 250000
    # ky = kx
    omegas_rpm = np.linspace(0.1, 30.0, 40) # Rotor speed.
    omegas = omegas_rpm * np.pi / 30.0  # Rotor speed [rad/s]
    nx = 10  # Number of states.

    # %% Simulation.
    # Initial condition.
    x0 = np.zeros((nx,))
    # x0[:5] = 1.0
    x0[0] = 1.0
    x0[1] = 1.0
    x0[2] = 1.0
    # x0[3] = 1.0
    # x0[4] = 1.0

    # Set rotor speed.

    omega_rpm = 10.0
    omega = omega_rpm * np.pi / 30.0  # [rad/s]

    # Functions of time

    Mt = lambda t: mass(M, m, l, omega, t)
    Kt = lambda t: stiffness(edgNatFreq, m, l, kx, ky, omega, t)
    Ct = lambda t: damping(omega, t)
    At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t))
    sys = lambda t, x: At(t) @ x

    # Simulate free response.

    time_simulation = np.linspace(0.0, 1000.0, 10001)
    dt_simulation = time_simulation[1]
    df_simulation = 1.0 / time_simulation[-1]
    sampling_frequency_simulation = 1.0 / dt_simulation
    nyquist_frequency_simulation = sampling_frequency_simulation / 2.0
    simulation = solve_ivp(sys, (time_simulation[0], time_simulation[-1]), x0,
                        vectorized=True,
                        rtol=1e-6,
                        t_eval=time_simulation)
    

    if False:
        fig, ax = plt.subplots(dpi=300)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Output")
        ax.plot(simulation.t, simulation.y[0, :], label="Blade 0")
        ax.plot(simulation.t, simulation.y[1, :], label="Blade 1")
        ax.plot(simulation.t, simulation.y[2, :], label="Blade 2")
        ax.plot(simulation.t, simulation.y[3, :], label="Tower x")
        ax.plot(simulation.t, simulation.y[4, :], label="Tower y")
        ax.legend()

    # Compute PSD.
    window = np.ones(simulation.t.size - 1)
    noverlap = 0
    nperseg = len(window)
    nfft = nperseg
    detrend = False
    return_onesided = True
    scaling = "density"

    simulation_freq, simulation_PSD = welch(
                simulation.y,
                sampling_frequency_simulation,
                window,
                nperseg,
                noverlap,
                nfft,
                detrend,
                return_onesided,
                scaling,
                axis=1,
            )

    fig, ax = plt.subplots()
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Output")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.5)
    ax.grid(True)
    ax.plot(simulation_freq, simulation_PSD[0, :], label="Blade 0")
    ax.plot(simulation_freq, simulation_PSD[1, :], label="Blade 1")
    ax.plot(simulation_freq, simulation_PSD[2, :], label="Blade 2")
    ax.plot(simulation_freq, simulation_PSD[3, :], label="Tower x")
    ax.plot(simulation_freq, simulation_PSD[4, :], label="Tower y")
    ax.legend()

    # %% Floquet.

    # Preallocate arrays to collect the principal harmonic natural frequency for all rotor speeds.
    # Since this system is isotropic, we know that it has only the -1, principal and +1 harmonics.
    # By definition, the principal harmonicis observed in the ground-fixed frame.

    campbell_m1 = np.full((omegas.size, nx), np.nan)
    campbell_principal = campbell_m1.copy()
    campbell_p1 = campbell_m1.copy()


    # Loop over rotor speed.

    for iom in range(omegas.size):
        omega = omegas[iom]
        period = 2*np.pi/omega

        # Functions of time
        Mt = lambda t: mass(M, m, l, omega, t)
        Kt = lambda t: stiffness(edgNatFreq, m, l, kx, ky, omega, t)
        Ct = lambda t: damping(omega, t)
        At = lambda t: A_fromMCK(Mt(t), Ct(t), Kt(t) )

        # C matrix in the 1st order system.
        # Output the tower.

        out_mat = np.zeros((2, 10))
        out_mat[0, 3] = 1.0
        out_mat[1, 4] = 1.0

        # Output blade 0
        # out_mat = np.zeros((1, 10))
        # out_mat[0, 0] = 1.0

        # Apply Floquet theory.
        # 1 period.

        time_stm = np.linspace(0.0, period, 1001)
        dt_stm = time_stm[1]
        df_stm = 1.0 / time_stm[-1]
        sampling_frequency_stm = 1.0 / dt_stm
        nyquist_frequency_stm = sampling_frequency_stm / 2.0
        # Set initial condition.

        tm0 = np.eye(nx)
        
        d = ro_riva(time_stm, At, out_mat, rtol=1e-6, period=period)

        n_principal = d['n_principal']
        damped_frequency = d['damped_frequency']
        natural_frequency = d['natural_frequency']

        # Store harmonics.

        for ix in range(nx):
            # We skip modes with negative damping frequency.
            if damped_frequency[n_principal[ix], ix] < 0.0:
                continue
            campbell_m1[iom, ix] = natural_frequency[n_principal[ix]-1, ix]
            campbell_principal[iom, ix] = natural_frequency[n_principal[ix], ix]
            campbell_p1[iom, ix] = natural_frequency[n_principal[ix]+1, ix]


    # Plot Campbell diagram.
    fig, ax = plt.subplots()
    ax.set_xlabel("Rotor speed [rpm]")
    ax.set_ylabel("Natural frequency [Hz]")
    ax.grid(True)
    ax.set_yticks(np.arange(0.0, 2.1, 0.25))
    ax.set_ylim(0.0, 1.5)
    ax.scatter(np.broadcast_to(omegas_rpm[:, np.newaxis], campbell_m1.shape), campbell_m1, color="C1", label="-1")
    ax.scatter(np.broadcast_to(omegas_rpm[:, np.newaxis], campbell_principal.shape), campbell_principal, color="C0", label="Principal")
    ax.scatter(np.broadcast_to(omegas_rpm[:, np.newaxis], campbell_p1.shape), campbell_p1, color="C2", label="+1")
    ax.legend()

    plt.show()

    # And here is the one that I used to verify the FFT. One day I should put it on my website 😅

    # # -*- coding: utf-8 -*-

    # """

    # Compute Fourier series using FFT.

    # @author: ricriv

    # """

    

    # # %% Import.

    

    # import numpy as np

    # from scipy.fft import fft, fftfreq, fftshift

    # import matplotlib.pyplot as plt

    

    # plt.close("all")

    

    

    # # %% Functions.

    

    # def fourier_series_trig(avg, a, b, p, time):

    #     """

    #     Evaluate a Fourier series in trigonometric form.

    

    #     Parameters

    #     ----------

    #     avg : float

    #         Average.

    #     a : (N, ) array_like

    #         Amplitude of cosine terms.

    #     b : (N, ) array_like

    #         Amplitude of sine terms.

    #     p : float

    #         Period

    #     time : (M, ) array_like

    #         Time array.

    

    #     Returns

    #     -------

    #     y : (M, ) array_like

    #         Fourier series evaluated over the time array.

    #     """

    #     assert a.size == b.size

    #     Omega = 2 * np.pi / period

    #     y = np.full_like(time, avg)

    #     for n in range(a.size):

    #         phi = Omega * (n + 1) * time

    #         y += a[n] * np.cos(phi) + b[n] * np.sin(phi)

    #     return y

    

    

    # def trig_to_exp_coeff(avg, a, b):

    #     """

    #     Convert Fourier series coefficients from trigonometric to exponential form. The order is the same used by the FFT.

    

    #     Parameters

    #     ----------

    #     avg : float

    #         Average.

    #     a : (N, ) array_like

    #         Amplitude of cosine terms.

    #     b : (N, ) array_like

    #         Amplitude of sine terms.

    

    #     Returns

    #     -------

    #     c : (2*N+1, ) array_like

    #         Amplitude of exponential terms.

    #     """

    #     assert a.size == b.size

    #     c = np.zeros((2*a.size+1), dtype=complex)

    #     c[0] = avg

    #     c[1:a.size+1] = 0.5 * (a - 1j * b)

    #     c[a.size+1:] = np.flip(np.conj(c[1:a.size+1]))

    #     return c

    

    

    # # %% Make test signal.

    

    # # Time properties.

    # period = 1.5  # [s]

    # n_periods = 1

    # time = np.linspace(0.0, n_periods*period, 1000, endpoint=False)

    # dt = time[1]

    # sampling_frequency = 1.0 / dt

    

    # # Set coefficients of the Fourier series.

    # avg = 0.75

    # a = np.array([3.0, 2.0, 1.0])

    # b = np.array([3.5, 2.5, 1.5])

    

    # # Evaluate Fourier series.

    # y = fourier_series_trig(avg=avg,

    #                         a=a,

    #                         b=b,

    #                         p=period,

    #                         time=time)

    

    # # Plot time history.

    # fig, ax = plt.subplots(dpi=300)

    # ax.set_xlabel("Time [s]")

    # ax.set_ylabel("Signal [-]")

    # ax.grid(True)

    # ax.plot(time, y)

    

    # # Convert coefficients to exponential form.

    # c = trig_to_exp_coeff(avg, a, b)

    

    # # Sort harmonics from -n to +n.

    # c_centered = fftshift(c)

    # freq_exact = np.arange(-a.size, +a.size+1)

    

    # # Plot exact Fourier coefficients.

    # fig, ax = plt.subplots(nrows=2, sharex=True, dpi=300)

    # ax[0].set_title("Exact Fourier coefficients")

    # ax[0].set_ylabel("Real")

    # ax[1].set_ylabel("Imag")

    # ax[1].set_xlabel("n")

    # ax[0].grid(True)

    # ax[1].grid(True)

    # ax[0].stem(freq_exact, c_centered.real)

    # ax[1].stem(freq_exact, c_centered.imag)

    # ax[0].set_xlim(-a.size-1, +a.size+1)

    

    # # Compute FFT.

    # y_fft = fft(y) / y.size

    # freq_fft = np.arange(y.size) / n_periods

    

    # # Sort harmonics from -n to +n.

    # y_fft_centered = fftshift(y_fft)

    # if y.size % 2 == 0:

    #     n = y.size

    #     freq_fft_centered = np.arange(-int(n/2), +int(n/2)) / n_periods

    # else:

    #     n = y.size - 1

    #     freq_fft_centered = np.arange(-int(n/2), +int(n/2)+1) / n_periods

    

    # # Plot Fourier coefficients from FFT.

    # fig, ax = plt.subplots(nrows=2, sharex=True, dpi=300)

    # ax[0].set_title("Fourier coefficients from FFT")

    # ax[0].set_ylabel("Real")

    # ax[1].set_ylabel("Imag")

    # ax[1].set_xlabel("n")

    # ax[0].grid(True)

    # ax[1].grid(True)

    # # ax[0].plot(freq_fft, y_fft.real)

    # # ax[1].plot(freq_fft, y_fft.imag)

    # ax[0].plot(freq_fft_centered, y_fft_centered.real)

    # ax[1].plot(freq_fft_centered, y_fft_centered.imag)

    # ax[0].set_xlim(-a.size-1, +a.size+1)

    

    # # Select only non-zero Fourier coefficients.

    # y_fft_centered_abs = np.abs(y_fft_centered)

    # i_non_zero = np.where(y_fft_centered_abs > 1e-14)[0]

    # harmonics_fft = y_fft_centered[i_non_zero]

    # harmonics_freq = freq_fft_centered[i_non_zero]

    

    # # Plot all Fourier coefficients.

    # fig, ax = plt.subplots(nrows=2, sharex=True, dpi=300)

    # ax[0].set_title("Fourier coefficients")

    # ax[0].set_ylabel("Real")

    # ax[1].set_ylabel("Imag")

    # ax[1].set_xlabel("n")

    # ax[0].grid(True)

    # ax[1].grid(True)

    # ax[0].stem(freq_exact, c_centered.real, label="Exact")

    # ax[1].stem(freq_exact, c_centered.imag, label="Exact")

    # ax[0].stem(harmonics_freq, harmonics_fft.real, linefmt="C1-", label="FFT")

    # ax[1].stem(harmonics_freq, harmonics_fft.imag, linefmt="C1-", label="FFT")

    # ax[0].legend()

    # ax[0].set_xlim(-a.size-1, +a.size+1)
