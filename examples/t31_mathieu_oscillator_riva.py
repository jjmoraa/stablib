# -*- coding: utf-8 -*-
"""
Stability analysis of the Mathieu oscillator with Floquet theory.

Adapted from: 
   https://gitlab.windenergy.dtu.dk/wtstab/stability-analysis-of-wind-turbines

    @author: ricriv


"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.linalg import eig, inv


def mo_riva(plot=False):
    print(os.getcwd())
    if plot:
        os.makedirs("./_mathieu_oscillator/", exist_ok=True)


    # --- Define system.

    # Define parameters for the Mathieu oscillator from Allen's paper.
    m = 1.0
    k0 = 1.0
    k1 = 0.4
    damp = 0.04
    W = 0.8  # [rad/s]

    period = 2 * np.pi / W  # [s]
    w0 = k0 / m  # =omega_0^2
    w1 = k1 / m  # =omega_1^2
    cc = damp / m  # =2*zeta*omega_0


    # Define matrices for the Mathieu oscillator.
    def mathieu_a(t, w0, w1, cc, W):
        return np.array([[0.0, 1.0], [-w0 - w1 * np.cos(W * t), -cc]])


    def mathieu_b(m):
        return np.array([[0], [1 / m]])


    mathieu_c = np.array([[1, 0]])

    # Number of states.
    nx = 2


    # --- Simulate free response.

    # Set initial condition.
    x0 = np.array([1.0, 0.0])

    # Integrate autonomus system.
    time_free = np.linspace(0.0, 4000.0, 100001)
    dt_free = time_free[1]
    df_free = 1.0 / time_free[-1]
    sampling_frequency_free = 1.0 / dt_free
    nyquist_frequency_free = sampling_frequency_free / 2.0

    sol_free = solve_ivp(
        fun=lambda t, x: mathieu_a(t, w0, w1, cc, W) @ x,
        t_span=(time_free[0], time_free[-1]),
        y0=x0,
        t_eval=time_free,
        vectorized=True,
    )

    # Plot time series.
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Displacement [m]")
        ax.set_xlim(0.0, 100.0)
        ax.plot(sol_free.t, sol_free.y[0, :])
        fig.savefig( "./_mathieu_oscillator/free_response_time.svg", bbox_inches="tight")

    # --- Compute PSD.
    # window = np.ones(sol_free.t.size)
    window = signal.get_window("hann", int(sol_free.t.size // 4))
    nperseg = len(window)
    # noverlap = 0
    noverlap = None
    nfft = nperseg
    detrend = "constant"
    return_onesided = True
    scaling = "density"

    dt = time_free[1]
    sampling_frequency = 1.0 / dt

    y_freq, y_PSD = signal.welch(
        sol_free.y[0, :],
        sampling_frequency,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
    )

    # Plot PSD.
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Displacement [m²/Hz]")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.5)
        # ax.set_ylim(1e-4, 1e1)  # Ok for rectangular window.
        ax.set_ylim(1e-15, 1e-2)  # Ok for Hann window.
        ax.plot(y_freq, y_PSD)
        fig.savefig( "./_mathieu_oscillator/free_response_psd.svg", bbox_inches="tight")

    # --- Apply Floquet theory.

    # 1 period.
    time_stm = np.linspace(0.0, period, 2001)
    dt_stm = time_stm[1]
    df_stm = 1.0 / time_stm[-1]
    sampling_frequency_stm = 1.0 / dt_stm
    nyquist_frequency_stm = sampling_frequency_stm / 2.0

    # Set initial condition.
    tm0 = np.eye(nx)

    # Integrate Phi' = A @ Phi.
    sol_stm = solve_ivp(
        fun=lambda t, stm: (mathieu_a(t, w0, w1, cc, W) @ stm.reshape(nx, nx)).reshape(-1),
        t_span=(time_stm[0], time_stm[-1]),
        y0=tm0.reshape(-1),
        t_eval=time_stm,
        vectorized=True,
    )

    # Plot state transition matrix.
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State transition matrix")
        ll = ax.plot(sol_stm.t, sol_stm.y.T)
        ax.legend(
            ll,
            (r"$\Phi_{1,1}$", r"$\Phi_{1,2}$", r"$\Phi_{2,1}$", r"$\Phi_{2,2}$"),
            loc="center",
        )
        fig.savefig("./_mathieu_oscillator/state_transition_matrix.svg", bbox_inches="tight")

    # Reshape to 3D array.
    # The state transition matrix, stm, is ordered as:
    #  - axis 0, 1: state
    #  - axis 2: time
    stm = np.reshape(sol_stm.y, (nx, nx, sol_stm.t.size))

    # Get monodromy matrix.
    monodromy = stm[:, :, -1]

    # Compute state at every period using the monodromy matrix.
    time_sampled = np.arange(int(time_free[-1] // period)) * period
    x_sampled = np.zeros((nx, time_sampled.size))
    x_sampled[:, 0] = x0
    for i in range(1, time_sampled.size):
        x_sampled[:, i] = monodromy @ x_sampled[:, i - 1]

    # Plot state at every period.
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Displacement [m]")
        ax.set_xlim(0.0, 100.0)
        ax.plot(sol_free.t, sol_free.y[0, :])
        ax.scatter(time_sampled, x_sampled[0, :], color="C1")
        fig.savefig("./_mathieu_oscillator/free_response_time_with_monodromy.svg", bbox_inches="tight",
        )

    # Compute characteristic multipliers.
    theta, S = eig(monodromy)

    # Plot characteristic multipliers.
    if plot:
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
        fig.savefig( "./_mathieu_oscillator/characteristic_multiplier.svg", bbox_inches="tight")

    # Define shift for characteristic exponents.
    if time_stm.size % 2 == 0:
        max_shift = int(time_stm.size / 2)
    else:
        max_shift = int((time_stm.size - 1) / 2)
    shift = np.arange(-max_shift, +max_shift)
    shift0 = max_shift  # n = 0.

    # Compute characteristic exponents.
    # eta is ordered as:
    #  - axis 0: harmonics.
    #  - axis 1: modes.
    abs_theta = np.abs(theta)
    ang_theta = np.angle(theta)
    eta = (np.log(abs_theta) + 1j * ang_theta)[
        np.newaxis, :
    ] / period + 2j * np.pi / period * shift[:, np.newaxis]
   # Plot characteristic exponents.
    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Real [rad/s]")
        ax.set_ylabel("Imag [rad/s]")
        ax.set_xlim(-0.04, +0.02)
        ax.set_ylim(-3.2, +3.2)
        for i in range(eta.shape[1]):
            ax.scatter(eta[:, i].real, eta[:, i].imag, label=f"Mode {i+1}")
        ax.legend()
        fig.savefig( "./_mathieu_oscillator/characteristic_exponent.svg", bbox_inches="tight")

    # Compute frequency and damping.
    natural_frequency = np.abs(eta)  # [rad/s]
    damping_ratio = -eta.real / natural_frequency  # [-]
    natural_frequency /= 2 * np.pi  # [Hz]
    damped_frequency = np.abs(eta.imag) / (2 * np.pi)  # [Hz]

    # Set initial value of periodic transformation.
    P0 = np.eye(nx)

    # Compute eigenvectors of Floquet factor.
    V = S  # Trivial since P0 = I.

    # Compute Floquet factor.
    # Should use solve() for large number of states.
    # R = V @ np.diag(eta[shift_principal, :]) @ inv(V)
    R = V * eta[shift0, :] @ inv(V)

    # Compute periodic transformation.
    # Time must be the first dimension to allow computing Xi using matmul.
    # P is periodic, therefore we skip the last time instant.
    P = np.zeros((time_stm.size - 1, nx, nx), dtype=complex)
    invS = inv(S)
    for k in range(P.shape[0]):
        # P[k, :, :] = stm[:, :, k] @ P0 @ expm(- R * time_stm[k])
        P[k, :, :] = stm[:, :, k] @ S * np.exp(-eta[shift0, :] * time_stm[k]) @ invS @ P0

    # Compute Xi.
    Xi = mathieu_c @ P @ V[np.newaxis, :, :]

    # Expand Xi in Fourier series. psi contains the mode shapes, and is ordered as:
    # axis 0: harmonics.
    # axis 1: states or outputs if C is used.
    # axis 2: modes.
    psi = np.fft.fft(Xi, axis=0) / Xi.shape[0]

    # Sort harmonics from -n to +n.
    psi_centered = np.fft.fftshift(psi, axes=0)

    # Compute mode shapes norm.
    participation = np.linalg.norm(psi_centered, ord=2, axis=1)

    # Normalize across the harmonics to get the output-specific participation factors.
    participation /= participation.sum(axis=0)

    # Find the principal harmonic for each mode.
    n_principal = np.argmax(participation, axis=0)

    # Collect stability results.
    # stability_red is ordered as:
    #  axis 0: shift, with 0 in the center.
    #  axis 1: shift, natural frequency, damping ratio and output-specific participation factor.
    #  axis 2: mode.
    n_print = 4
    stability_red = np.zeros((2 * n_print + 1, 4, nx))
    for i in range(nx):
        shift_red = range(n_principal[i] - n_print, n_principal[i] + n_print + 1)
        # stability_red[:, 0, i] = shift[shift_red]
        stability_red[:, 0, i] = np.arange(-n_print, +n_print + 1)
        stability_red[:, 1, i] = natural_frequency[shift_red, i]
        # stability_red[:, 1, i] = damped_frequency[shift_red, i]
        stability_red[:, 2, i] = damping_ratio[shift_red, i]
        stability_red[:, 3, i] = participation[shift_red, i]

    # Print mode 1.
    i_mode = 0
    # print(stability_red[:, :, i_mode])

    # Plot again the PSD, this time with the harmonics labelled.
    if plot :
        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Displacement [m²/Hz]")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1e-15, 1e0)  # Ok for Hann window.
        ax.plot(y_freq, y_PSD)
        y_PSD_at_natural_freq = np.interp(stability_red[:, 1, i_mode], y_freq, y_PSD)
        for i_harmonic in range(stability_red.shape[0]):
            ax.annotate(
                f"{int(stability_red[i_harmonic, 0, i_mode])}"+ r"$\Omega$",
                (stability_red[i_harmonic, 1, i_mode], y_PSD_at_natural_freq[i_harmonic]),
                xytext=(
                    stability_red[i_harmonic, 1, i_mode],
                    y_PSD_at_natural_freq[i_harmonic] * 10.0,
                ),
                arrowprops={"arrowstyle": "->", "color": "k"},
                horizontalalignment="center",
                verticalalignment="bottom",
                color="k",
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "k"},
            )
        fig.savefig( "./_mathieu_oscillator/free_response_psd_with_labels.svg", bbox_inches="tight")


    # --- Return outputs in dict
    d = dict()
    d['R'] = R
    d['P'] = P
    d['V'] = V
    d['S'] = S
    d['theta'] = theta
    d['eta'] = eta
    d['monodromy'] = monodromy
    d['sol_stm'] = sol_stm
    d['sol_free'] =sol_free
    d['Xi'] = Xi
    d['n_principal'] = n_principal
    
    return d


if __name__ == "__main__":
    mo_riva()
    
