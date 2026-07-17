import numpy as np
import matplotlib.pyplot as plt
# Local
from welib.system.wtmodels.model5CS import *
from welib.tools.eva import eigMCK
from welib.system.mbc import Coleman2Comp

def Campbell5CS(vOmega=None, imbalance=0.0, f_imbalance=0.0):

    if vOmega is None:
        vOmega = np.linspace(0,1,11)

    # --- System parameters
    om0 = 0.7*2*np.pi
    kx  = 200000
    ky  = 350000
    l   = 1000
    m   = 1000
    M   = 200000
    kb = m*l**2*om0**2

    plane='XYneg'
    ordering='decreasing'
    method='analytical'

    # ---- Coleman
    f_NR = np.empty((len(vOmega), 5))

    for i, Omega in enumerate(vOmega):

        MNR, CNR, KNR = systemMatricesNR(
            M, m, l, kb, kx, ky, Omega,
            plane=plane,
            ordering=ordering,
            method=method
        )

        # -----------------------------
        # APPLY IMBALANCE HERE (clean hook)
        # -----------------------------
        MNR, CNR, KNR = systemMatricesNR(
            M, m, l, kb, kx, ky, Omega,
            plane=plane,
            ordering=ordering,
            method=method
        )

        # re-evaluate with imbalance directly through base matrices
        # (since analytical branch doesn't accept imbalance yet)
        from welib.system.wtmodels.model5CS_imb import systemMatrices

        psi1 = 0  # reference
        MM, DD, KK = systemMatrices(
            M, m, l, kb, kx, ky,
            Omega,
            psi1,
            plane=plane,
            ordering=ordering,
            imbalance=imbalance,
            f_imbalance=f_imbalance
        )

        # convert to non-rotating frame
        from welib.system.mbc import MBC3_Bmat, MBC3_MCK
        B, Binv, Bdot, Bddot, _, _ = MBC3_Bmat(1, 2, psi1=psi1, Omega=Omega, ordering=ordering)
        MNR, CNR, KNR = MBC3_MCK(MM, DD, KK, B, Binv, Bdot, Bddot)

        # eigenanalysis
        fd, zeta, Q, f0 = eigMCK(MNR, CNR, KNR, method='state_space')
        f_NR[i, :] = f0

    # --- Plot Campbell
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    vOmega_Hz = vOmega / (2 * np.pi)
    COLRS = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Modes (unchanged plotting logic) ---
    '''
    ax.plot(vOmega_Hz, f_NR[:,0], '-',  c=COLRS[0], linewidth=3,
            label=r'$\boldsymbol{\mathit{Tower\ side-side}}$')

    ax.plot(vOmega_Hz, f_NR[:,1], '-',  c=COLRS[1], linewidth=3,
            label=r'$\boldsymbol{\mathit{Tower\ fore-aft}}$')
    '''
    
    ax.plot(vOmega_Hz, f_NR[:,2], ':',  c=COLRS[2], linewidth=3,
            label=r'$\boldsymbol{\mathit{Backward\ whirling}}$')
    
    ax.plot(vOmega_Hz, f_NR[:,3], '-',  c=COLRS[2], linewidth=3,
            label=r'$\boldsymbol{\mathit{Symmetric}}$')

    ax.plot(vOmega_Hz, f_NR[:,4], '--', c=COLRS[2], linewidth=3,
            label=r'$\boldsymbol{\mathit{Forward\ whirling}}$')
    
    # --- 1P / 3P (styled like your preferred format) ---
    ax.plot(vOmega_Hz, vOmega_Hz, ':', c='0.2', linewidth=3,
            label=r'$\boldsymbol{\mathit{1P}}$')

    ax.plot(vOmega_Hz, 3 * vOmega_Hz, '--', c='0.2', linewidth=3,
            label=r'$\boldsymbol{\mathit{3P}}$')

    # --- Labels (math-style like your second block) ---
    ax.set_xlabel(r'$\boldsymbol{\mathit{Rotor\ speed\ f\ (Hz)}}$')
    ax.set_ylabel(r'$\boldsymbol{\mathit{Modal\ frequency\ f\ (Hz)}}$')

    ax.set_title(r'$\boldsymbol{\mathit{Campbell\ Diagram\ (MBC\ +\ imbalance)}}$')

    # --- Styling to match your preferred aesthetic ---
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize='small', frameon=False)
    ax.set_ylim([-0.05, 0.9])

    plt.tight_layout()

    return vOmega, f_NR, fig


if __name__ == '__main__':
    vOmega, f_NR, fig = Campbell5CS(
        imbalance=0.35,
        f_imbalance=0.05
    )
    plt.show()