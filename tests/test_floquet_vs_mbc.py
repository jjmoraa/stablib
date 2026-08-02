import unittest
import numpy as np

import stablib as stab
from stablib.dyn_sys_Param import dynSysParametricRange
from stablib.models import model5DOFs_mass_imbalance


class TestMBC(unittest.TestCase):

    def test_mbc_matches_floquet(self):

        # ------------------------------------------------------------------
        # Model parameters
        # ------------------------------------------------------------------

        m = 1000
        l = 100
        M = 200000

        imbalance = 0.5

        edgNatFreq_hz = 0.7
        edgNatFreq_rad = edgNatFreq_hz * 2 * np.pi

        kx = 200000
        ky = 350000

        omegas = np.linspace(0.1, 1.0, 10)

        A_vector = [None] * len(omegas)

        for iom, omega in enumerate(omegas):

            Mt = lambda t, omega=omega: model5DOFs_mass_imbalance.mass(
                M, m, imbalance, l, omega, t
            )

            Kt = lambda t, omega=omega: model5DOFs_mass_imbalance.stiffness(
                edgNatFreq_rad, m, imbalance, l, kx, ky, omega, t
            )

            Ct = lambda t, omega=omega: model5DOFs_mass_imbalance.damping(
                m, imbalance, l, omega, t
            )

            A_vector[iom] = lambda t, Mt=Mt, Ct=Ct, Kt=Kt: \
                stab.state_space.A_fromMCK(Mt(t), Ct(t), Kt(t))

        # ------------------------------------------------------------------
        # Floquet solution
        # ------------------------------------------------------------------

        floquet = dynSysParametricRange(omegas, A_vector)

        floquet.runFloquetAnalyses(
            harmonics=3,
            rtol=1e-4
        )

        floquet.sort_results()

        # ------------------------------------------------------------------
        # MBC solution
        # ------------------------------------------------------------------

        mbc = dynSysParametricRange(omegas, A_vector)

        mbc.runMBCAnalyses([0, 1, 2])

        mbc.sort_results()

        # ------------------------------------------------------------------
        # Compare
        # ------------------------------------------------------------------

        f0_floquet = floquet.q_of_interest["vf_0_sorted"]
        f0_mbc = mbc.q_of_interest["vf_0_sorted"]

        zeta_floquet = floquet.q_of_interest["zeta_for_range_sorted"]
        zeta_mbc = mbc.q_of_interest["zeta_for_range_sorted"]

        np.testing.assert_allclose(
            f0_mbc,
            f0_floquet,
            rtol=1e-4,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            zeta_mbc,
            zeta_floquet,
            rtol=1e-4,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()