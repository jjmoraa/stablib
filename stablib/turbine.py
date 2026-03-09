# turbine.py

import numpy as np
from pathlib import Path

# locals
from collections import defaultdict
from stablib import state_space
from stablib import openfast
from stablib.tictoc import Timer
from stablib.models.model5DOFs import mass, damping, stiffness
from stablib.state_space import A_fromMCK, computeDamping, mac_sort_modes, reorder_parameters_by_assignment
from stablib.floquet import  solve, floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection, mode_projection_multiple_harmonics, mode_projection_multiple_harmonics_v2
from stablib.PostProcessing import plot_freq_heatmap, evaluateStabilityMonodromy, plotCampbellDiagram, plotCampbellDiagramAllModesSingleHarmonic, plotCampbellDiagramMultipleHarmonics


class Turbine:

    def __init__(self, model_folder):
        self.model_folder = Path(model_folder)

        # Load model. Can we add a parametric A directly instead?
        (
            self.arrays_by_op,
            self.A_interp,
            self.u_vel,
            self.omegas,
            self.T_rotor,
            self.q_of_interest,
            self.results
        ) = openfast.openFAST_A_interpreter(self.model_folder)

        # Storage results later
        self.results = {}

    def run_floquet_analysis(self, plotIVP, out_spec_matrix = np.eye(16), n_harmonics=3, rtol=1e-4):
        """
        Run full Campbell / Floquet analysis over all operating points.
        """
        omegas = self.omegas
        T_rotor = self.T_rotor
        arrays_by_op = self.arrays_by_op
        eigenvalues_for_range = []
        mode_shapes = []
        eigenvalues_exp_corrected_for_range = []
        participation_factor_for_range = []
        max_index_for_range = []

        for iom, omega in enumerate(omegas): #rads
            print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ------------------------')
            period = T_rotor[iom]
            num_points=10001 #give odd number to get even number in fft (very important)
            time=np.linspace(0.0, period, num_points)
            
            # Functions of time
            # Suppose you have 36 A matrices for an operating point
            A_matrices = arrays_by_op[iom]  # shape (36, nStates, nStates)

            # Create interpolator
            At = state_space.make_matrix_interpolator(A_matrices, period) # PUT IT OUTSIDE
            test_periodic(At, period)

            # Time domain solution of the x_dot=Ax system
            with Timer('solve-ivp'):
                sol=solve(At,time,plot=plotIVP)
            print('Solution is finished')

            with Timer('floquet_eig'):
                [monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] = floquet_eigenanalysis(sol,time,omega, plot=plotFloquet, sanityChecks=False, period = period)
                eigenvalues_for_range.append(eigenvalues_exp)
            print('Floquet eigenanalysis is finished')

            stabilityMon = evaluateStabilityMonodromy(eigenvalues_mon, doPlot=False)
            SS = eigenvectors_exp            

            n_harmonics=3
            with Timer('mode_proj'):
                [max_vals, max_index, participation_factor, basis, out_spec_basis, fourier_coefficients, participation_factor, freqs, ifreq0] = mode_projection_multiple_harmonics_v2(out_spec_matrix, q_values, eigenvectors_exp, time, n_harmonics, plot=False, sanityChecks=False)
            # correct index with strongest frequency
            eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)
            eigenvalues_exp_corrected_for_range.append(eigenvalues_exp_corrected)

            # save participation factors max indices and modes
            participation_factor_for_range.append(participation_factor)
            max_index_for_range.append(max_index)
            mode_shapes.append(eigenvectors_exp)
        

        # Save results to object state
        self.results["Stable"] = stabilityMon
        self.results["eigenvalues"] = eigenvalues_exp_corrected_for_range
        self.results["mode_shapes"] = mode_shapes
        self.results["participation_factor"] = participation_factor_for_range
        self.results["max_index"] = max_index_for_range

    def offloadFloquet(self):
        '''Onwards it's offload calculations. I want to pack these too'''
        eigenvalues_exp_corrected_for_range = np.array(self.results["eigenvalues"])
        participation_factor_for_range = np.array(self.results["participation_factor"])
        max_index_for_range = np.array(elf.results["max_index"])
        off_indices = max_index_for_range - max_index_for_range[:, 0, :][:, None, :]

        all_modes = off_indices.reshape(off_indices.shape[0], -1, order='F')
        # unique_by_mode = [np.unique(all_modes[i]) for i in range(all_modes.shape[0])]
        # Use np.unique with return_index
        unique_indices, idx = np.unique(all_modes, return_index=True)
        unique_indices = unique_indices[np.argsort(idx)]

        omegas = self.omegas
        eigenvalues_unique = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        f_d = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        f_0 = np.zeros_like(f_d)
        zeta = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        pf_index_unique = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]))

        ''' vectorial instead of '''
        for iom, omega in enumerate(omegas): #rads
            for iind in range(len(unique_indices)):
                eigenvalues_unique [iom, iind, :] = eigenvalues_exp_corrected_for_range[iom, 0, :] + 1j * np.tile(unique_indices[None, :] * omega, (At(0).shape[0], 1))[:,iind]
                pf_index_unique[iom, iind, :] = max_index_for_range[iom,0,:] + unique_indices[iind]+ifreq0
            f_d[iom], f_0[iom], zeta[iom] = computeDamping(eigenvalues_unique[iom,:,:])

        n_omega, n_harmonics, n_modes = participation_factor_for_range.shape
        # fancy indexing along axis=1 (harmonics)
        pf_of_interest = participation_factor_for_range[
            np.arange(n_omega)[:, None, None],       # omega axis
            pf_index_unique.astype(int),             # harmonics axis
            np.arange(n_modes)[None, None, :]        # modes axis
        ]

        self.results["f_d"] = f_d
        self.results["eigenvalues_unique"] = eigenvalues_unique
        self.results["f_0"] = f_0
        self.results["zeta"] = zeta
        self.results["pf_index_unique"] = pf_index_unique
        self.results["pf_of_interest"] = pf_of_interest

    def sort_results(self):
        mode_shapes = self.results["mode_shapes"]
        f_0 = self.results["f_0"]
        f_d = self.results["f_d"]
        zeta = self.results["zeta"]
        pf_of_interest = self.results["pf_of_interest"]
        unique_indices = self.results["unique_indices"]
        
        # sort the modes
        mode_shapes_sorted, assignment_array = mac_sort_modes(mode_shapes, use_macx=False, debug=False)

        # sort the first harmonic
        vf_0_sorted = np.zeros_like(f_0[:, :, :])
        vf_d_sorted = np.zeros_like(f_d[:, :, :])
        zeta_for_range_sorted = np.zeros_like(zeta[:, :, :])
        participation_factor_for_range_sorted = np.zeros_like(pf_of_interest[:, :, :])

        for iind in range(len(unique_indices)):
            vf_0_sorted[:, iind, :] = reorder_parameters_by_assignment(f_0[:,iind,:], assignment_array)
            vf_d_sorted[:, iind, :] = reorder_parameters_by_assignment(f_d[:,iind,:], assignment_array)
            zeta_for_range_sorted[:, iind, :] = reorder_parameters_by_assignment(zeta[:,iind,:], assignment_array)
            participation_factor_for_range_sorted[:, iind, :] = reorder_parameters_by_assignment(pf_of_interest[:,iind,:], assignment_array)

        self.q_of_interest["vf_0_sorted"] = vf_0_sorted
        self.q_of_interest["vf_d_sorted"] = vf_d_sorted
        self.q_of_interest["zeta_for_range_sorted"] = zeta_for_range_sorted
        self.q_of_interest["participation_factor_for_range_sorted"] = participation_factor_for_range_sorted

    def plot_campbell(self):
        """
        Generate standard Campbell plots.
        """
        vf_0_sorted = self.results["vf_0_sorted"]
        omegas = self.omegas

        ...
