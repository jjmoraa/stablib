# floquetParam.py

import numpy as np
from pathlib import Path

# locals
from collections import defaultdict
from stablib import state_space
from stablib.tictoc import Timer
from stablib.models.model5DOFs import mass, damping, stiffness
from stablib.state_space import A_fromMCK, computeDamping, mac_sort_modes, reorder_parameters_by_assignment
from stablib.floquet import  solve, floquet_eigenanalysis, test_periodic
from stablib.modeProjection import mode_projection, mode_projection_multiple_harmonics, mode_projection_multiple_harmonics_v2
from stablib.PostProcessing import plot_freq_heatmap, evaluateStabilityMonodromy, plotCampbellDiagram, plotCampbellDiagramAllModesSingleHarmonic, plotCampbellDiagramMultipleHarmonics


class floquetParametric:

    def __init__(self, omega, A, param = None, param_label=""):
        self.omega = omega
        self.param = param
        self.A = A
        self.dofs = np.shape(A(0))[0]
        self.results = {}

        if self.param is not None:
            assert(len(self.omega)) == len(self.param)

            def plot____(self):
                if self.param is not None:
                    x = self.param
                    label = self.param_label
                else:
                    x = self.omega
                    label = 'Angular speed'

    def run_floquet_analysis(self, plotIVP, out_spec_matrix = [], n_harmonics=3, rtol=1e-4):
        """
        Run full Campbell / Floquet analysis over all operating points.
        """
        omega = self.omega
        period = 2*np.pi / omega
        num_points=10001 #give odd number to get even number in fft (very important)
        time=np.linspace(0.0, period, num_points)
        At = self.A
        
        # to check periodicity of system
        test_periodic(At, period)

        # Time domain solution of the x_dot=Ax system
        with Timer('solve-ivp'):
            sol=solve(At,time,plot=plotIVP)
        print('Solution is finished')

        with Timer('floquet_eig'):
            [monodromy, exponent_matrix, eigenvalues_mon, eigenvectors_mon, eigenvalues_exp, eigenvectors_exp, q_values] \
            = floquet_eigenanalysis(sol,time,omega, plot=False, sanityChecks=False, period = period)
        print('Floquet eigenanalysis is finished')

        stabilityMon = evaluateStabilityMonodromy(eigenvalues_mon, doPlot=False)
        SS = eigenvectors_exp            
        self.results["Stable"] = stabilityMon
        self.results["eigenvalues_mon"] = eigenvalues_mon
        self.results["eigenvectors_mon"] = eigenvectors_mon
        self.results["eigenvalues_exp"] = eigenvalues_exp
        self.results["eigenvectors_exp"] = eigenvectors_exp
        self.results["q_values"] = q_values

    def run_modal_projection(self, out_spec_matrix = [], n_harmonics = 3):
        omega = self.omega
        period = 2*np.pi / omega
        num_points=10001 #give odd number to get even number in fft (very important)
        time=np.linspace(0.0, period, num_points)

        q_values = self.results["q_values"]
        eigenvectors_exp = self.results["eigenvectors_exp"]
        eigenvalues_exp = self.results["eigenvalues_exp"]

        if out_spec_matrix == []:
            out_spec_basis = np.eye(self.dofs)
        
        with Timer('mode_proj'):
            [max_vals, max_index, participation_factor, basis, out_spec_basis, \
             fourier_coefficients, participation_factor, freqs, ifreq0] \
            = mode_projection_multiple_harmonics_v2\
                (out_spec_matrix, q_values, eigenvectors_exp, time, n_harmonics, plot=False, sanityChecks=False)
        # correct index with strongest frequency
        eigenvalues_exp_corrected = eigenvalues_exp + 1j*(max_index)*(omega)

        # Save results to object state
        self.results["participation_factor"] = participation_factor
        self.results["max_index"] = max_index
        self.results["ifreq0"] = ifreq0
        self.results["mode_shapes"] = eigenvectors_exp
        self.results["eigenvalues_exp_corrected"] = eigenvalues_exp_corrected

class floquetParametricRange:
    def __init__(self, omegas, A_vector, param = None, param_label=""):
        self.omegas = omegas
        self.A_vector = A_vector
        self.param = param
        self.label = param_label
        self.operating_points = np.empty_like(omegas)
        self.eigenvalues_exp_corrected_for_range = np.empty_like(omegas)
        self.participation_factor_for_range = np.empty_like(omegas)
        self.max_index_for_range = np.empty_like(omegas)
        self.off_indices = np.empty_like(omegas)
        self.unique_indices = np.empty_like(omegas)
        self.ifreq0 = np.empty_like(omegas)
        self.results = []

        self.__createOPobjects()

    def __createOPobjects(self):
        
        omegas = self.omegas
        A_vector = self.A_vector
        param = self.param
        param_label = self.param_label
        for iom, omega in enumerate(omegas): #rads
            print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ------------------------')
            floquet_obj = floquetParametric(self, omega, A_vector[iom], param, param_label)
            self.operating_points[iom] = floquet_obj
    
    def runAnalyses(self, out_spec_matrix = None, harmonics=3, rtol=1e-4):
         
        for iom, omega in enumerate(self.omegas): #rads
            print(f'------------------{iom+1}/{len(omegas)}, omega = {omega} ----------------------')
            #  def run_floquet_analysis(self, plotIVP, out_spec_matrix = [], n_harmonics=3, rtol=1e-4):
            self.operating_points[iom].run_floquet_analysis(plotIVP=False, out_spec_matrix = out_spec_matrix, n_harmonics=n_harmonics, rtol=rtol)
        self.__offloadFloquet()
        self.__campbellData()
 


    def __offloadFloquet(self):
        '''Onwards it's offload calculations. I want to pack these too'''
        for iom, omega in enumerate(self.omegas): #rads
            self.eigenvalues_exp_corrected_for_range[iom] = np.array(self.operating_points[iom].results["eigenvalues"])
            self.max_index_for_range[iom] = np.array(self.operating_points[iom].results["max_index"])
            self.participation_factor_for_range[iom] = np.array(self.operating_points[iom].results["participation_factor"])
            max_index_for_range = np.array(self.operating_points[iom].results["max_index"])
            self.off_indices[iom] = max_index_for_range - max_index_for_range[:, 0, :][:, None, :]
            self.ifreq0[iom] = self.operating_points[iom].results["ifreq0"]
            all_modes = self.off_indices.reshape(self.off_indices.shape[0], -1, order='F')
            unique_indices, idx = np.unique(all_modes, return_index=True)
            self.unique_indices = unique_indices[np.argsort(idx)]

    def __campbellData(self):
        unique_indices = self.unique_indices
        eigenvalues_exp_corrected_for_range = self.eigenvalues_exp_corrected_for_range
        max_index_for_range = self.max_index_for_range
        omegas = self.omegas
        participation_factor_for_range = self.participation_factor_for_range
        eigenvalues_unique = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        f_d = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        f_0 = np.zeros_like(f_d)
        zeta = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]), dtype=complex)
        pf_index_unique = np.zeros((len(omegas), len(unique_indices), eigenvalues_exp_corrected_for_range.shape[2]))
        
        ''' vectorial instead of '''
        for iom, omega in enumerate(omegas): #rads
            for iind in range(len(unique_indices)):
                eigenvalues_unique [iom, iind, :] = eigenvalues_exp_corrected_for_range[iom, 0, :] + 1j * np.tile(unique_indices[None, :] * omega, (self.operating_points[iom].dofs, 1))[:,iind]
                pf_index_unique[iom, iind, :] = max_index_for_range[iom,0,:] + unique_indices[iind]+self.operating_points[iom].ifreq0
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
