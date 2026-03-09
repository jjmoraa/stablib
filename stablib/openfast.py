from collections import defaultdict
from stablib import state_space
from pathlib import Path
import numpy as np

def get_operating_point(path):
    """
    Extract operating point from filename.
    Example: '00_NREL_5MW.1.lin' → 0
    """
    prefix = path.stem.split("_")[0]   # '00'
    return int(prefix)


def openFAST_A_interpreter(folder):
    """
    Docstring for openFAST_A_interpreter
    
    :param folder: Description
    """
    files = list(folder.glob("*.lin"))   # change .txt to your extension
    files_by_op = defaultdict(list)   # Define a dictionary to hold the filenames for each operating point
    dfs_by_op = defaultdict(list)   # Define a dictionary to hold the dataframes for each operating point

    # Here we group files by operating point using a dictionary
    for f in files:
        op = get_operating_point(f)
        files_by_op[op].append(f)
        #lin = state_space.readLinFiles(folder_path, print=True)

    # Now we make sure the files are sorted (maybe unnecessary because acdc will do it for you)
    for op in files_by_op:
        files_by_op[op].sort(
            key=lambda p: int(p.stem.split(".")[-1])
        )

    # Now we make the dataframes and store them in the other dictionary

    arrays_by_op = {}
    metadata_by_op = {}
    u_vel = {}
    omega_rad = {}
    T_rotor = {}

    for op in files_by_op:
        arrays_by_op[op] = []   # ← initialize list for this OP

        for f in files_by_op[op]:
            lin = state_space.readLinFiles(f, print=False)

            A = np.asarray(lin['A'])   # safer than np.array
            arrays_by_op[op].append(A)
            metadata_by_op[op] = lin['y']   # store metadata (same for all files at this OP)
            u_vel[op] = lin['y'].iloc[0]['Wind1VelX_[m/s]']
            omega_rad[op] = lin['y'].iloc[0]['RotSpeed_[rpm]'] * 2 * np.pi / 60
            T_rotor[op] = 2 * np.pi / omega_rad[op]

        # convert list → true NumPy array (Nt, n, n)
        arrays_by_op[op] = np.stack(arrays_by_op[op], axis=0)

    u_vel = np.array(list(u_vel.values()))
    omega_rad = np.array(list(omega_rad.values()))
    T_rotor = np.array(list(T_rotor.values()))

    #arrays by op is a 3d array of shape [op, linfiles, n, n]
    print("Done loading dataframes.")

    # Suppose you have 36 A matrices for an operating point
    A_matrices = arrays_by_op[1]  # shape (36, nStates, nStates)

    # Create interpolator
    A_interp = state_space.make_matrix_interpolator(A_matrices)

    return arrays_by_op, A_interp, u_vel, omega_rad, T_rotor


import openfast_toolbox
import os
from openfast_toolbox.io.fast_linearization_file import FASTLinearizationFile

def readLinFiles(filename, print = False):
    # --- Open and convert files to DataFrames
    lin = FASTLinearizationFile(filename)

    if print:
        print(lin)
        print(lin['A'])
        print(lin['A'].shape)
        print('Using to dataframe:')

    dfs = lin.toDataFrame()

    if print:
        print(dfs.keys())
        print('A:')
        print(dfs['A'])

    return dfs

from scipy.interpolate import interp1d
import numpy as np

def make_matrix_interpolator(matrices, period=None, positions=None, kind='cubic'):
    """
    Create a lambda function to interpolate between a series of matrices.
    
    If only one matrix is given, returns a constant function.

    Parameters
    ----------
    matrices : np.ndarray
        Array of shape (nMatrices, nStates, nStates) containing the matrices to interpolate.
    positions : np.ndarray, optional
        Positions corresponding to each matrix. If None, uses np.linspace(0, 1, nMatrices).
    kind : str
        Type of interpolation ('linear', 'cubic', etc.)
    period : float, optional
        Period in seconds. If given, the returned lambda accepts time in seconds.

    Returns
    -------
    interp_lambda : function
        Lambda function interp_lambda(pos_or_time) that returns an interpolated matrix.
        If period is given, input is in seconds; otherwise input is normalized [0,1].
    """
    nMatrices, nStates, _ = matrices.shape

    if nMatrices == 1:
        return lambda pos: matrices[0]

    if positions is None:
        positions = np.linspace(0, 1, nMatrices)
    
    matrices_flat = matrices.reshape(nMatrices, -1)
    interp_func = interp1d(positions, matrices_flat, axis=0, kind=kind, fill_value="extrapolate")
    
    if period is None:
        # Input is normalized position
        return lambda pos: interp_func(pos).reshape(nStates, nStates)
    else:
        # Input is physical time; scale and wrap
        return lambda t: interp_func((t / period) % 1.0).reshape(nStates, nStates)
