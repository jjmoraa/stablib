from collections import defaultdict
from stablib import state_space
from pathlib import Path
from stablib import floquetParam
from stablib.floquet import test_periodic
import numpy as np
import openfast_toolbox


from openfast_toolbox.io.fast_linearization_file import FASTLinearizationFile

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
    files = list(folder.glob("*.lin"))   # list all documents in the folder
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
    interp_arrays_by_op = {}
    metadata_by_op = {}
    u_vel = np.zeros(len(files_by_op))
    omega_rad = np.zeros(len(files_by_op))
    T_rotor = np.zeros(len(files_by_op))

    for op in files_by_op:
        #arrays_by_op[op] = []   # ← initialize list for this OP
        
        for ifile, f in enumerate(files_by_op[op]):
            dfs, lin = readLinFiles(f, print=False)

            if ifile ==0:
                n_az = len((files_by_op[op])) # number of azimuthal positions by op
                n_dof = np.asarray(dfs['A']).shape[0] # get problem size for preallocation
                arrays_by_op[op] = np.zeros((n_az, n_dof, n_dof))
            

            A = np.asarray(dfs['A'])   # safer than np.array
            arrays_by_op[op][ifile] = A
            metadata_by_op[op] = lin['header']  # store metadata (same for all files at this OP)

            u_vel[op] = float(metadata_by_op[op][10].split(':')[1].split()[0]) # This takes the string from header and splits into b/a : and then b/a space
            omega_rad[op] = float(metadata_by_op[op][8].split(':')[1].split()[0])
            T_rotor[op] = 2 * np.pi / omega_rad[op]

        # convert list → true NumPy array (Nt, n, n)
        # arrays_by_op[op] = np.stack(arrays_by_op[op], axis=0)

        # # Check periodicity: coarse tolerance (2nd decimal)
        # if np.allclose(arrays_by_op[op][0], arrays_by_op[op][-1], atol=1e-2):
        #     print(f"[ OK ] Coarse periodicity check passed for OP {op}.")
        # else:
        #     print(f"[WARN] Coarse periodicity mismatch for OP {op}.")

        # # Check fine tolerance (3rd decimal)
        # if not np.allclose(arrays_by_op[op][0], arrays_by_op[op][-1], atol=1e-3):
        #     print(f"[INFO] Small mismatch detected for OP {op} at fine tolerance. Enforcing periodicity by replacing last matrix.")
        #     arrays_by_op[op][-1] = arrays_by_op[op][0]  # enforce periodicity

        # it looks like openfast does not repeat the period at the end. To build the interpolator, that has to be done
        # Repeat the first matrix at the end to ensure periodicity <consult to Branlard>
        # arrays_by_op[op] = np.concatenate([arrays_by_op[op], [arrays_by_op[op][0]]], axis=0)

        # Now create the interpolator with the fixed array
        interp_func = state_space.make_matrix_interpolator(arrays_by_op[op], period=T_rotor[op])

        # Store interpolator
        interp_arrays_by_op[op] = interp_func

        # Validation stage: Check if the interpolation matches the original data
        nMatrices = arrays_by_op[op].shape[0]
        tol = 1e-3  # Tolerance for matrix comparison

        test_periodic(interp_arrays_by_op[op], T_rotor[op])


    u_vel = np.array(u_vel)
    omega_rad = np.array(omega_rad)
    T_rotor = np.array(T_rotor)

    #arrays by op is a 3d array of shape [op, linfiles, n, n]
    print("Done loading dataframes.")

    return arrays_by_op, interp_arrays_by_op, u_vel, omega_rad, T_rotor

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

    return dfs, lin

class turbine(floquetParam.floquetParametricRange):

    def __init__(self, foldername):
        # call openfast interpreter to get arguments from folder
        arrays_by_op, A_interp, u_vel, omegas, T_rotor = openFAST_A_interpreter(foldername)

        # call the parent constructor
        super().__init__(omegas, A_interp, param=u_vel, param_label='u velocity')
        