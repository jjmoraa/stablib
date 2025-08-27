import os
import unittest
import numpy as np
from floquet import *
import matplotlib.pyplot as plt

class TestFloquet(unittest.TestCase):
    def test_vector_vs_fundamental(self, plot=False):
        pass
        #raise Exception()
        # Describe 
            # Compute state at every period using the monodromy matrix.
    # time_sampled = np.arange(int(time_free[-1] // period)) * period
    # x_sampled = np.zeros((nx, time_sampled.size))
    # x_sampled[:, 0] = x0
    # for i in range(1, time_sampled.size):
    #     x_sampled[:, i] = monodromy @ x_sampled[:, i - 1]
        # np.testing.assert_allclose(x x)

        if plot:
            plt.plot()
            plt.show()
        

if __name__ == '__main__':
    unittest.main()