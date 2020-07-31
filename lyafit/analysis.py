# import numpy as np


class Analysis:
    """Lyafit analysis class
    - Parameter minimization using iminuit
    - Create Monte Carlo realizations of the data
    - Run FastMC analysis
    - Compute parameter scan
    """

    def __init__(self, main_config, chi2_func):
        self.chi2 = chi2_func
        pass

    def run(self):
        print(self.chi2())
