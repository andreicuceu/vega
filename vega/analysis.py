import numpy as np


class Analysis:
    """Vega analysis class.

    - Compute parameter scan

    - Create Monte Carlo realizations of the data

    - Run FastMC analysis
    """

    def __init__(self, minimizer, main_config, mc_config=None):
        """

        Parameters
        ----------
        minimizer : Minimizer
            Minimizer object initialized from the same vega instance
        main_config : ConfigParser
            Main config file
        mc_config : dict, optional
            Monte Carlo config with the model and sample parameters,
            by default None
        """
        self.config = main_config
        self.minimizer = minimizer
        self.mc_config = mc_config
        pass

    def chi2_scan(self):
        """Compute a chi^2 scan over one or two parameters.

        Returns
        -------
        List
            Scan results
        """
        # Check if we have the scan section in config
        if 'chi2 scan' not in self.config:
            raise ValueError('Called chi2_scan, but no config specified in'
                             ' main.ini. Add a "[chi2 scan]" section to main.')

        # Read the config and initialize the grids
        self.grids = {}
        for param, value in self.config.items('chi2 scan'):
            par_config = value.split()
            start = float(par_config[0])
            end = float(par_config[1])
            num_points = int(par_config[2])
            self.grids[param] = np.linspace(start, end, num_points)

        # We only support one or two dimensions
        dim = len(self.grids.keys())
        if dim > 2:
            raise ValueError('chi2_scan only supports one/two parameter scans')

        # Initialize the sample params and fix the right values
        sample_params = {}
        sample_params['fix'] = {}
        sample_params['values'] = {}
        sample_params['errors'] = {}
        for param in self.grids.keys():
            sample_params['fix'][param] = True
            sample_params['errors'][param] = 0.

        # Compute the scan
        self.scan_results = []
        # TODO Could add try/except to catch unwanted errors
        par1 = list(self.grids.keys())[0]
        if dim == 1:
            for i, value in enumerate(self.grids[par1]):
                # Overwrite params with the grid value
                sample_params['values'][par1] = value

                # Minimize and get bestfit values
                self.minimizer.minimize(sample_params)
                result = self.minimizer.values
                result['fval'] = self.minimizer.fmin.fval
                self.scan_results.append(result)

                print('INFO: finished chi2scan iteration {} of {}'.format(
                    i + 1, len(self.grids[par1])))
        else:
            par2 = list(self.grids.keys())[1]
            for i, value_1 in enumerate(self.grids[par1]):
                for j, value_2 in enumerate(self.grids[par2]):
                    # Overwrite params with the grid values
                    sample_params['values'][par1] = value_1
                    sample_params['values'][par2] = value_2

                    # Minimize and get bestfit values
                    self.minimizer.minimize(sample_params)
                    result = self.minimizer.values
                    result['fval'] = self.minimizer.fmin.fval
                    self.scan_results.append(result)

                    print('INFO: finished chi2scan iteration {} of {}'.format(
                        i * len(self.grids[par2]) + j + 1,
                        len(self.grids[par1]) * len(self.grids[par2])))

        return self.scan_results
