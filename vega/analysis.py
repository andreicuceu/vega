import sys

import numpy as np

from vega.minimizer import Minimizer


class Analysis:
    """Vega analysis class.

    - Compute parameter scan

    - Create Monte Carlo realizations of the data

    - Run FastMC analysis
    """
    current_mc_mock = None

    def __init__(
        self, chi2_func, sampler_params, main_config, corr_items, data,
        mc_config=None, global_cov=None
    ):
        """

        Parameters
        ----------
        chi2_func : function
            Chi2 function to minimize
        sampler_params : dict
            Dictionary with the sampling parameters
        main_config : ConfigParser
            Main config file
        corr_items : dict
            Dictionary with the correlation items
        data : dict
            Dictionary with the data
        mc_config : dict, optional
            Monte Carlo config with the model and sample parameters,
            by default None
        """
        self.config = main_config
        self._chi2_func = chi2_func
        self._scan_minimizer = Minimizer(chi2_func, sampler_params)
        self._corr_items = corr_items
        self._data = data
        self.mc_config = mc_config
        self.has_monte_carlo = False
        self._global_cov = global_cov
        self._cholesky_global_cov = None

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
                self._scan_minimizer.minimize(sample_params)
                result = self._scan_minimizer.values
                result['fval'] = self._scan_minimizer.fmin.fval
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
                    self._scan_minimizer.minimize(sample_params)
                    result = self._scan_minimizer.values
                    result['fval'] = self._scan_minimizer.fmin.fval
                    self.scan_results.append(result)

                    print('INFO: finished chi2scan iteration {} of {}'.format(
                        i * len(self.grids[par2]) + j + 1,
                        len(self.grids[par1]) * len(self.grids[par2])))

        return self.scan_results

    def create_monte_carlo_sim(self, fiducial_model, seed=None, scale=None, forecast=False):
        """Compute Monte Carlo simulations for each Correlation item.

        Parameters
        ----------
        fiducial_model : dict
            Fiducial model for the correlation functions
        seed : int, optional
            Seed for the random number generator, by default None
        scale : float/dict, optional
            Scaling for the covariance, by default None
        forecast : boolean, optional
            Forecast option. If true, we don't add noise to the mock,
            by default False

        Returns
        -------
        dict
            Dictionary with MC mocks for each item
        """
        mocks = {}
        for name in self._corr_items:
            # Get scale
            if scale is None:
                item_scale = self._corr_items[name].cov_rescale
            elif type(scale) is float or type(scale) is int:
                item_scale = scale
            elif type(scale) is dict and name in scale:
                item_scale = scale[name]
            else:
                item_scale = 1.

            # Create the mock
            mocks[name] = self._data[name].create_monte_carlo(
                fiducial_model[name], item_scale, seed, forecast)

        return mocks

    def create_global_monte_carlo(self, fiducial_model, scale=None, forecast=False):
        """Create Monte Carlo simulation using global covariance matrix

        Parameters
        ----------
        fiducial_model : dict
            Input fiducial model to use as mean of the multvariate Gaussian
        scale : float, optional
            Optional rescaling of the covariance matrix, by default None
        forecast : bool, optional
            Forecast mode flag, by default False

        Returns
        -------
        array
            Global masked MC data vector
        """
        assert self._global_cov is not None

        # TODO The corr items need to have an imposed order
        full_data_mask = []
        for name in self._corr_items:
            full_data_mask.append(self._data[name].data_mask)
        full_data_mask = np.concatenate(full_data_mask)

        if self._cholesky_global_cov is None:
            masked_cov = self._global_cov[:, full_data_mask]
            masked_cov = masked_cov[full_data_mask, :]
            if scale is None:
                scale = 1
            self._cholesky_global_cov = np.linalg.cholesky(scale * masked_cov)

        # TODO The corr items need to have an imposed order
        masked_fiducial = []
        for name, data in self._data.items():
            mask = data.dist_model_coordinates.get_mask_to_other(data.data_coordinates)
            if data.data_mask.size == fiducial_model[name].size:
                masked_fiducial.append(fiducial_model[name][data.data_mask])
            elif mask.size == fiducial_model[name].size:
                masked_fiducial.append(fiducial_model[name])
            else:
                raise ValueError('Input fiducial has unknown size. '
                                 'It must match the data or the model.')
        masked_fiducial = np.concatenate(masked_fiducial)

        if forecast:
            mc_mock = masked_fiducial
        else:
            ran_vec = np.random.randn(full_data_mask.sum())
            mc_mock = masked_fiducial[full_data_mask] + self._cholesky_global_cov.dot(ran_vec)

        return mc_mock

    def run_monte_carlo(
            self, fiducial_model, num_mocks=1, seed=0, scale=None, forecast=False, run_mc_fits=True
    ):
        """Run Monte Carlo simulations

        Parameters
        ----------
        fiducial_model : dict
            Fiducial model for the correlation functions
        num_mocks : int, optional
            Number of mocks to run, by default 1
        seed : int, optional
            Starting seed, by default 0
        scale : float/dict, optional
            Scaling for the covariance, by default None
        """
        assert self.mc_config is not None, 'No Monte Carlo config provided'

        np.random.seed(seed)
        sample_params = self.mc_config['sample']
        minimizer = Minimizer(self._chi2_func, sample_params)

        self.mc_bestfits = {}
        self.mc_covariances = []
        self.mc_chisq = []
        self.mc_valid_minima = []
        self.mc_valid_hesse = []
        self.mc_mocks = {}
        self.mc_failed_mask = []

        for i in range(num_mocks):
            print(f'INFO: Running Monte Carlo realization {i}')
            sys.stdout.flush()

            # Create the mocks
            if self._global_cov is None:
                mocks = self.create_monte_carlo_sim(
                    fiducial_model, seed=None, scale=scale, forecast=forecast)

                for name, cf_mock in mocks.items():
                    if name not in self.mc_mocks:
                        self.mc_mocks[name] = []
                    self.mc_mocks[name].append(cf_mock)
            else:
                self.current_mc_mock = self.create_global_monte_carlo(
                    fiducial_model, scale=scale, forecast=forecast)

                if 'global' not in self.mc_mocks:
                    self.mc_mocks['global'] = []
                self.mc_mocks['global'].append(self.current_mc_mock)

            if not run_mc_fits:
                continue

            try:
                # Run minimizer
                minimizer.minimize()
                self.mc_failed_mask.append(False)
            except ValueError:
                print('WARNING: Minimizer failed for mock {}'.format(i))
                self.mc_failed_mask.append(True)
                self.mc_chisq.append(np.nan)
                self.mc_valid_minima.append(False)
                self.mc_valid_hesse.append(False)
                continue

            for param, value in minimizer.values.items():
                if param not in self.mc_bestfits:
                    self.mc_bestfits[param] = []
                self.mc_bestfits[param].append([value, minimizer.errors[param]])

            self.mc_covariances.append(minimizer.covariance)
            self.mc_chisq.append(minimizer.fmin.fval)
            self.mc_valid_minima.append(minimizer.fmin.is_valid)
            self.mc_valid_hesse.append(not minimizer.fmin.hesse_failed)

        if run_mc_fits:
            for param in self.mc_bestfits.keys():
                self.mc_bestfits[param] = np.array(self.mc_bestfits[param])

        self.has_monte_carlo = True
