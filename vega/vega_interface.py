"""Main module."""
import os.path
import numpy as np
import scipy.stats
from astropy.io import fits
import configparser
import copy

from . import correlation_item, data, utils
from vega.scale_parameters import ScaleParameters
from vega.model import Model
from vega.minimizer import Minimizer
from vega.analysis import Analysis
from vega.output import Output
from vega.parameters.param_utils import get_default_values
from vega.plots.plot import VegaPlots


class VegaInterface:
    """Main Vega class.

    Parse the main config and initialize a correlation item for each component.

    If there is data, initialize data and model objects for each component.

    Handle the parameter config and call the analysis class.
    """
    _blind = None
    _use_global_cov = False
    global_cov = None

    def __init__(self, main_path):
        """

        Parameters
        ----------
        main_path : string
            Path to main.ini config file
        """
        # Read the main config file
        self.main_config = configparser.ConfigParser()
        self.main_config.optionxform = lambda option: option
        self.main_config.read(utils.find_file(main_path))

        # Read the fiducial pk file
        self.fiducial = self._read_fiducial(self.main_config['fiducial'])

        # Read the effective redshift and the data config paths
        self.fiducial['z_eff'] = self.main_config['data sets'].getfloat('zeff')
        write_cf = self.main_config['output'].getboolean('write_cf', False)
        write_pk = self.main_config['output'].getboolean('write_pk', False)
        self.fiducial['save-components'] = write_cf or write_pk
        ini_files = self.main_config['data sets'].get('ini files').split()
        global_cov_file = self.main_config['data sets'].get('global-cov-file', None)

        self.model_pk = self.main_config['control'].getboolean('model_pk', False)
        self.low_mem_mode = self.main_config['control'].getboolean('low_mem_mode', False)
        self.low_mem_mode &= global_cov_file is not None

        # Initialize the individual components
        self.corr_items = {}
        for path in ini_files:
            config = configparser.ConfigParser()
            config.optionxform = lambda option: option
            config.read(utils.find_file(os.path.expandvars(path)))

            name = config['data'].get('name')
            self.corr_items[name] = correlation_item.CorrelationItem(config, self.model_pk)
            self.corr_items[name].low_mem_mode = self.low_mem_mode

        # Read parameters
        self.params = self._read_parameters(self.corr_items, self.main_config['parameters'])
        self.sample_params = self._read_sample(self.main_config['sample'])

        # Set growth rate
        use_template_growth_rate = self.main_config['control'].getboolean(
            'use_template_growth_rate', True)
        if use_template_growth_rate and 'growth_rate' in self.fiducial:
            assert 'growth_rate' not in self.sample_params['limits']
            self.params['growth_rate'] = self.fiducial['growth_rate']
        elif 'growth_rate' not in self.fiducial:
            print('WARNING: No growth rate specified in the template file. Using input value.')
            if 'growth_rate' in self.params:
                self.fiducial['growth_rate'] = self.params['growth_rate']

        if 'par_sigma_smooth' in self.params:
            self.fiducial['par_sigma_smooth'] = self.params['par_sigma_smooth']
        if 'per_sigma_smooth' in self.params:
            self.fiducial['per_sigma_smooth'] = self.params['per_sigma_smooth']

        # Check if all correlations have data files
        self.data = {}
        self._has_data = True
        for name, corr_item in self.corr_items.items():
            if not corr_item.has_data:
                self._has_data = False

        # Initialize the data
        for name, corr_item in self.corr_items.items():
            if self._has_data:
                self.data[name] = data.Data(corr_item)
            else:
                self.data[name] = None

        # Check blinding
        self._blind = False
        self._rnsps = None
        if self._has_data:
            self._init_blinding()

        # Initialize scale parameters
        self.scale_params = ScaleParameters(self.main_config['cosmo-fit type'])

        # initialize the models
        self.models = {}
        if self._has_data:
            for name, corr_item in self.corr_items.items():
                self.models[name] = Model(
                    corr_item, self.fiducial, self.scale_params, self.data[name])

        # Read the monte carlo parameters
        self.mc_config = None
        if 'monte carlo' in self.main_config:
            self.mc_config = {}
            config = self.main_config['monte carlo']

            self.mc_config['params'] = {}
            mc_params = self.main_config['mc parameters']
            for param, value in mc_params.items():
                self.mc_config['params'][param] = float(value)

            self.mc_config['sample'] = self._read_sample(config)

        # Get priors
        self.priors = {}
        if 'priors' in self.main_config:
            self.priors = self._init_priors(self.main_config['priors'])
            for param in self.priors.keys():
                param_is_not_sampled = param not in self.sample_params['limits']
                if self.mc_config is not None:
                    param_is_not_sampled &= param not in self.mc_config['sample']['limits']
                if param_is_not_sampled:
                    raise ValueError(
                        f'Prior specified for a parameter that is not sampled: {param}')

        # Read the global covariance
        cov_scale = self.main_config['control'].getfloat('cov_scale', None)
        if global_cov_file is not None:
            self.read_global_cov(global_cov_file, cov_scale)
            self._use_global_cov = True

        # Initialize the minimizer and the analysis objects
        if not self.sample_params['limits']:
            self.minimizer = None
        else:
            self.minimizer = Minimizer(self.chi2, self.sample_params)
        self.analysis = Analysis(
            self.chi2, self.sample_params, self.main_config,
            self.corr_items, self.data, self.mc_config, self.global_cov
        )

        # Check for sampler
        self.run_sampler = False
        if 'control' in self.main_config:
            self.run_sampler = self.main_config['control'].getboolean('run_sampler', False)
            self.sampler = self.main_config['control'].get('sampler', None)
            if self.run_sampler:
                if self.sampler not in ['Polychord', 'PocoMC']:
                    raise ValueError('Sampler not recognized. Please use Polychord or PocoMC.')
                if self.sampler not in self.main_config:
                    raise RuntimeError('run_sampler called, but no sampler config found')

        # Initialize the output object
        self.output = Output(self.main_config['output'], self.data, self.corr_items, self.analysis)

        # Initialize vega plots
        self.monte_carlo = False
        self.plots = None
        if self._has_data:
            self.plots = VegaPlots(vega_data=self.data)

    def compute_model(self, params=None, run_init=True, direct_pk=None):
        """Compute correlation function model using input parameters.

        Parameters
        ----------
        params : dict, optional
            Computation parameters, by default None
        run_init: boolean, optional
            Whether to run model.init() before computing the model, by default True
        direct_pk: 1D array or None, optional
            If not None, the full Pk (e.g. from CLASS/CAMB) to be used directly, by default None

        Returns
        -------
        dict
            Dictionary of cf models for each component
        """
        # Overwrite computation parameters
        local_params = copy.deepcopy(self.params)
        if params is not None:
            for par, val in params.items():
                local_params[par] = val

        assert self._blind is not None
        print(local_params)
        if self._rnsps is not None:
            assert self._blind
            local_params = utils.apply_blinding(local_params, self._rnsps)
        print(local_params)

        # Go through each component and compute the model cf
        model_cf = {}
        if run_init:
            self.models = {}
        for name, corr_item in self.corr_items.items():
            if run_init:
                self.models[name] = Model(corr_item, self.fiducial, self.scale_params,
                                          self.data[name])

            if direct_pk is None:
                model_cf[name] = self.models[name].compute(local_params, self.fiducial['pk_full'],
                                                           self.fiducial['pk_smooth'])
            else:
                model_cf[name] = self.models[name].compute_direct(local_params, direct_pk)

        return model_cf

    def chi2(self, params=None, direct_pk=None):
        """Compute full chi2 for all components.

        Parameters
        ----------
        params : dict, optional
            Computation parameters, by default None
        direct_pk: 1D array or None, optional
            If not None, the full Pk (e.g. from CLASS/CAMB) to be used directly, by default None

        Returns
        -------
        float
            chi^2
        """
        assert self._has_data

        # Overwrite computation parameters
        local_params = copy.deepcopy(self.params)
        if params is not None:
            for par, val in params.items():
                local_params[par] = val

        assert self._blind is not None
        if self._rnsps is not None:
            assert self._blind
            local_params = utils.apply_blinding(local_params, self._rnsps)

            # Enforce blinding
            for par, val in local_params.items():
                if par in utils.BLIND_FIXED_PARS:
                    local_params[par] = 1.

        # Compute chisq
        chi2 = 0
        full_model = []
        full_masked_data = []
        for name in self.corr_items:
            try:
                if direct_pk is None:
                    model_cf = self.models[name].compute(
                        local_params, self.fiducial['pk_full'], self.fiducial['pk_smooth'])
                else:
                    model_cf = self.models[name].compute_direct(local_params, direct_pk)
            except utils.VegaBoundsError:
                self.models[name].PktoXi.cache_pars = None
                return 1e100

            if self._use_global_cov:
                full_model.append(model_cf)

            if self.monte_carlo:
                if not self._use_global_cov:
                    diff = self.data[name].masked_mc_mock - model_cf[self.data[name].model_mask]
                    chi2 += diff.T.dot(self.data[name].scaled_inv_masked_cov.dot(diff))
            else:
                if self._use_global_cov:
                    full_masked_data.append(self.data[name].masked_data_vec)
                else:
                    diff = self.data[name].masked_data_vec - model_cf[self.data[name].model_mask]
                    chi2 += diff.T.dot(self.data[name].inv_masked_cov.dot(diff))

        if self._use_global_cov:
            full_model = np.concatenate(full_model)
            if self.monte_carlo:
                full_masked_data = self.analysis.current_mc_mock
            else:
                full_masked_data = np.concatenate(full_masked_data)
            diff = full_masked_data - full_model[self.full_model_mask]
            chi2 = diff.T.dot(self.masked_global_invcov.dot(diff))

        # Add priors
        for param, prior in self.priors.items():
            if param not in local_params:
                err_msg = ("You have specified a prior for a parameter not in "
                           f"the model. Offending parameter: {param}")
                assert param in local_params, err_msg
            chi2 += self._gaussian_chi2_prior(local_params[param], prior[0], prior[1])

        assert isinstance(chi2, float)
        return chi2

    def log_lik(self, params=None, direct_pk=None):
        """Compute full log likelihood for all components.

        Parameters
        ----------
        params : dict, optional
            Computation parameters, by default None
        direct_pk: 1D array or None, optional
            If not None, the full Pk (e.g. from CLASS/CAMB) to be used directly, by default None

        Returns
        -------
        float
            log Likelihood
        """
        assert self._has_data

        # Get the full chi2
        chi2 = self.chi2(params, direct_pk)

        # Compute the normalization for each component
        log_norm = 0
        for name in self.corr_items:
            log_norm -= 0.5 * self.data[name].data_size * np.log(2 * np.pi)

            if not self._use_global_cov:
                if self.monte_carlo:
                    log_norm -= 0.5 * self.data[name].scaled_log_cov_det
                else:
                    log_norm -= 0.5 * self.data[name].log_cov_det

        if self._use_global_cov:
            log_norm -= 0.5 * self.masked_global_log_cov_det

        # Compute log lik
        log_lik = log_norm - 0.5 * chi2

        # Add priors normalization
        for param, prior in self.priors.items():
            log_lik += self._gaussian_lik_prior(prior[1])

        return log_lik

    def monte_carlo_sim(self, params=None, scale=None, seed=int(0), forecast=False):
        """Compute Monte Carlo simulations for each Correlation item.

        Parameters
        ----------
        params : dict, optional
            Computation parameters, by default None
        scale : float/dict, optional
            Scaling for the covariance, by default 1.
        seed : int, optional
            Seed for the random number generator, by default 0
        forecast : boolean, optional
            Forecast option. If true, we don't add noise to the mock,
            by default False

        Returns
        -------
        dict
            Dictionary with MC mocks for each item
        """
        assert self._has_data

        # Overwrite computation parameters
        local_params = copy.deepcopy(self.params)
        if params is not None:
            for par, val in params.items():
                local_params[par] = val

        mocks = {}
        for name in self.corr_items:
            # Compute fiducial model
            fiducial_model = self.models[name].compute(
                local_params, self.fiducial['pk_full'],
                self.fiducial['pk_smooth'])

            # Get scale
            if scale is None:
                item_scale = self.corr_items[name].cov_rescale
            elif type(scale) is float or type(scale) is int:
                item_scale = scale
            elif name in scale:
                item_scale = scale[name]
            else:
                item_scale = 1.

            # Create the mock
            mocks[name] = self.data[name].create_monte_carlo(
                fiducial_model, item_scale, seed, forecast)

        self.monte_carlo = True
        return mocks

    def minimize(self):
        """Minimize the chi2 over the sampled parameters.
        """
        if self.minimizer is None:
            print("No sampled parameters. Skipping minimization.")
            return

        # if not self.fiducial['save-components']:
            # self.set_fast_metals()

        self.minimizer.minimize()

        self.bestfit_model = self.compute_model(self.minimizer.values, run_init=False)
        self.total_data_size = 0
        self.bestfit_corr_stats = {}

        num_pars = len(self.sample_params['limits'])
        print('\n----------------------------------------------------')
        for name in self.corr_items:
            data_size = self.data[name].data_size
            self.total_data_size += data_size

            if self.monte_carlo:
                diff = self.data[name].masked_mc_mock \
                    - self.bestfit_model[name][self.data[name].model_mask]
                chisq = diff.T.dot(self.data[name].scaled_inv_masked_cov.dot(diff))
            else:
                diff = self.data[name].masked_data_vec \
                    - self.bestfit_model[name][self.data[name].model_mask]
                chisq = diff.T.dot(self.data[name].inv_masked_cov.dot(diff))

            reduced_chisq = chisq / (data_size - num_pars)
            p_value = 1 - scipy.stats.chi2.cdf(chisq, data_size - num_pars)

            print(f'{name} chi^2/(ndata-nparam): {chisq:.1f}/({data_size}-{num_pars}) '
                  f'= {reduced_chisq:.3f}, PTE={p_value:.2f}')
            print('----------------------------------------------------')

            self.bestfit_corr_stats[name] = {'size': data_size, 'chisq': chisq,
                                             'reduced_chisq': reduced_chisq, 'p_value': p_value}

        self.chisq = self.minimizer.fmin.fval
        self.reduced_chisq = self.chisq / (self.total_data_size - num_pars)
        self.p_value = 1 - scipy.stats.chi2.cdf(self.chisq, self.total_data_size - num_pars)
        print(f'Total chi^2/(ndata-nparam): {self.chisq:.1f}/({self.total_data_size}-{num_pars}) '
              f'= {self.reduced_chisq:.3f}, PTE={self.p_value:.2f}')
        print('----------------------------------------------------\n')

        if not self.minimizer.fmin.is_valid:
            print('Invalid fit!!! Check data, covariance, model and priors.')

    @property
    def bestfit(self):
        """Access the bestfit results from iminuit.

        Returns
        -------
        Minimizer
            Returns the Minimizer class which stores the bestfit values
        """
        return self.minimizer

    def set_fast_metals(self):
        """Activate fast metals. This is automatically called when
        running the minimizer or the sampler.
        """
        print('Warning! Activating fast metals for minimizing/sampling.')
        for name in self.corr_items:
            if self.models[name].metals is not None:
                self.models[name].metals.fast_metals = True

    @staticmethod
    def _read_fiducial(fiducial_config):
        """Read the fiducial pk file and get the configs.

        Parameters
        ----------
        fiducial_config : ConfigParser
            fiducial section from the main config file

        Returns
        -------
        dict
            dictionary with the fiducial data and config
        """
        # First check the path and replace with the right model if necessary
        path = fiducial_config.get('filename')
        path = utils.find_file(os.path.expandvars(path))
        # if not os.path.isfile(path):
        # path = resource_filename('vega', 'models') + '/{}'.format(path)
        print('INFO: reading input Pk {}'.format(path))

        fiducial = {}

        # Open the fits file and get what we need
        hdul = fits.open(path)
        fiducial['z_fiducial'] = hdul[1].header['ZREF']
        fiducial['Omega_m'] = hdul[1].header['OM']
        fiducial['Omega_de'] = hdul[1].header['OL']
        fiducial['k'] = hdul[1].data['K']
        fiducial['pk_full'] = hdul[1].data['PK']
        fiducial['pk_smooth'] = hdul[1].data['PKSB']

        if 'F_ZREF' in hdul[1].header:
            fiducial['growth_rate'] = hdul[1].header['F_ZREF']

        hdul.close()

        return fiducial

    @staticmethod
    def _read_parameters(corr_items, parameters_config):
        """Read computation parameters.

        If a parameter is specified multiple times,
        the parameters in the main config file have priority.

        Parameters
        ----------
        corr_items : dict
            Dictionary of correlation items
        parameters_config : ConfigParser
            parameters section from main config

        Returns
        -------
        dict
            Computation parameters
        """
        params = {}

        # First get the parameters from each component config
        for name, corr_item in corr_items.items():
            if 'parameters' in corr_item.config:
                for param, value in corr_item.config.items('parameters'):
                    params[param] = float(value)

        # Next get the parameters in the main config
        for param, value in parameters_config.items():
            params[param] = float(value)

        return params

    def _read_sample(self, sample_config):
        """Read sample parameters.

        These must be of the form:

        param = min max / for sampler only
        or
        param = min max val err / for both sampler and fitter.

        Fitter accepts None for min/max, but the sampler does not.

        Parameters
        ----------
        sample_config : ConfigParser
            sample section from main config

        Returns
        -------
        dict
            Config for the sampled parameters
        """
        # Initialize the dictionaries we need
        sample_params = {}
        sample_params['limits'] = {}
        sample_params['values'] = {}
        sample_params['errors'] = {}
        sample_params['fix'] = {}

        default_values = get_default_values()

        def check_param(param):
            if param not in default_values:
                raise ValueError('Default values not found for: %s. Please add'
                                 ' them to default_values.txt, or provide the'
                                 ' full sampling specification.' % param)

        for param, values in sample_config.items():
            if param not in self.params:
                print('Warning: You tried sampling the parameter: %s.'
                      ' As this parameter was not specified under'
                      ' [parameters], it will be skipped.' % param)
                continue

            values_list = values.split()

            # Get the prior limits
            # ! Sampler needs actual values (no None)
            if len(values_list) > 1:
                lower_limit = None
                upper_limit = None
                if values_list[0] != 'None':
                    lower_limit = float(values_list[0])
                if values_list[1] != 'None':
                    upper_limit = float(values_list[1])
                sample_params['limits'][param] = (lower_limit, upper_limit)
            else:
                if values_list[0] not in ['True', 'true', 't', 'y', 'yes']:
                    continue
                check_param(param)
                sample_params['limits'][param] = default_values[param]['limits']

            # Get the values and errors for the fitter
            if len(values_list) > 2:
                sample_params['values'][param] = float(values_list[2])
            else:
                check_param(param)
                sample_params['values'][param] = self.params[param]

            if len(values_list) > 3:
                assert len(values_list) == 4
                sample_params['errors'][param] = float(values_list[3])
            else:
                check_param(param)
                sample_params['errors'][param] = default_values[param]['error']

            # Populate the fix values
            sample_params['fix'][param] = False

        return sample_params

    @staticmethod
    def _gaussian_chi2_prior(value, mean, sigma):
        return (value - mean)**2 / sigma**2

    @staticmethod
    def _gaussian_lik_prior(sigma):
        return -0.5 * np.log(2 * np.pi) - np.log(sigma)

    @staticmethod
    def _init_priors(prior_config):
        """Initialize the priors. Only gaussian priors are currently supported

        Parameters
        ----------
        prior_config : ConfigParser
            priors section from main config

        Returns
        -------
        dict
            Dictionary of priors (mean, sigma) with the keys as parameter names
        """
        prior_dict = {}
        for param, prior in prior_config.items():
            prior_list = prior.split()
            if len(prior_list) != 3:
                raise ValueError('Prior configuration must have the format:'
                                 ' "<param> = gaussian <mean> <sigma>"')
            if prior_list[0] not in ['gaussian', 'Gaussian']:
                raise ValueError('Only gaussian priors are supported.')

            prior_dict[param] = np.array(prior_list[1:]).astype(float)

        return prior_dict
    
    def _init_blinding(self):
        """Initialize blinding at the parameter level.
        """
        blinding_strat = None
        for data_obj in self.data.values():
            if data_obj.blind:
                self._blind = True

                if blinding_strat is None:
                    blinding_strat = data_obj.blinding_strat
                elif blinding_strat != data_obj.blinding_strat:
                    raise ValueError('Different blinding strategies found in the data sets.')

        if not self._blind:
            return

        blind_pars = []
        for par in self.sample_params['limits'].keys():
            if par in utils.BLIND_FIXED_PARS:
                raise ValueError(f'Running on blind data, parameter {par} must be fixed.')

            if par not in utils.VEGA_BLINDED_PARS:
                continue

            tracers = utils.VEGA_BLINDED_PARS[par]
            if any([corr.check_if_blind_corr(tracers) for corr in self.corr_items.values()]):
                blind_pars += [par]

        if len(blind_pars) > 0:
            self._rnsps = utils.get_blinding(blind_pars, blinding_strat)

        if ('bias_QSO' in self.sample_params['limits']) and (
                'beta_QSO' in self.sample_params['limits']):
            raise ValueError('Running on blind data and sampling bias_QSO and beta_QSO.')

    def read_global_cov(self, global_cov_file, scale=None):
        print(f'INFO: Reading global covariance from {global_cov_file}')
        with fits.open(utils.find_file(global_cov_file)) as hdul:
            self.global_cov = hdul[1].data['COV']

        if scale is not None:
            print('Rescaling covariance by a factor of: ', scale)
            self.global_cov *= scale
        self._use_global_cov = True

        self.full_data_mask = []
        self.full_model_mask = []
        for name in self.corr_items:
            self.full_data_mask.append(self.data[name].data_mask)
            self.full_model_mask.append(self.data[name].model_mask)

        self.full_data_mask = np.concatenate(self.full_data_mask)
        self.full_model_mask = np.concatenate(self.full_model_mask)

        if self.low_mem_mode:
            masked_cov = self.global_cov[:, self.full_data_mask]
            masked_cov = masked_cov[self.full_data_mask, :]
            del self.global_cov

            self.masked_global_log_cov_det = np.linalg.slogdet(masked_cov)[1]
            self.masked_global_invcov = np.linalg.inv(masked_cov)
            del masked_cov
        else:
            self.masked_global_invcov = utils.compute_masked_invcov(
                self.global_cov, self.full_data_mask)
            self.masked_global_log_cov_det = utils.compute_log_cov_det(
                self.global_cov, self.full_data_mask)

    def compute_sensitivity(self, nominal=None, frac=0.1, verbose=True):
        """Compute the model sensitivity to each floating parameter.

        Calculate numerical partial derivatives of the model with respect to each floating
        pararameter, evaluated at a specified point in parameter space. Calculate Fisher information
        distributed over bins of (rt,rp).  Results are stored in a dictionary attribute
        named `sensitivity` with keys `nominal`, `partials`, and `fisher`.

        Parameters
        ----------
        nominal : dict or None
            Dictionary of (value,error) tuples for each floating parameter. Uses the results
            of the last call to minimize when None, or raises a RuntimeError when minimize
            has not yet been called.
        frac : float
            Estimate partial derivatives of the likelihood using central finite differences
            at value +/- frac * error for each floating parameter.
        verbose : bool
            Print progress of the computation when True.
        """
        # Copy the baseline parameters to use.
        if nominal is None:
            if self.bestfit.params is None:
                raise RuntimeError('No nominal parameter values provided or saved by minimize()')
            nominal = {p.name: ( p.value, p.error ) for p in self.bestfit.params}
        nfloating = len(nominal)
        ninfo = (nfloating * (nfloating + 1)) // 2
        params = copy.deepcopy(self.params)
        for pname, (pvalue, perror) in nominal.items():
            params[pname] = pvalue
        # Initialize the sensitivity results.
        self.sensitivity = dict(nominal = copy.deepcopy(nominal), partials={ }, fisher={ })
        for n in self.corr_items:
            rp = self.corr_items[n].model_coordinates.rp_grid
            rt = self.corr_items[n].model_coordinates.rt_grid
            self.sensitivity['partials'][n] = np.zeros((nfloating, 2, 2, len(rp)))
            self.sensitivity['fisher'][n] = np.zeros((ninfo, 2, len(rp)))
        # Loop over fit parameters
        self.fiducial['save-components'] = True
        bao_amp = self.params['bao_amp']
        for pindex, (pname, (pvalue, perror)) in enumerate(nominal.items()):
            if verbose:
                print(f'Calculating sensitivity for [{pindex}] {pname} at {pvalue:.4f} ± {perror:.4f}')
            # Compute partial derivatives wrt to p for each multipole
            delta = frac * perror
            for sign in (+1, -1):
                params[pname] = pvalue + sign * delta
                # Compute the model for all datasets.
                cfs = self.compute_model(params, run_init=True)
                # Loop over datasets to update the partial derivative calculations.
                for n, cf in cfs.items():
                    model = self.models[n]
                    # Distorted peak
                    self.sensitivity['partials'][n][pindex,0,0] += sign * bao_amp * model.xi_distorted['peak']['core']
                    # Distorted smooth
                    self.sensitivity['partials'][n][pindex,0,1] += sign * model.xi_distorted['smooth']['core']
                    # Undistorted peak
                    self.sensitivity['partials'][n][pindex,1,0] += sign * bao_amp * model.xi['peak']['core']
                    # Distorted smooth
                    self.sensitivity['partials'][n][pindex,1,1] += sign * model.xi['smooth']['core']
            # Normalize the partial derivatives.
            for n in self.corr_items:
                self.sensitivity['partials'][n][pindex] /= 2 * delta
            # Restore the fitted parameter value.
            params[pname] = pvalue

        # Loop over pairs of fit parameters.
        if verbose:
            print('Computing Fisher information for each pair of parameters...')
        idx = 0
        for pindex1, (pname1, (pvalue1, perror1)) in enumerate(nominal.items()):
            for pindex2, (pname2, (pvalue2, perror2)) in enumerate(nominal.items()):
                if pindex1 > pindex2:
                    continue
                # Loop over datasets.
                for n in self.corr_items:
                    fisher = self.sensitivity['fisher'][n][idx]
                    # Lookup the data vector mask for this dataset
                    mask = self.data[n].data_mask
                    # Loop over distorted / non-distorted.
                    for idistort in range(2):
                        # Combine peak + smooth partials.
                        partial1 = self.sensitivity['partials'][n][pindex1,idistort].sum(axis=0)
                        partial2 = self.sensitivity['partials'][n][pindex2,idistort].sum(axis=0)
                        # Calculate the Fisher info for all unmasked correlation bins.
                        masked_info = partial1[mask] * self.data[n].inv_masked_cov.dot(partial2[mask])
                        fisher[idistort, self.data[n].data_mask] = masked_info
                        # Calculate the predicted inverse covariance for this parameter pair.
                        #ivar[idistort] = np.sum(fisher[idistort])
                        #ferror = ivar ** -0.5 if ivar > 0 else np.nan
                        # Set unused bins to NaN for plotting
                        fisher[idistort, ~mask] = np.nan
                idx += 1

