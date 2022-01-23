"""Main module."""
import os.path
import numpy as np
from astropy.io import fits
import configparser
import copy

from . import correlation_item, data, utils
from vega.scale_parameters import ScaleParameters
from vega.model import Model
from vega.minimizer import Minimizer
from vega.analysis import Analysis
from vega.output import Output
from vega.postprocess.param_utils import get_default_values


class VegaInterface:
    """Main Vega class.

    Parse the main config and initialize a correlation item for each component.

    If there is data, initialize data and model objects for each component.

    Handle the parameter config and call the analysis class.
    """
    _blind = None

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

        # Initialize the individual components
        self.corr_items = {}
        for path in ini_files:
            config = configparser.ConfigParser()
            config.optionxform = lambda option: option
            config.read(utils.find_file(os.path.expandvars(path)))

            name = config['data'].get('name')
            self.corr_items[name] = correlation_item.CorrelationItem(config)

        # TODO Can we make this completely optional?
        # initialize the data
        self.data = {}
        self._has_data = True
        for name, corr_item in self.corr_items.items():
            has_datafile = corr_item.config['data'].getboolean('has_datafile',
                                                               True)
            if has_datafile:
                self.data[name] = data.Data(corr_item)
            else:
                self.data[name] = None
                self._has_data = False

        # Initialize scale parameters
        self.scale_params = ScaleParameters(self.main_config['cosmo-fit type'])

        # initialize the models
        self.models = {}
        if self._has_data:
            for name, corr_item in self.corr_items.items():
                self.models[name] = Model(corr_item, self.fiducial, self.scale_params,
                                          self.data[name])

        # Read parameters
        self.params = self._read_parameters(self.corr_items, self.main_config['parameters'])
        self.sample_params = self._read_sample(self.main_config['sample'])

        # Check blinding
        self._scale_par_names = ['ap', 'at', 'ap_sb', 'at_sb', 'phi', 'gamma', 'alpha',
                                 'phi_smooth', 'gamma_smooth', 'alpha_smooth', 'aiso', 'epsilon']
        if self._has_data:
            self._blind = False
            for data_obj in self.data.values():
                if data_obj.blind:
                    self._blind = True
            if self._blind:
                for par in self.sample_params['limits'].keys():
                    if par in self._scale_par_names:
                        raise ValueError('Running on blind data, please fix scale parameters')

        # Get priors
        self.priors = {}
        if 'priors' in self.main_config:
            self.priors = self._init_priors(self.main_config['priors'])
            for param in self.priors.keys():
                if param not in self.sample_params['limits'].keys():
                    print('Warning: Prior specified for a parameter that is'
                          ' not sampled!')

        # Read the monte carlo parameters
        self.mc_config = None
        if 'monte carlo' in self.main_config:
            self.mc_config = {}
            config = self.main_config['monte carlo']

            self.mc_config['params'] = copy.deepcopy(self.params)
            mc_params = self.main_config['mc parameters']
            for param, value in mc_params.items():
                self.mc_config['params'][param] = float(value)

            self.mc_config['sample'] = self._read_sample(config)

        # Initialize the minimizer and the analysis objects
        if not self.sample_params['limits']:
            self.minimizer = None
        else:
            self.minimizer = Minimizer(self.chi2, self.sample_params)
        self.analysis = Analysis(Minimizer(self.chi2, self.sample_params),
                                 self.main_config, self.mc_config)

        # Check for sampler
        self.has_sampler = self.main_config['control'].getboolean(
            'sampler', False)
        if self.has_sampler:
            if 'Polychord' not in self.main_config:
                raise RuntimeError('run_sampler called, but'
                                   ' no sampler initialized')

        self.output = Output(self.main_config['output'], self.data, self.corr_items, self.analysis)

        self.monte_carlo = False

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

        # Check if blinding is initialized
        if self._blind is None:
            self._blind = False
            for data_obj in self.data.values():
                if data_obj.blind:
                    self._blind = True

        # Overwrite computation parameters
        local_params = copy.deepcopy(self.params)
        if params is not None:
            for par, val in params.items():
                local_params[par] = val

        # Enforce blinding
        if self._blind:
            for par, val in local_params.items():
                if par in self._scale_pars:
                    local_params[par] = 1.

        # Go trough each component and compute the chi^2
        chi2 = 0
        for name in self.corr_items:
            if direct_pk is None:
                model_cf = self.models[name].compute(local_params, self.fiducial['pk_full'],
                                                     self.fiducial['pk_smooth'])
            else:
                model_cf = self.models[name].compute_direct(local_params, direct_pk)

            if self.monte_carlo:
                diff = self.data[name].masked_mc_mock - model_cf[self.data[name].mask]
                chi2 += diff.T.dot(self.data[name].scaled_inv_masked_cov.dot(diff))
            else:
                diff = self.data[name].masked_data_vec - model_cf[self.data[name].mask]
                chi2 += diff.T.dot(self.data[name].inv_masked_cov.dot(diff))

        # Add priors
        for param, prior in self.priors.items():
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

            if self.monte_carlo:
                log_norm -= 0.5 * self.data[name].scaled_log_cov_det
            else:
                log_norm -= 0.5 * self.data[name].log_cov_det

        # Compute log lik
        log_lik = log_norm - 0.5 * chi2

        # Add priors normalization
        for param, prior in self.priors.items():
            log_lik += self._gaussian_lik_prior(prior[1])

        return log_lik

    def monte_carlo_sim(self, params=None, scale=None, seed=0, forecast=False):
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
            mocks[name] = self.data[name].create_monte_carlo(fiducial_model, item_scale, seed,
                                                             forecast)

        self.monte_carlo = True
        return mocks

    def minimize(self):
        """Minimize the chi2 over the sampled parameters.
        """
        if self.minimizer is None:
            print("No sampled parameters. Skipping minimization.")
            return
        self.set_fast_metals()
        self.minimizer.minimize()

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
