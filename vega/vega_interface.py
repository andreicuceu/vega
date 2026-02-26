"""Main module."""
import os.path
import numpy as np
import scipy.stats
from astropy.io import fits
import configparser
import copy
from scipy.special import loggamma

from . import correlation_item, data, utils
from vega.scale_parameters import ScaleParameters
from vega.model import Model
from vega.minimizer import Minimizer
from vega.analysis import Analysis
from vega.output import Output
from vega.parameters.param_utils import get_default_values
from vega.plots.plot import VegaPlots
from vega.postprocess.fit_results import FitResults

import numdifftools as ndt



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

        # Initialize the compression
        self.use_compression = False
        if 'compression' in self.main_config:
            _compression_config = self.main_config['compression']
            self.use_compression = _compression_config.getboolean('use-compression', False)
            if self.use_compression:
                self._compression_type = _compression_config.get('compression-type', 'score')  
                if self._compression_type == 'score':
                    self._init_score_compression(_compression_config)
                elif self._compression_type == 'cca':
                    raise NotImplementedError('CCA compression not implemented yet')

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

        # Initialize the monte carlo flag
        self.monte_carlo = False

        # Initialize vega plots
        self.plots = None
        if self._has_data:
            self.plots = VegaPlots(vega_data=self.data)

    def _init_score_compression(self, config):

        print('INFO: Initializing compression')

        ### Likelihood and covariance information ###
        self.compression_likelihood = config.get('likelihood', 'gaussian')
        self.num_sims = config.getint('num sims', None)

        if self.compression_likelihood == 't-distribution':
            if self.num_sims is None:
                raise ValueError('Number of simulations must be specified for t-distribution')

        self._mock2mock_cov = config.get('mock-to-mock-cov', None)
        if not self._use_global_cov and self._mock2mock_cov is None:
            raise RuntimeError('Compression requires a global covariance'
                            'or mock-to-mock covariance to run compressed analysis')

        ### Compression code ###
        ### Load compression parameters ###

        compress_params_names = config.get("compression-parameters", None)
        if compress_params_names is not None:
            # (Non-maximal compression) currently not implemented
            compress_params_names = compress_params_names.split(' ')
            self.compress_params = {
                par: self.params[par] for par in compress_params_names
            }
        else:
            self.compress_params = self.sample_params["values"]

        pnames = list(self.compress_params.keys())
        p0     = np.array(list(self.compress_params.values()))

        ### Step size for numerical differentiation ###
        epses = np.full_like(p0, 1e-8, dtype=float)

        ### Define vector model functions as closures ###
        # ==============================================================

        def vector_model_full(pvec):
            """Full concatenated model vector (masked)."""
            p_dict = dict(zip(pnames, pvec))

            model = self.compute_model(params=p_dict, run_init=False)

            full_vec = np.concatenate(list(model.values()))

            return full_vec[self.full_model_mask]

        def vector_model_block(pvec, block_name):
            p_dict = dict(zip(pnames, pvec))
            model = self.compute_model(params=p_dict, run_init=False)
            return model[block_name]

        # ==============================================================

        #### Compute Jacobian of full model ###

        self._full_jacobian = ndt.Jacobian(
            vector_model_full,
            step=epses,
            method="central"
        )(p0)

        ### Build full data vector + fiducial model ###

        self._full_datavec = np.concatenate(
            [cf.data_vec for cf in self.data.values()]
        )[self.full_data_mask]

        # Full model is concatenation of all blocks
        full_model_blocks = [
            vector_model_block(p0, name) for name in self.corr_items
        ]

        self._full_fidmod = np.concatenate(full_model_blocks)[self.full_model_mask]

        residual = self._full_datavec - self._full_fidmod
    
        ### Compute score vector ###

        self.score = (
            self._full_jacobian.T
            @ self.masked_global_invcov
            @ residual
        )

        ### Metadata ###
        self.ndim_compressed = self.score.size

        ### Compressed covariance ###
        if self._mock2mock_cov is not None:
            _mock2mock_cov = np.load(self._mock2mock_cov)['cov']
            self.masked_compressed_global_cov = _mock2mock_cov
        else:
            self.masked_compressed_global_cov = (
            self._full_jacobian.T
            @ self.masked_global_invcov
            @ self._full_jacobian
        )

        self.masked_compressed_global_invcov = np.linalg.inv(
            self.masked_compressed_global_cov
        )
        
        #hartlap correction
        if self._mock2mock_cov is not None:
            alpha = (self.num_sims - self.ndim_compressed - 2) / (self.num_sims - 1)
            self.masked_compressed_global_invcov *= alpha

        self.masked_compressed_global_cov_logdet = np.linalg.slogdet(self.masked_compressed_global_cov)[1]


        print("INFO: Compression initialized successfully")

    def _compress(self, vec):
        """
        Score compress a vector
        
        :param vec: vector to compress
        :return: compressed vector (dimension = N_params)
        """
        return self._full_jacobian.T @ self.masked_global_invcov @ (vec - self._full_fidmod)
    
    def compute_compressed_model(self, params=None, run_init=True, direct_pk=None, marg_coeff=None):

        print('INFO: Computing compressed model')
        if not self.use_compression:
            raise ValueError('Compression not initialized')

        #compute model for given set of parameters (compressed?)
        model = self.compute_model(params=params, 
                                     run_init=run_init, direct_pk=direct_pk,
                                       marg_coeff=marg_coeff)

        #full model, shape = (N_auto + N_cross,)
        full_model = np.concatenate([cf for cf in model.values()])[self.full_model_mask]

        #compute compressed model
        compressed_model = self._full_jacobian.T @  self.masked_global_invcov @ (full_model - self._full_fidmod)

        return compressed_model

    def compute_model(self, params=None, run_init=True, direct_pk=None, marg_coeff=None):
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
        # Get computation parameters
        local_params = self._get_lcl_prms(params)

        # Go through each component and compute the model cf
        model_cf = {}
        if run_init:
            self.models = {}
        for name, corr_item in self.corr_items.items():
            if run_init:
                self.models[name] = Model(
                    corr_item, self.fiducial, self.scale_params, self.data[name])

            if direct_pk is None:
                model_cf[name] = self.models[name].compute(
                    local_params, self.fiducial['pk_full'], self.fiducial['pk_smooth'])
            else:
                model_cf[name] = self.models[name].compute_direct(local_params, direct_pk)

        if marg_coeff is not None:
            for name in self.data:
                if self.data[name].marg_templates is not None:
                    model_cf[name] += self.data[name].marg_templates.dot(marg_coeff[name])

        return model_cf

    def chi2(self, params=None, direct_pk=None):
        #alter function for compression
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

        # Compute model
        try:
            model_cf = self.compute_model(params, run_init=False, direct_pk=direct_pk)
        except utils.VegaModelError:
            for name in self.corr_items:
                self.models[name].PktoXi.cache_pars = None
            return 1e100

        # Compute chisq for the case where we use the global covariance
        if self._use_global_cov:
            if self.monte_carlo:
                full_masked_data = self.analysis.current_mc_mock
            else:
                full_masked_data = np.concatenate(
                    [self.data[name].masked_data_vec for name in self.corr_items])

            full_model = np.concatenate([model_cf[name] for name in self.corr_items])
            if self.use_compression:
                diff = self._compress(full_masked_data) - self._compress(full_model[self.full_model_mask])
                chi2 = diff.T.dot(self.masked_compressed_global_invcov.dot(diff))
            else:
                diff = full_masked_data - full_model[self.full_model_mask]
                chi2 = diff.T.dot(self.masked_global_invcov.dot(diff))

        # Compute chisq for the case where the correlations are independent
        else:
            chi2 = 0
            for name in self.corr_items:
                model_corr = model_cf[name][self.data[name].model_mask]
                if self.monte_carlo:
                    diff = self.data[name].masked_mc_mock - model_corr
                    chi2 += diff.T.dot(self.data[name].scaled_inv_masked_cov.dot(diff))
                else:
                    diff = self.data[name].masked_data_vec - model_corr
                    chi2 += diff.T.dot(self.data[name].inv_masked_cov.dot(diff))

        # Add priors
        chi2 += self.compute_prior_chi2(params)

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

        if self.use_compression:
            if self.compression_likelihood == 'gaussian':
                log_norm -= 0.5 * self.ndim_compressed * np.log(2 * np.pi)
                log_norm -= 0.5 * self.masked_compressed_global_cov_logdet
            elif self.compression_likelihood == 't-distribution':
                _pre_factor = loggamma(0.5 * self.num_sims)
                _pre_factor -= 0.5 * self.ndim_compressed * np.log(np.pi * (self.num_sims - 1))
                _pre_factor -= loggamma(0.5 * (self.num_sims - self.ndim_compressed))

                log_norm = _pre_factor - 0.5 * self.masked_compressed_global_cov_logdet
            else:
                raise  ValueError(
                    f'Unknown likelihood type {self.compression_likelihood}.'
                    ' Choose from ["gaussian", "t-distribution"].'
                )
        else:
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
        if self.compression_likelihood == 't-distribution':
            log_lik -= 0.5 * self.num_sims * np.log(1 + chi2 / (self.num_sims - 1))
        else:
            log_lik = log_norm - 0.5 * chi2

        # Add priors normalization
        for prior in self.priors.values():
            log_lik += self._gaussian_lik_prior(prior[1])

        return log_lik

    def _get_lcl_prms(self, params=None):
        local_params = copy.deepcopy(self.params)
        if params is not None:
            local_params |= params

        assert self._blind is not None
        if self._rnsps is not None:
            assert self._blind
            local_params = utils.apply_blinding(local_params, self._rnsps)

            # Enforce blinding
            for par, val in local_params.items():
                if par in utils.BLIND_FIXED_PARS:
                    local_params[par] = 1.

        return local_params

    def compute_prior_chi2(self, params=None):
        local_params = self._get_lcl_prms(params)

        chi2 = 0
        for param, prior in self.priors.items():
            if param not in local_params:
                err_msg = ("You have specified a prior for a parameter not in "
                           f"the model. Offending parameter: {param}")
                assert param in local_params, err_msg
            chi2 += self._gaussian_chi2_prior(local_params[param], prior[0], prior[1])

        return chi2

    def get_fiducial_for_monte_carlo(self, print_func=print):
        mc_params = self.mc_config['params']
        mc_start_from_fit = self.main_config['control'].get('mc_start_from_fit', None)

        # Read existing fit and use the bestfit values for the MC template
        if mc_start_from_fit is not None:
            print_func(f'Reading input fit {mc_start_from_fit}')
            existing_fit = FitResults(utils.find_file(mc_start_from_fit))
            mc_params = existing_fit.params | mc_params
            print_func(f'Set template parameters to {mc_params}.')

        # Do fit on input data and use the bestfit values for the MC template
        elif self.sample_params['limits']:
            print_func('Running initial fit')
            # run compute_model once to initialize all the caches
            _ = self.compute_model(run_init=False)

            # Run minimizer
            self.minimize()

            mc_params = self.bestfit.values | mc_params
            print_func(f'Set template parameters to {mc_params}.')

        # Get fiducial model
        use_measured_fiducial = self.main_config['control'].getboolean(
            'use_measured_fiducial', False)
        if use_measured_fiducial:
            fiducial_model = {}
            for name in self.corr_items.keys():
                fiducial_path = self.main_config['control'].get(f'mc_fiducial_{name}')
                with fits.open(utils.find_file(fiducial_path)) as hdul:
                    fiducial_model[name] = hdul[1].data['DA']
        else:
            use_full_pk = self.main_config['control'].getboolean('use_full_pk_for_mc', False)
            if use_full_pk:
                fiducial_model = self.compute_model(
                    mc_params, run_init=False, direct_pk=self.fiducial['pk_full'])
            else:
                fiducial_model = self.compute_model(mc_params, run_init=False)

        return fiducial_model

    def initialize_monte_carlo(self, scale=None, print_func=print):
        # Get the fiducial model
        fiducial_model = self.get_fiducial_for_monte_carlo(print_func)

        # Reset the minimizer
        sample_params = self.mc_config['sample']
        self.minimizer = Minimizer(self.chi2, sample_params)

        # Check if we need to run a forecast and get the seed
        forecast = self.main_config['control'].getboolean('forecast', False)
        seed = self.main_config['control'].getint('mc_seed', 0)

        if self._use_global_cov:
            if scale is None and 'global_cov_rescale' in self.main_config['control']:
                scale = self.main_config['control'].getfloat('global_cov_rescale')

            mocks = self.analysis.create_global_monte_carlo(
                fiducial_model, seed=seed, scale=scale, forecast=forecast)
        else:
            mocks = self.analysis.create_monte_carlo_sim(
                fiducial_model, seed=seed, scale=scale, forecast=forecast)

        # Activate monte carlo mode
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
            corr_data = self.data[name]
            data_size = corr_data.effective_data_size
            self.total_data_size += data_size

            if self.monte_carlo and self._use_global_cov:
                # TODO Figure out a better way to handle this
                chisq = 0
            elif self.monte_carlo:
                diff = corr_data.masked_mc_mock \
                    - self.bestfit_model[name][corr_data.model_mask]
                chisq = diff.T.dot(corr_data.scaled_inv_masked_cov.dot(diff))
            else:
                diff = corr_data.masked_data_vec \
                    - self.bestfit_model[name][corr_data.model_mask]
                chisq = diff.T.dot(corr_data.inv_masked_cov.dot(diff))

            # Calculate best-fitting values for the marginalized templates.
            # This approximation ignores global_cov, hence correlations between
            # CFs. Bestfit_model is updated in-place.
            bestfit_marg_coeff = None
            if corr_data.marg_diff2coeff_matrix is not None:
                bestfit_marg_coeff = corr_data.marg_diff2coeff_matrix.dot(diff)
                self.bestfit_model[name] += corr_data.marg_templates.dot(bestfit_marg_coeff)

            reduced_chisq = chisq / (data_size - num_pars)
            p_value = 1 - scipy.stats.chi2.cdf(chisq, data_size - num_pars)

            print(f'{name} chi^2/(ndata-nparam): {chisq:.1f}/({data_size}-{num_pars}) '
                  f'= {reduced_chisq:.3f}, PTE={p_value:.2f}')
            print('----------------------------------------------------')

            self.bestfit_corr_stats[name] = {
                'masked_size': data_size, 'chisq': chisq, 'reduced_chisq': reduced_chisq,
                'p_value': p_value, 'bestfit_marg_coeff': bestfit_marg_coeff
            }

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

        # Construct combined templates for the mode marginalization
        # Following just updates the covariance matrix
        # More stable inversion can be achieved through Woodbury, but
        # requires handling masked pixels without removing them from cov.
        if any(
                corr_item.marginalize_small_scales
                for corr_item in self.corr_items.values()
        ):
            print('Updating global covariance with marginalization templates.')
            j = 0
            for name in self.corr_items:
                data = self.data[name]
                ndata = data.full_data_size
                wd = data.data_mask

                if self.corr_items[name].marginalize_small_scales:
                    M1 = self.global_cov[j:j + ndata, j:j + ndata]
                    M1[np.ix_(wd, wd)] += data.cov_marg_update

                    if self.low_mem_mode:
                        del data.cov_marg_update

                j += ndata
            del j

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
        parameter, evaluated at a specified point in parameter space. Calculate Fisher information
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
            nominal = {p.name: (p.value, p.error) for p in self.bestfit.params}

        params = copy.deepcopy(self.params)
        for pname, (pvalue, perror) in nominal.items():
            params[pname] = pvalue

        # Initialize the sensitivity results.
        self.sensitivity = dict(nominal=copy.deepcopy(nominal), partials={}, fisher={})
        for name in self.corr_items:
            self.sensitivity['partials'][name] = {}
            self.sensitivity['fisher'][name] = {}

        # Loop over fit parameters
        self.fiducial['save-components'] = True
        bao_amp = self.params['bao_amp']
        for pindex, (pname, (pvalue, perror)) in enumerate(nominal.items()):
            if verbose:
                print(
                    f'Calculating sensitivity for [{pindex}] {pname} at'
                    f' {pvalue:.4f} ± {perror:.4f}'
                )

            # Compute partial derivatives wrt to p for each multipole
            delta = frac * perror
            for sign in (+1, -1):
                params[pname] = pvalue + sign * delta
                # Compute the model for all datasets.
                cfs = self.compute_model(params, run_init=True)

                # Loop over datasets to update the partial derivative calculations.
                for n, cf in cfs.items():
                    if pname not in self.sensitivity['partials'][n]:
                        rp = self.corr_items[n].model_coordinates.rp_grid
                        self.sensitivity['partials'][n][pname] = np.zeros((2, 2, len(rp)))

                    model = self.models[n]
                    # Distorted peak
                    self.sensitivity['partials'][n][pname][0, 0] += (
                        sign * bao_amp * model.xi_distorted['peak']['core'])

                    # Distorted smooth
                    self.sensitivity['partials'][n][pname][0, 1] += (
                        sign * model.xi_distorted['smooth']['core'])

                    # Undistorted peak
                    self.sensitivity['partials'][n][pname][1, 0] += (
                        sign * bao_amp * model.xi['peak']['core'])

                    # Distorted smooth
                    self.sensitivity['partials'][n][pname][1, 1] += (
                        sign * model.xi['smooth']['core'])

            # Normalize the partial derivatives.
            for n in self.corr_items:
                self.sensitivity['partials'][n][pname] /= 2 * delta

            # Restore the fitted parameter value.
            params[pname] = pvalue

        # Loop over pairs of fit parameters.
        if verbose:
            print('Computing Fisher information for each pair of parameters...')
        for pindex1, pname1 in enumerate(nominal):
            for pindex2, pname2 in enumerate(nominal):
                if pindex1 > pindex2:
                    continue

                # Loop over datasets.
                for n in self.corr_items:
                    if (pname1, pname2) not in self.sensitivity['fisher'][n]:
                        rp = self.corr_items[n].model_coordinates.rp_grid
                        self.sensitivity['fisher'][n][(pname1, pname2)] = np.zeros((2, len(rp)))

                    fisher = self.sensitivity['fisher'][n][(pname1, pname2)]
                    # Lookup the data vector mask for this dataset
                    mask = self.data[n].data_mask

                    # Loop over distorted / non-distorted.
                    for idistort in range(2):
                        # Combine peak + smooth partials.
                        partial1 = self.sensitivity['partials'][n][pname1][idistort].sum(axis=0)
                        partial2 = self.sensitivity['partials'][n][pname2][idistort].sum(axis=0)

                        # Calculate the Fisher info for all unmasked correlation bins.
                        masked_info = (
                            partial1[mask] * self.data[n].inv_masked_cov.dot(partial2[mask]))
                        fisher[idistort, mask] = masked_info
                        # Calculate the predicted inverse covariance for this parameter pair.
                        # ivar[idistort] = np.sum(fisher[idistort])
                        # ferror = ivar ** -0.5 if ivar > 0 else np.nan
                        # Set unused bins to NaN for plotting
                        fisher[idistort, ~mask] = np.nan
