"""Main module."""
import os.path
import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename
import configparser

from . import correlation_item, data, model, utils


class LyaFit:
    """Main lyafit class
    Handles the parsing of the main config
    Initializes a correlation item for each component
    Handles the data and model objects for each component
    Handles the parameters and calls the analysis class
    """

    def __init__(self, main_path):
        # Read the main config file
        self.main_config = configparser.ConfigParser()
        self.main_config.optionxform = lambda option: option
        self.main_config.read(main_path)

        # Read the fiducial pk file
        self.fiducial = self._read_fiducial(self.main_config['fiducial'])

        # Read the effective redshift and the data config paths
        self.fiducial['z_eff'] = self.main_config['data sets'].getfloat('zeff')
        ini_files = self.main_config['data sets'].get('ini files').split()

        # Initialize the individual components
        self.corr_items = {}
        for path in ini_files:
            config = configparser.ConfigParser()
            config.optionxform = lambda option: option
            config.read(os.path.expandvars(path))

            name = config['data'].get('name')
            self.corr_items[name] = correlation_item.CorrelationItem(config)

        # TODO Can we make this completely optional?
        # initialize the data
        self.data = {}
        for name, corr_item in self.corr_items.items():
            self.data[name] = data.Data(corr_item)

        # initialize the models
        self.models = {}
        for name, corr_item in self.corr_items.items():
            self.models[name] = model.Model(corr_item, self.data[name],
                                            self.fiducial)

        # TODO Get rid of this and replace with something better
        utils.cosmo_fit_func = getattr(
            utils, self.main_config.get('cosmo-fit type', 'cosmo fit func'))

        # TODO add option to read a setup config
        # Read parameters
        self._read_parameters()

    def compute_model(self, params=None):
        """Compute correlation function model using input parameters

        Parameters
        ----------
        params : dict
            Computation parameters, will overwrite the saved ones

        Returns
        -------
        dict
            Dictionary of cf models for each component
        """
        # Overwrite computation parameters
        if params is not None:
            for par, val in params.items():
                self.params[par] = val

        # Go through each component and compute the model cf
        model_cf = {}
        for name in self.corr_items:
            model_cf[name] = self.models[name].compute(
                params, self.fiducial['pk_ful'], self.fiducial['pk_smooth'])

        return model_cf

    def chi2(self, params=None):
        """Compute full chi2 for all components

        Parameters
        ----------
        params : dict
            Computation parameters, will overwrite the saved ones

        Returns
        -------
        float
            chi^2
        """
        # Overwrite computation parameters
        if params is not None:
            for par, val in params.items():
                self.params[par] = val

        # Go trough each component and compute the chi^2
        chi2 = 0
        for name in self.corr_items:
            model_cf = self.models[name].compute(
                params, self.fiducial['pk_ful'], self.fiducial['pk_smooth'])

            diff = self.data[name].masked_data_vec \
                - model_cf[self.data[name].mask]
            chi2 += diff.T.dot(self.data[name].inv_masked_cov.dot(diff))

        assert isinstance(chi2, float)
        return chi2

    def log_lik(self, params=None):
        """Compute full log likelihood for all components

        Parameters
        ----------
        params : dict
            Computation parameters, will overwrite the saved ones

        Returns
        -------
        float
            log Likelihood
        """
        # Get the full chi2
        chi2 = self.chi2(params)

        # Compute the normalization for each component
        log_norm = 0
        for name in self.corr_items:
            log_norm -= 0.5 * self.data[name].data_size * np.log(2 * np.pi)
            log_norm -= 0.5 * self.data[name].log_cov_det

        # Compute log lik
        log_lik = log_norm - 0.5 * chi2

        return log_lik

    def _read_parameters(self):
        """Read computation parameters
        If a parameter is specified multiple times,
        the parameters in the main config file have priority
        """
        self.params = {}

        # First get the parameters from each component config
        for name, corr_item in self.corr_items.items():
            if 'parameters' in corr_item.config:
                for param, value in corr_item.config.items('parameters'):
                    self.params[param] = float(value)

        # Next get the parameters in the main config
        for param, value in self.main_config.items('parameters'):
            self.params[param] = float(value)

    @staticmethod
    def _read_fiducial(fiducial_config):
        """Read the fiducial pk file and get the configs

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
        path = os.path.expandvars(path)
        if not os.path.isfile(path):
            path = resource_filename('lyafit', 'models') + '/{}'.format(path)
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

        # check full shape or smooth scaling
        fiducial['full-shape'] = fiducial_config.getboolean(
                                    'full-shape', False)
        fiducial['smooth-scaling'] = fiducial_config.getboolean(
                                    'smooth-scaling', False)
        if fiducial['full-shape'] or fiducial['smooth-scaling']:
            print('WARNING!!!: Using full-shape fit or scaling of the \
            smooth cf component. Sailor you are reaching unexplored \
            territories, precede at your own risk.')

        return fiducial
