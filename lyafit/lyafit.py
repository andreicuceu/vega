"""Main module."""
import os.path
import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename
import configparser

from .correlation_item import CorrelationItem
from .new_data import Data
from .new_model import Model
from . import new_utils

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
            config.read(os.path.expandvars(path))

            name = config['data'].get('name')
            self.corr_items[name] = CorrelationItem(config)

        # TODO Can we make this completely optional?
        # initialize the data
        self.data = {}
        for name, corr_item in self.corr_items.items():
            self.data[name] = Data(corr_item)

        # initialize the models
        self.models = {}
        for name, corr_item in self.corr_items.items():
            self.models[name] = Model(corr_item, self.data[name],
                                      self.fiducial)

        new_utils.cosmo_fit_func = getattr(new_utils, self.main_config.get('cosmo-fit type', 'cosmo fit func'))

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
