import numpy as np


class CorrelationItem:
    """Class for handling the info and config of
    each correlation function component.
    """
    cosmo_params = None
    model_coordinates = None
    dist_model_coordinates = None
    data_coordinates = None
    # _rp_rt_grid = None
    # _r_mu_grid = None
    # _z_grid = None
    # _bin_size_rp_model = None
    # _bin_size_rt_model = None
    # _bin_size_rp_data = None
    # _bin_size_rt_data = None
    # rp_min_model = None
    # rp_max_model = None
    # rt_max_model = None
    # num_bins_rp_model = None
    # num_bins_rt_model = None

    def __init__(self, config, model_pk=False):
        """

        Parameters
        ----------
        config : ConfigParser
            parsed config file
        """
        # Save the config and read the tracer info
        self.config = config
        self.model_pk = model_pk
        self.name = config['data'].get('name')
        self.tracer1 = {}
        self.tracer2 = {}
        self.tracer1['name'] = config['data'].get('tracer1')
        self.tracer1['type'] = config['data'].get('tracer1-type')
        self.tracer2['name'] = config['data'].get('tracer2', self.tracer1['name'])
        self.tracer2['type'] = config['data'].get('tracer2-type', self.tracer1['type'])

        self.cov_rescale = config['data'].getfloat('cov_rescale', 1.)
        self.has_distortion = config['data'].getboolean('distortion', True)
        self.old_fftlog = config['model'].getboolean('old_fftlog', False)

        self.has_data = config['data'].getboolean('has_datafile', True)
        if 'filename' not in config['data']:
            self.has_data = False

        self.new_metals = config['model'].getboolean('new_metals', False)
        if self.new_metals:
            self.tracer1['weights-path'] = config['data'].get('weights-tracer1')
            self.tracer2['weights-path'] = config['data'].get('weights-tracer2', None)
            if self.tracer2['weights-path'] is None:
                self.tracer2['weights-path'] = self.tracer1['weights-path']

        self.test_flag = config['data'].getboolean('test', False)

        self.has_metals = False
        self.has_bb = False

    def init_metals(self, tracer_catalog, metal_correlations):
        """Initialize the metal config.

        This should be called from the data object if we have metal matrices.

        Parameters
        ----------
        tracer_catalog : dict
            Dictionary containing all tracer objects (metals and the core ones)
        metal_correlations : list
            list of all metal correlations we need to compute
        """
        self.tracer_catalog = tracer_catalog
        self.metal_correlations = metal_correlations
        self.has_metals = True

    def init_broadband(self, coeff_binning_model):
        """Initialize the parameters we need to compute
        the broadband functions

        Parameters
        ----------
        bin_size_rp : int
            Size of r parallel bins
        coeff_binning_model : float
            Ratio of distorted coordinate grid bin size to undistorted bin size
        """
        self.coeff_binning_model = coeff_binning_model
        self.has_bb = True

    def init_coordinates(
            self, model_coordinates, dist_model_coordinates=None, data_coordinates=None):
        """Initialize the coordinate grids.

        Parameters
        ----------
        model_coordinates : Coordinates
            Coordinates for the model
        dist_model_coordinates : Coordinates, optional
            Distorted coordinates for the model, by default None
        data_coordinates : Coordinates, optional
            Coordinates for the data, by default None
        """
        self.model_coordinates = model_coordinates
        self.data_coordinates = model_coordinates if data_coordinates is None else data_coordinates
        self.dist_model_coordinates = (model_coordinates if dist_model_coordinates is None
                                       else dist_model_coordinates)

    # @property
    # def r_mu_grid(self):
    #     return self._r_mu_grid

    # @r_mu_grid.setter
    # def r_mu_grid(self, r_mu_grid):
    #     self._r_mu_grid = np.array(r_mu_grid)
    #     assert (self._r_mu_grid[1] <= 1).all()
    #     assert (self._r_mu_grid[1] >= -1).all()

    #     # Compute rp/rt from r/mu
    #     rp_grid = self._r_mu_grid[0] * self._r_mu_grid[1]
    #     rt_grid = np.sqrt(self._r_mu_grid[0]**2 - rp_grid**2)

    #     # Save the rp/rt grid
    #     self._rp_rt_grid = np.array([rp_grid, rt_grid])

    # @property
    # def rp_rt_grid(self):
    #     return self._rp_rt_grid

    # @rp_rt_grid.setter
    # def rp_rt_grid(self, rp_rt_grid):
    #     self._rp_rt_grid = np.array(rp_rt_grid)

    #     # Compute r/mu from rp/rt
    #     r_grid = np.sqrt(self._rp_rt_grid[0]**2 + self._rp_rt_grid[1]**2)
    #     mu_grid = np.zeros(r_grid.size)
    #     w = r_grid > 0.
    #     mu_grid[w] = self._rp_rt_grid[0, w] / r_grid[w]

    #     # Save the r/mu grid
    #     self._r_mu_grid = np.array([r_grid, mu_grid])

    # @property
    # def z_grid(self):
    #     return self._z_grid

    # @z_grid.setter
    # def z_grid(self, z_grid):
    #     if not isinstance(z_grid, float):
    #         assert (z_grid >= 0).all() and (z_grid <= 10).all()
    #         self._z_grid = np.array(z_grid)
    #     else:
    #         assert z_grid >= 0 and z_grid <= 10
    #         self._z_grid = z_grid

    # @property
    # def bin_size_rp_model(self):
    #     if self._bin_size_rp_model is None:
    #         raise ValueError('You must set the value of "bin_size_rp_model" in vega.corr_items to '
    #                          'compute model without data.')
    #     return self._bin_size_rp_model

    # @bin_size_rp_model.setter
    # def bin_size_rp_model(self, bin_size_rp_model):
    #     self._bin_size_rp_model = bin_size_rp_model

    # @property
    # def bin_size_rt_model(self):
    #     if self._bin_size_rt_model is None:
    #         raise ValueError('You must set the value of "bin_size_rt_model" in vega.corr_items to '
    #                          'compute model without data.')
    #     return self._bin_size_rt_model

    # @bin_size_rt_model.setter
    # def bin_size_rt_model(self, bin_size_rt_model):
    #     self._bin_size_rt_model = bin_size_rt_model

    # @property
    # def bin_size_rp_data(self):
    #     if self._bin_size_rp_data is None:
    #         return self.bin_size_rp_model
    #     return self._bin_size_rp_data

    # @bin_size_rp_data.setter
    # def bin_size_rp_data(self, bin_size_rp_data):
    #     self._bin_size_rp_data = bin_size_rp_data

    # @property
    # def bin_size_rt_data(self):
    #     if self._bin_size_rt_data is None:
    #         return self.bin_size_rt_model
    #     return self._bin_size_rt_data

    # @bin_size_rt_data.setter
    # def bin_size_rt_data(self, bin_size_rt_data):
    #     self._bin_size_rt_data = bin_size_rt_data
