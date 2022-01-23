import numpy as np


class CorrelationItem:
    """Class for handling the info and config of
    each correlation function component.
    """
    _rp_rt_grid = None
    _r_mu_grid = None
    _z_grid = None

    def __init__(self, config):
        """

        Parameters
        ----------
        config : ConfigParser
            parsed config file
        """
        # Save the config and read the tracer info
        self.config = config
        self.name = config['data'].get('name')
        self.tracer1 = {}
        self.tracer2 = {}
        self.tracer1['name'] = config['data'].get('tracer1')
        self.tracer1['type'] = config['data'].get('tracer1-type')
        self.tracer2['name'] = config['data'].get('tracer2',
                                                  self.tracer1['name'])
        self.tracer2['type'] = config['data'].get('tracer2-type',
                                                  self.tracer1['type'])
        self.cov_rescale = config['data'].getfloat('cov_rescale', 1.)
        self.has_distortion = config['data'].getboolean('distortion', True)
        self.old_fftlog = config['model'].getboolean('old_fftlog', False)
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

    def init_broadband(self, bin_size_rp, coeff_binning_model):
        """Initialize the parameters we need to compute
        the broadband functions

        Parameters
        ----------
        bin_size_rp : int
            Size of r parallel bins
        coeff_binning_model : float
            Ratio of distorted coordinate grid bin size to undistorted bin size
        """
        self.bin_size_rp = bin_size_rp
        self.coeff_binning_model = coeff_binning_model
        self.has_bb = True

    @property
    def r_mu_grid(self):
        return self._r_mu_grid

    @r_mu_grid.setter
    def r_mu_grid(self, r_mu_grid):
        self._r_mu_grid = np.array(r_mu_grid)
        assert (self._r_mu_grid[1] <= 1).all()
        assert (self._r_mu_grid[1] >= -1).all()

        # Compute rp/rt from r/mu
        rp_grid = self._r_mu_grid[0] * self._r_mu_grid[1]
        rt_grid = np.sqrt(self._r_mu_grid[0]**2 - rp_grid**2)

        # Save the rp/rt grid
        self._rp_rt_grid = np.array([rp_grid, rt_grid])

    @property
    def rp_rt_grid(self):
        return self._rp_rt_grid

    @rp_rt_grid.setter
    def rp_rt_grid(self, rp_rt_grid):
        self._rp_rt_grid = np.array(rp_rt_grid)

        # Compute r/mu from rp/rt
        r_grid = np.sqrt(self._rp_rt_grid[0]**2 + self._rp_rt_grid[1]**2)
        mu_grid = np.zeros(r_grid.size)
        w = r_grid > 0.
        mu_grid[w] = self._rp_rt_grid[0, w] / r_grid[w]

        # Save the r/mu grid
        self._r_mu_grid = np.array([r_grid, mu_grid])

    @property
    def z_grid(self):
        return self._z_grid

    @z_grid.setter
    def z_grid(self, z_grid):
        if not isinstance(z_grid, float):
            assert (z_grid >= 0).all() and (z_grid <= 10).all()
            self._z_grid = np.array(z_grid)
        else:
            assert z_grid >= 0 and z_grid <= 10
            self._z_grid = z_grid
