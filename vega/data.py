import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy.sparse import csr_matrix

from vega.utils import find_file


class Data:
    """Class for handling lya forest correlation function data.

    An instance of this is required for each cf component
    """
    _data_vec = None
    _masked_data_vec = None
    _cov_mat = None
    _distortion_mat = None
    _inv_masked_cov = None
    _log_cov_det = None
    _blind = None

    def __init__(self, corr_item):
        """

        Parameters
        ----------
        data_config : CorrelationItem
            Item object with the component config
        """
        # First save the tracer info
        self._corr_item = corr_item
        self._tracer1 = corr_item.tracer1
        self._tracer2 = corr_item.tracer2

        # Read the data file and init the corrdinate grids
        data_path = corr_item.config['data'].get('filename')
        rp_rt_grid, z_grid = self._read_data(data_path,
                                             corr_item.config['cuts'])
        self._corr_item.rp_rt_grid = rp_rt_grid
        self._corr_item.z_grid = z_grid

        # Read the metal file and init metals in the corr item
        if 'metals' in corr_item.config:
            tracer_catalog, metal_correlations = self._init_metals(
                                                 corr_item.config['metals'])
            self._corr_item.init_metals(tracer_catalog, metal_correlations)

        # Check if we have broadband
        if 'broadband' in corr_item.config:
            self._corr_item.init_broadband(self.bin_size_rp,
                                           self.coeff_binning_model)

        self._cholesky = None
        self._scale = 1.
        self.scaled_inv_masked_cov = None
        self.scaled_log_cov_det = None

    @property
    def blind(self):
        return self._blind

    @property
    def data_vec(self):
        return self._data_vec

    @property
    def masked_data_vec(self):
        if self._masked_data_vec is None:
            self._masked_data_vec = np.zeros(self.mask.sum())
            self._masked_data_vec[:] = self.data_vec[self.mask]
        return self._masked_data_vec

    @property
    def cov_mat(self):
        if self._cov_mat is None:
            raise AttributeError(
                'No covariance matrix found. Check for it in the data file: ',
                self._corr_item.config['data'].get('filename'))
        return self._cov_mat

    @property
    def distortion_mat(self):
        if self._distortion_mat is None:
            raise AttributeError(
                'No distortion matrix found. Check for it in the data file: ',
                self._corr_item.config['data'].get('filename'))
        return self._distortion_mat

    @property
    def inv_masked_cov(self):
        if self._inv_masked_cov is None:
            # Compute inverse of the covariance matrix
            masked_cov = self.cov_mat[:, self.mask]
            masked_cov = masked_cov[self.mask, :]
            try:
                linalg.cholesky(self.cov_mat)
                print('LOG: Full matrix is positive definite')
            except linalg.LinAlgError:
                print('WARNING: Full matrix is not positive definite')
            try:
                linalg.cholesky(masked_cov)
                print('LOG: Reduced matrix is positive definite')
            except linalg.LinAlgError:
                print('WARNING: Reduced matrix is not positive definite')
            self._inv_masked_cov = linalg.inv(masked_cov)

        return self._inv_masked_cov

    @property
    def log_cov_det(self):
        if self._log_cov_det is None:
            # Compute the log determinant using and LDL^T decomposition
            # |C| = Product of Diagonal components of D
            masked_cov = self.cov_mat[:, self.mask]
            masked_cov = masked_cov[self.mask, :]
            _, d, __ = linalg.ldl(masked_cov)
            self._log_cov_det = np.log(d.diagonal()).sum()
            assert isinstance(self.log_cov_det, float)

        return self._log_cov_det

    def has_cov_mat(self):
        return self._cov_mat is not None

    def has_distortion(self):
        return self._distortion_mat is not None

    def _read_data(self, data_path, cuts_config):
        """Read the data, mask it and prepare the environment.

        Parameters
        ----------
        data_path : string
            Path to fits data file
        cuts_config : ConfigParser
            cuts section from the config file
        """
        print('Reading data file {}\n'.format(data_path))
        hdul = fits.open(find_file(data_path))

        blinding = 'none'
        if 'BLINDING' in hdul[1].header:
            blinding = hdul[1].header['BLINDING']

        if blinding == 'corr_yshift':
            print('Warning! Running on blinded data {}'.format(data_path))
            print('Strategy: corr_yshift. BAO can be sampled')
            self._blind = False
            self._data_vec = hdul[1].data['DA_BLIND']
            if 'DM' in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data['DM_BLIND'])
        elif blinding == 'minimal':
            print('Warning! Running on blinded data {}'.format(data_path))
            print('Strategy: minimal. Scale parameters must be fixed to 1.')
            self._blind = True
            self._data_vec = hdul[1].data['DA_BLIND']
            if 'DM' in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data['DM_BLIND'])
        elif blinding == 'none':
            self._blind = False
            self._data_vec = hdul[1].data['DA']
            if 'DM' in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data['DM'])
        else:
            self._blind = True
            raise ValueError("Unknown blinding strategy. Only 'minimal' implemented.")

        if 'CO' in hdul[1].columns.names:
            self._cov_mat = hdul[1].data['CO']

        rp_grid = hdul[1].data['RP']
        rt_grid = hdul[1].data['RT']
        z_grid = hdul[1].data['Z']
        if 'NB' in hdul[1].columns.names:
            self.nb = hdul[1].data['NB']
        else:
            self.nb = None

        try:
            dist_rp_grid = hdul[2].data['DMRP']
            dist_rt_grid = hdul[2].data['DMRT']
            dist_z_grid = hdul[2].data['DMZ']
        except (IndexError, KeyError):
            dist_rp_grid = rp_grid.copy()
            dist_rt_grid = rt_grid.copy()
            dist_z_grid = z_grid.copy()
        self.coeff_binning_model = np.sqrt(dist_rp_grid.size / rp_grid.size)

        # Compute the mask and use it on the data
        self.mask, self.bin_size_rp, self.bin_size_rt = self._build_mask(rp_grid, rt_grid,
                                                                         cuts_config,
                                                                         hdul[1].header)

        self.data_size = len(self.masked_data_vec)
        self.full_data_size = len(self.data_vec)

        hdul.close()

        self.r_square_grid = np.sqrt(rp_grid**2 + rt_grid**2)
        self.mu_square_grid = np.zeros(self.r_square_grid.size)
        w = self.r_square_grid > 0.
        self.mu_square_grid[w] = rp_grid[w] / self.r_square_grid[w]

        # return the coordinate grids
        rp_rt_grid = np.array([dist_rp_grid, dist_rt_grid])
        return rp_rt_grid, dist_z_grid

    @staticmethod
    def _build_mask(rp_grid, rt_grid, cuts_config, data_header):
        """Build the mask for the data by comparing
        the cuts from config with the data limits.

        Parameters
        ----------
        rp_grid : 1D Array
            Vector of data rp coordinates
        rt_grid : 1D Array
            Vector of data rt coordinates
        cuts_config : ConfigParser
            cuts section from config
        data_header : fits header
            Data file header

        Returns
        -------
        ND Array
            Mask
        """
        # Read the cuts
        rp_min = cuts_config.getfloat('rp-min', 0.)
        rp_max = cuts_config.getfloat('rp-max', 200.)

        rt_min = cuts_config.getfloat('rt-min', 0.)
        rt_max = cuts_config.getfloat('rt-max', 200.)

        r_min = cuts_config.getfloat('r-min', 10.)
        r_max = cuts_config.getfloat('r-max', 180.)

        mu_min = cuts_config.getfloat('mu-min', -1.)
        mu_max = cuts_config.getfloat('mu-max', +1.)

        # TODO If RTMIN is ever added to the cf data files this needs modifying
        # Get the data bin size
        bin_size_rp = (data_header['RPMAX'] - data_header['RPMIN'])
        bin_size_rp /= data_header['NP']
        bin_size_rt = data_header['RTMAX'] / data_header['NT']

        # Compute bin centers
        bin_center_rp = np.zeros(rp_grid.size)
        for i, rp_value in enumerate(rp_grid):
            bin_index = np.floor((rp_value - data_header['RPMIN'])
                                 / bin_size_rp)
            bin_center_rp[i] = data_header['RPMIN'] \
                + (bin_index + 0.5) * bin_size_rp

        bin_center_rt = np.zeros(rt_grid.size)
        for i, rt_value in enumerate(rt_grid):
            bin_index = np.floor(rt_value / bin_size_rt)
            bin_center_rt[i] = (bin_index + 0.5) * bin_size_rt

        bin_center_r = np.sqrt(bin_center_rp**2 + bin_center_rt**2)
        bin_center_mu = bin_center_rp / bin_center_r

        # Build the mask by comparing the data bins to the cuts
        mask = (bin_center_rp > rp_min) & (bin_center_rp < rp_max)
        mask &= (bin_center_rt > rt_min) & (bin_center_rt < rt_max)
        mask &= (bin_center_r > r_min) & (bin_center_r < r_max)
        mask &= (bin_center_mu > mu_min) & (bin_center_mu < mu_max)

        return mask, bin_size_rp, bin_size_rt

    def _init_metals(self, metal_config):
        """Read the metal file and initialize all the metal data.

        Parameters
        ----------
        metal_config : ConfigParser
            metals section from the config file

        Returns
        -------
        dict
            Dictionary containing all tracer objects (metals and the core ones)
        list
            list of all metal correlations we need to compute
        """
        assert ('in tracer1' in metal_config) or ('in tracer2' in metal_config)

        # Read metal tracers
        metals_in_tracer1 = None
        metals_in_tracer2 = None
        if 'in tracer1' in metal_config:
            metals_in_tracer1 = metal_config.get('in tracer1').split()
        if 'in tracer2' in metal_config:
            metals_in_tracer2 = metal_config.get('in tracer2').split()

        self.metal_mats = {}
        self.metal_rp_grids = {}
        self.metal_rt_grids = {}
        self.metal_z_grids = {}

        # Build tracer Catalog
        tracer_catalog = {}
        tracer_catalog[self._tracer1['name']] = self._tracer1
        tracer_catalog[self._tracer2['name']] = self._tracer2

        if metals_in_tracer1 is not None:
            for metal in metals_in_tracer1:
                tracer_catalog[metal] = {'name': metal, 'type': 'continuous'}

        if metals_in_tracer2 is not None:
            for metal in metals_in_tracer2:
                tracer_catalog[metal] = {'name': metal, 'type': 'continuous'}

        # Read the metal file
        metal_hdul = fits.open(find_file(metal_config.get('filename')))

        metal_correlations = []
        # First look for correlations between tracer1 and metals
        if 'in tracer2' in metal_config:
            for metal in metals_in_tracer2:
                if not self._use_correlation(self._tracer1['name'], metal):
                    continue
                tracers = (self._tracer1['name'], metal)
                name = self._tracer1['name'] + '_' + metal
                if 'RP_' + name not in metal_hdul[2].columns.names:
                    name = metal + '_' + self._tracer1['name']
                self._read_metal_correlation(metal_hdul, tracers, name)
                metal_correlations.append(tracers)

        # Then look for correlations between metals and tracer2
        # If we have an auto-cf the files are saved in the format tracer-metal
        if 'in tracer1' in metal_config:
            for metal in metals_in_tracer1:
                if not self._use_correlation(metal, self._tracer2['name']):
                    continue
                tracers = (metal, self._tracer2['name'])
                name = metal + '_' + self._tracer2['name']
                if 'RP_' + name not in metal_hdul[2].columns.names:
                    name = self._tracer2['name'] + '_' + metal
                self._read_metal_correlation(metal_hdul, tracers, name)
                metal_correlations.append(tracers)

        # Finally look for metal-metal correlations
        # Some files are reversed order, so reverse order if we don't find it
        if ('in tracer1' in metal_config) and ('in tracer2' in metal_config):
            for i, metal1 in enumerate(metals_in_tracer1):
                j0 = i if self._tracer1 == self._tracer2 else 0

                for metal2 in metals_in_tracer2[j0:]:
                    if not self._use_correlation(metal1, metal2):
                        continue
                    tracers = (metal1, metal2)
                    name = metal1 + '_' + metal2

                    if 'RP_' + name not in metal_hdul[2].columns.names:
                        name = metal2 + '_' + metal1
                    self._read_metal_correlation(metal_hdul, tracers, name)
                    metal_correlations.append(tracers)

        metal_hdul.close()

        return tracer_catalog, metal_correlations

    @staticmethod
    def _use_correlation(name1, name2):
        """Check if a correlation should be used or not

        Parameters
        ----------
        name1 : string
            Name of tracer 1
        name2 : string
            Name of tracer 2

        Returns
        -------
        Bool
            Flag for using the correlation between tracer 1 and 2
        """
        # For CIV we only want it's autocorrelation
        if name1 == 'CIV(eff)' or name2 == 'CIV(eff)':
            return name1 == name2
        else:
            return True

    def _read_metal_correlation(self, metal_hdul, tracers, name):
        """Read a metal correlation from the metal file and add
        the data to the existing member dictionaries.

        Parameters
        ----------
        metal_hdul : hduList
            hduList object for the metal file
        tracers : tuple
            Tuple with the names of the two tracers
        name : string
            The name of the specific correlation to be read from file
        """
        self.metal_rp_grids[tracers] = metal_hdul[2].data['RP_' + name]
        self.metal_rt_grids[tracers] = metal_hdul[2].data['RT_' + name]
        self.metal_z_grids[tracers] = metal_hdul[2].data['Z_' + name]

        dm_name = 'DM_' + name
        if dm_name in metal_hdul[2].columns.names:
            self.metal_mats[tracers] = csr_matrix(metal_hdul[2].data[dm_name])
        else:
            self.metal_mats[tracers] = csr_matrix(metal_hdul[3].data[dm_name])

    def create_monte_carlo(self, fiducial_model, scale=1., seed=0,
                           forecast=False):
        """Create monte carlo mock of data using a fiducial model.

        Parameters
        ----------
        fiducial_model : 1D Array
            Fiducial model of the data
        scale : float, optional
            Scaling for the covariance, by default 1.
        seed : int, optional
            Seed for the random number generator, by default 0
        forecast : boolean, optional
            Forecast option. If true, we don't add noise to the mock,
            by default False

        Returns
        -------
        1D Array
            Monte Carlo mock of the data
        """
        # Check if scale has changed and we need to recompute
        if np.isclose(scale, self._scale):
            self._recompute = False
        else:
            self._scale = scale
            self._recompute = True
            self.scaled_inv_masked_cov = self.inv_masked_cov / self._scale
            self.scaled_log_cov_det = np.log(self._scale) + self.log_cov_det

        if self.scaled_inv_masked_cov is None:
            self.scaled_inv_masked_cov = self.inv_masked_cov
        if self.scaled_log_cov_det is None:
            self.scaled_log_cov_det = self.log_cov_det

        # Compute cholesky decomposition
        if (self._cholesky is None or self._recompute) and not forecast:
            self._cholesky = linalg.cholesky(self._scale * self.cov_mat)

        # Create the mock
        np.random.seed(seed)
        if forecast:
            self.mc_mock = fiducial_model
        else:
            ran_vec = np.random.randn(self.full_data_size)
            self.mc_mock = self._cholesky.dot(ran_vec) + fiducial_model
        self.masked_mc_mock = self.mc_mock[self.mask]

        return self.masked_mc_mock
