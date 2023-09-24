import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix

from vega.utils import find_file

BLINDING_STRATEGIES = ['desi_m2', 'desi_y1']


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
    rp_rt_custom_grid = None

    def __init__(self, corr_item):
        """

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        """
        # First save the tracer info
        self.corr_item = corr_item
        self.tracer1 = corr_item.tracer1
        self.tracer2 = corr_item.tracer2

        # Read the data file and init the corrdinate grids
        data_path = corr_item.config['data'].get('filename')
        dmat_path = corr_item.config['data'].get('distortion-file', None)
        rp_rt_grid, z_grid = self._read_data(data_path, corr_item.config['cuts'], dmat_path)
        self.corr_item.rp_rt_grid = rp_rt_grid
        self.corr_item.z_grid = z_grid
        self.corr_item.bin_size_rp_model = self.bin_size_rp_model
        self.corr_item.bin_size_rt_model = self.bin_size_rt_model
        self.corr_item.bin_size_rp_data = self.bin_size_rp_data
        self.corr_item.bin_size_rt_data = self.bin_size_rt_data

        # Read the metal file and init metals in the corr item
        if 'metals' in corr_item.config:
            tracer_catalog, metal_correlations = self._init_metals(
                                                 corr_item.config['metals'])
            self.corr_item.init_metals(tracer_catalog, metal_correlations)

        # Check if we have broadband
        if 'broadband' in corr_item.config:
            self.corr_item.init_broadband(self.coeff_binning_model)

        if not self.has_distortion:
            self._distortion_mat = np.eye(self.full_data_size)
        if not self.has_cov_mat:
            self._cov_mat = np.eye(self.full_data_size)

        self._cholesky = None
        self._scale = 1.
        self.scaled_inv_masked_cov = None
        self.scaled_log_cov_det = None

    @property
    def blind(self):
        """Blinding flag property

        Returns
        -------
        bool
            Blinding flag
        """
        return self._blind

    @property
    def data_vec(self):
        """Full data vector property

        Returns
        -------
        1D array
            Full data vector (xi)
        """
        return self._data_vec

    @property
    def masked_data_vec(self):
        """Masked data vector property

        Returns
        -------
        1D array
            Masked data vector (xi[mask])
        """
        if self._masked_data_vec is None:
            self._masked_data_vec = np.zeros(self.data_mask.sum())
            self._masked_data_vec[:] = self.data_vec[self.data_mask]
        return self._masked_data_vec

    @property
    def cov_mat(self):
        """Covariance matrix property

        Returns
        -------
        2D array
            Covariance matrix
        """
        if self._cov_mat is None:
            raise AttributeError(
                'No covariance matrix found. Check for it in the data file: ',
                self.corr_item.config['data'].get('filename'))
        return self._cov_mat

    @property
    def distortion_mat(self):
        """Distortion matrix property

        Returns
        -------
        2D array
            Distortion matrix
        """
        if self._distortion_mat is None:
            raise AttributeError(
                'No distortion matrix found. Check for it in the data file: ',
                self.corr_item.config['data'].get('filename'))
        return self._distortion_mat

    @property
    def inv_masked_cov(self):
        """Inverse masked covariance matrix property

        Returns
        -------
        2D array
            Inverse masked covariance matrix
        """
        if self._inv_masked_cov is None:
            # Compute inverse of the covariance matrix
            masked_cov = self.cov_mat[:, self.data_mask]
            masked_cov = masked_cov[self.data_mask, :]
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
        """Logarithm of the determinant of the covariance matrix property

        Returns
        -------
        float
            Logarithm of the determinant of the covariance matrix
        """
        if self._log_cov_det is None:
            # Compute the log determinant using and LDL^T decomposition
            # |C| = Product of Diagonal components of D
            masked_cov = self.cov_mat[:, self.data_mask]
            masked_cov = masked_cov[self.data_mask, :]
            _, d, __ = linalg.ldl(masked_cov)
            self._log_cov_det = np.log(d.diagonal()).sum()
            assert isinstance(self.log_cov_det, float)

        return self._log_cov_det

    @property
    def has_cov_mat(self):
        """Covariance matrix flag

        Returns
        -------
        bool
            Covariance matrix flag
        """
        return self._cov_mat is not None

    @property
    def has_distortion(self):
        """Distortion matrix flag

        Returns
        -------
        bool
            Distortion matrix flag
        """
        return self._distortion_mat is not None

    def _read_data(self, data_path, cuts_config, dmat_path=None):
        """Read the data, mask it and prepare the environment.

        Parameters
        ----------
        data_path : string
            Path to fits data file
        cuts_config : ConfigParser
            cuts section from the config file
        """
        print(f'Reading data file {data_path}\n')
        hdul = fits.open(find_file(data_path))
        header = hdul[1].header

        self._blinding_strat = None
        if 'BLINDING' in header:
            self._blinding_strat = header['BLINDING']

            if self._blinding_strat == 'none' or self._blinding_strat == 'None':
                self._blinding_strat = None

        dmat_column_name = 'DM'
        if self._blinding_strat in BLINDING_STRATEGIES:
            print(f'Warning! Running on blinded data {data_path}')
            print(f'Strategy: {self._blinding_strat}. BAO can be sampled')

            self._blind = True
            self._data_vec = hdul[1].data['DA_BLIND']
            dmat_column_name += '_BLIND'
            if dmat_column_name in hdul[1].columns.names and dmat_path is None:
                self._distortion_mat = csr_matrix(hdul[1].data[dmat_column_name])

        elif self._blinding_strat == 'desi_y3':
            raise ValueError('Fits are forbidden on Y3 data as we do not have'
                             ' a coherent blinding strategy yet.')

        elif self._blinding_strat is None:
            self._blind = False
            self._data_vec = hdul[1].data['DA']
            if dmat_column_name in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data[dmat_column_name])

        else:
            self._blind = True
            raise ValueError(f"Unknown blinding strategy {self._blinding_strat}.")

        if 'CO' in hdul[1].columns.names:
            self._cov_mat = hdul[1].data['CO']

        rp_grid = hdul[1].data['RP']
        rt_grid = hdul[1].data['RT']
        z_grid = hdul[1].data['Z']

        self.rp_min_data = header['RPMIN']
        self.rp_max_data = header['RPMAX']
        self.rt_max_data = header['RTMAX']
        self.num_bins_rp_data = header['NP']
        self.num_bins_rt_data = header['NT']

        # Get the data bin size
        # TODO If RTMIN is ever added to the cf data files this needs modifying
        self.bin_size_rp_data = (self.rp_max_data - self.rp_min_data) / self.num_bins_rp_data
        self.bin_size_rt_data = self.rt_max_data / self.num_bins_rt_data

        if 'NB' in hdul[1].columns.names:
            self.nb = hdul[1].data['NB']
        else:
            self.nb = None

        if len(hdul) > 2:
            rp_grid_model = hdul[2].data['DMRP']
            rt_grid_model = hdul[2].data['DMRT']
            z_grid_model = hdul[2].data['DMZ']
        else:
            rp_grid_model = rp_grid
            rt_grid_model = rt_grid
            z_grid_model = z_grid
        hdul.close()

        # Compute the data mask
        self.data_mask = self._build_mask(rp_grid, rt_grid, cuts_config, self.rp_min_data,
                                          self.bin_size_rp_data, self.bin_size_rt_data)

        # Read distortion matrix and initialize coordinate grids for the model
        if dmat_path is not None:
            rp_grid_model, rt_grid_model, z_grid_model = self._read_dmat(dmat_path, cuts_config)
        else:
            self.rp_min_model = self.rp_min_data
            self.rp_max_model = self.rp_max_data
            self.rt_max_model = self.rt_max_data
            self.num_bins_rp_model = self.num_bins_rp_data
            self.num_bins_rt_model = self.num_bins_rt_data
            self.bin_size_rp_model = self.bin_size_rp_data
            self.bin_size_rt_model = self.bin_size_rt_data
            self.coeff_binning_model = 1
            self.model_mask = self.data_mask

        self.data_size = len(self.masked_data_vec)
        self.full_data_size = len(self.data_vec)

        # TODO this was needed for post distortion BB polynomials. Fix at some point!
        # self.r_square_grid = np.sqrt(rp_grid**2 + rt_grid**2)
        # self.mu_square_grid = np.zeros(self.r_square_grid.size)
        # w = self.r_square_grid > 0.
        # self.mu_square_grid[w] = rp_grid[w] / self.r_square_grid[w]

        # return the model coordinate grids
        rp_rt_grid = np.array([rp_grid_model, rt_grid_model])
        return rp_rt_grid, z_grid_model

    def _check_if_blinding_matches(self, blinding_flag, dmat_path):
        if self._blinding_strat is None:
            if not (blinding_flag == 'none' or blinding_flag == 'None'):
                print(f'Warning: Data has no blinding, but distortion matrix at {dmat_path} '
                      f'has a blinding flag {blinding_flag}')
        else:
            if self._blinding_strat != blinding_flag:
                print(f'Warning: Data has a blinding flag {blinding_flag} that does not match '
                      f'the flag of the distortion matrix at {dmat_path}')

    def _read_dmat(self, dmat_path, cuts_config):
        print(f'Reading distortion matrix file {dmat_path}\n')
        hdul = fits.open(find_file(dmat_path))
        header = hdul[1].header

        if 'BLINDING' in header:
            self._check_if_blinding_matches(header['BLINDING'], dmat_path)

        dmat_column_name = 'DM'
        if 'BLINDING' in header:
            if header['BLINDING'] != 'none':
                dmat_column_name = 'DM_BLIND'
        self._distortion_mat = csr_matrix(hdul[1].data[dmat_column_name])

        rp_grid = hdul[2].data['RP']
        rt_grid = hdul[2].data['RT']
        z_grid = hdul[2].data['Z']
        hdul.close()

        self.rp_min_model = header['RPMIN']
        self.rp_max_model = header['RPMAX']
        self.rt_max_model = header['RTMAX']
        self.coeff_binning_model = header['COEFMOD']
        self.num_bins_rp_model = header['NP'] * self.coeff_binning_model
        self.num_bins_rt_model = header['NT'] * self.coeff_binning_model

        # Get the model bin size
        # TODO If RTMIN is ever added to the cf data files this needs modifying
        self.bin_size_rp_model = (self.rp_max_model - self.rp_min_model) / self.num_bins_rp_model
        self.bin_size_rt_model = self.rt_max_model / self.num_bins_rt_model

        if ((self.bin_size_rp_model != self.bin_size_rp_data)
                or (self.bin_size_rt_model != self.bin_size_rt_data)):
            rp_custom_grid = np.arange(self.rp_min_model + self.bin_size_rp_data / 2,
                                       self.rp_max_model, self.bin_size_rp_data)
            rt_custom_grid = np.arange(self.bin_size_rt_data / 2,
                                       self.rt_max_model, self.bin_size_rt_data)

            rt_custom_grid, rp_custom_grid = np.meshgrid(rt_custom_grid, rp_custom_grid)

            self.model_mask = self._build_mask(
                rp_custom_grid.flatten(), rt_custom_grid.flatten(), cuts_config, self.rp_min_model,
                self.bin_size_rp_data, self.bin_size_rt_data
            )
            self.rp_rt_custom_grid = np.r_[rp_custom_grid, rt_custom_grid]
        else:
            self.model_mask = self._build_mask(
                rp_grid, rt_grid, cuts_config, self.rp_min_model,
                self.bin_size_rp_data, self.bin_size_rt_data
            )

        return rp_grid, rt_grid, z_grid

    def _build_mask(self, rp_grid, rt_grid, cuts_config, rp_min, bin_size_rp, bin_size_rt):
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
        header : fits header
            Data file header

        Returns
        -------
        (ND Array, float, float)
            Mask, Bin size in rp, Bin size in rt
        """
        # Read the cuts
        rp_min_cut = cuts_config.getfloat('rp-min', 0.)
        rp_max_cut = cuts_config.getfloat('rp-max', 200.)

        rt_min_cut = cuts_config.getfloat('rt-min', 0.)
        rt_max_cut = cuts_config.getfloat('rt-max', 200.)

        self.r_min_cut = cuts_config.getfloat('r-min', 10.)
        self.r_max_cut = cuts_config.getfloat('r-max', 180.)

        self.mu_min_cut = cuts_config.getfloat('mu-min', -1.)
        self.mu_max_cut = cuts_config.getfloat('mu-max', +1.)

        # Compute bin centers
        bin_index_rp = np.floor((rp_grid - rp_min) / bin_size_rp)
        bin_center_rp = rp_min + (bin_index_rp + 0.5) * bin_size_rp
        bin_center_rt = (np.floor(rt_grid / bin_size_rt) + 0.5) * bin_size_rt

        bin_center_r = np.sqrt(bin_center_rp**2 + bin_center_rt**2)
        bin_center_mu = bin_center_rp / bin_center_r

        # Build the mask by comparing the data bins to the cuts
        mask = (bin_center_rp > rp_min_cut) & (bin_center_rp < rp_max_cut)
        mask &= (bin_center_rt > rt_min_cut) & (bin_center_rt < rt_max_cut)
        mask &= (bin_center_r > self.r_min_cut) & (bin_center_r < self.r_max_cut)
        mask &= (bin_center_mu > self.mu_min_cut) & (bin_center_mu < self.mu_max_cut)

        return mask

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
        tracer_catalog[self.tracer1['name']] = self.tracer1
        tracer_catalog[self.tracer2['name']] = self.tracer2

        if metals_in_tracer1 is not None:
            for metal in metals_in_tracer1:
                tracer_catalog[metal] = {'name': metal, 'type': 'continuous'}

        if metals_in_tracer2 is not None:
            for metal in metals_in_tracer2:
                tracer_catalog[metal] = {'name': metal, 'type': 'continuous'}

        metal_corr_sets = []

        # Read the metal file
        metal_hdul = fits.open(find_file(metal_config.get('filename')))

        dm_prefix = 'DM_'
        if 'BLINDING' in metal_hdul[1].header:
            if metal_hdul[1].header['BLINDING'] != 'none':
                dm_prefix = 'DM_BLIND_'

        metal_correlations = []
        # First look for correlations between tracer1 and metals
        if 'in tracer2' in metal_config:
            for metal in metals_in_tracer2:
                if not self._use_correlation(self.tracer1['name'], metal):
                    continue

                if set((self.tracer1['name'], metal)) not in metal_corr_sets:
                    metal_corr_sets.append(set((self.tracer1['name'], metal)))
                else:
                    continue

                tracers = (self.tracer1['name'], metal)
                name = self.tracer1['name'] + '_' + metal
                if 'RP_' + name not in metal_hdul[2].columns.names:
                    name = metal + '_' + self.tracer1['name']
                self._read_metal_correlation(metal_hdul, tracers, name, dm_prefix)
                metal_correlations.append(tracers)

        # Then look for correlations between metals and tracer2
        # If we have an auto-cf the files are saved in the format tracer-metal
        if 'in tracer1' in metal_config:
            for metal in metals_in_tracer1:
                if not self._use_correlation(metal, self.tracer2['name']):
                    continue

                if set((self.tracer1['name'], metal)) not in metal_corr_sets:
                    metal_corr_sets.append(set((self.tracer1['name'], metal)))
                else:
                    continue

                tracers = (metal, self.tracer2['name'])
                name = metal + '_' + self.tracer2['name']
                if 'RP_' + name not in metal_hdul[2].columns.names:
                    name = self.tracer2['name'] + '_' + metal
                self._read_metal_correlation(metal_hdul, tracers, name, dm_prefix)
                metal_correlations.append(tracers)

        # Finally look for metal-metal correlations
        # Some files are reversed order, so reverse order if we don't find it
        if ('in tracer1' in metal_config) and ('in tracer2' in metal_config):
            for i, metal1 in enumerate(metals_in_tracer1):
                j0 = i if self.tracer1 == self.tracer2 else 0

                for metal2 in metals_in_tracer2[j0:]:
                    if not self._use_correlation(metal1, metal2):
                        continue

                    if set((self.tracer1['name'], metal)) not in metal_corr_sets:
                        metal_corr_sets.append(set((self.tracer1['name'], metal)))
                    else:
                        continue

                    tracers = (metal1, metal2)
                    name = metal1 + '_' + metal2

                    if 'RP_' + name not in metal_hdul[2].columns.names:
                        name = metal2 + '_' + metal1
                    self._read_metal_correlation(metal_hdul, tracers, name, dm_prefix)
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
        if 'SiII' in name1 and 'SiII' in name2:
            return False
        else:
            return True

    def _read_metal_correlation(self, metal_hdul, tracers, name, dm_prefix):
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

        metal_mat_size = len(self.metal_rp_grids[tracers])

        dm_name = dm_prefix + name
        if dm_name in metal_hdul[2].columns.names:
            self.metal_mats[tracers] = csr_matrix(metal_hdul[2].data[dm_name])
        elif len(metal_hdul) > 3 and dm_name in metal_hdul[3].columns.names:
            self.metal_mats[tracers] = csr_matrix(metal_hdul[3].data[dm_name])
        elif self.corr_item.test_flag:
            self.metal_mats[tracers] = sparse.eye(metal_mat_size)
        else:
            raise ValueError("Cannot find correct metal matrices."
                             " Check that blinding is consistent between cf and metal files.")

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
        self.masked_mc_mock = self.mc_mock[self.model_mask]

        return self.masked_mc_mock
