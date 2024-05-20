import numpy as np
from astropy.io import fits
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix

from vega.utils import find_file, compute_masked_invcov, compute_log_cov_det
from vega.coordinates import Coordinates, PkCoordinates

BLINDING_STRATEGIES = ['desi_m2', 'desi_y1']


class Data:
    """Class for handling lya forest correlation function data.

    An instance of this is required for each cf component
    """
    _mode = None
    _data_vec = None
    _masked_data_vec = None
    _cov_mat = None
    _distortion_mat = None
    _inv_masked_cov = None
    _log_cov_det = None
    _blind = None
    cosmo_params = None
    dist_model_coordinates = None
    model_coordinates = None
    data_coordinates = None

    def __init__(self, corr_item):
        """Read the data and initialize the coordinate grids.

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        """
        # First save the tracer info
        self.corr_item = corr_item
        self.tracer1 = corr_item.tracer1
        self.tracer2 = corr_item.tracer2
        self.use_metal_autos = corr_item.config['model'].getboolean('use_metal_autos', True)
        self.cholesky_masked_cov = corr_item.config['data'].getboolean('cholesky-masked-cov', True)

        # Read the data file and init the corrdinate grids
        data_path = corr_item.config['data'].get('filename')
        dmat_path = corr_item.config['data'].get('distortion-file', None)
        cov_path = corr_item.config['data'].get('covariance-file', None)
        cov_rescale = corr_item.config['data'].getfloat('cov_rescale', None)

        if corr_item.mode == 'xi_rprt':
            self._read_data(data_path, corr_item.config['cuts'], dmat_path, cov_path, cov_rescale)
        elif corr_item.mode == 'pk_ell':
            self._read_pk_ell(data_path, corr_item.config['cuts'], dmat_path, cov_path, cov_rescale)

        self.corr_item.init_coordinates(
            self.model_coordinates, self.dist_model_coordinates, self.data_coordinates)

        # Read the metal file and init metals in the corr item
        if 'metals' in corr_item.config:
            if not corr_item.new_metals:
                tracer_catalog, metal_correlations = self._init_metals(corr_item.config['metals'])
            else:
                metals_in_tracer1, metals_in_tracer2, tracer_catalog = self._init_metal_tracers(
                    corr_item.config['metals'])
                metal_correlations = self._init_metal_correlations(
                    corr_item.config['metals'], metals_in_tracer1, metals_in_tracer2)

            self.corr_item.init_metals(tracer_catalog, metal_correlations)

        # Check if we have broadband
        if 'broadband' in corr_item.config:
            self.corr_item.init_broadband(self.coeff_binning_model)

        if self.cosmo_params is not None:
            self.corr_item.cosmo_params = self.cosmo_params

        if not self.has_distortion:
            self._distortion_mat = np.eye(self.full_data_size)
        if not self.has_cov_mat and not self.corr_item.low_mem_mode:
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
            self._inv_masked_cov = compute_masked_invcov(self.cov_mat, self.data_mask)

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
            self._log_cov_det = compute_log_cov_det(self.cov_mat, self.data_mask)

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

    def _read_pk_ell(self, data_path, cuts_config, dmat_path, cov_path, cov_rescale=None):
        print(f'Reading data file {data_path}\n')
        data = np.loadtxt(data_path)

        # Get normalization factor
        with open(data_path) as f:
            lines = f.readlines()
            assert 'norm = ' in lines[1]
            self.pk_norm = float(lines[1].split('=')[-1].strip())

        k_edges = data[:, :2]
        pk_ell = (data[:, 2:] * self.pk_norm).flatten()

        # Initialize the data coordinates and mask
        self._blind = False
        self._blinding_strat = None
        self.data_coordinates = PkCoordinates(k_edges, num_ells=data[:, 2:].shape[1])
        self.data_mask = self.data_coordinates.get_mask_scale_cuts(cuts_config).flatten()
        self.model_mask = self.data_mask

        # Save data and compute data size
        assert self.data_mask.shape == pk_ell.shape
        self._masked_data_vec = pk_ell[self.data_mask]
        self._data_vec = pk_ell
        self.data_size = self.data_mask.sum()
        self.full_data_size = pk_ell.size

        # Read distortion matrix and initialize coordinate grids for the model
        assert dmat_path is not None, 'Distortion matrix file is required for P(k) data'
        self._read_dmat(dmat_path)
        assert self.cosmo_params is not None, 'Fiducial cosmo parameters in dmat file are required.'

        # Compute the model mask
        # self.model_mask = self.dist_model_coordinates.get_mask_scale_cuts(cuts_config)

        # Read the covariance matrix
        assert cov_path is not None, 'Covariance matrix file is required for P(k) data'
        print(f'Reading covariance matrix file {cov_path}\n')
        self._cov_mat = np.loadtxt(cov_path)
        assert self._cov_mat.shape == (self.full_data_size, self.full_data_size)

        if cov_rescale is not None:
            self._cov_mat *= cov_rescale

    def _read_data(self, data_path, cuts_config, dmat_path=None, cov_path=None, cov_rescale=None):
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

        # Read the data vector
        self._blinding_strat = None
        if 'BLINDING' in header:
            self._blinding_strat = header['BLINDING']

            if self._blinding_strat == 'none' or self._blinding_strat == 'None':
                self._blinding_strat = None

        if self._blinding_strat in BLINDING_STRATEGIES:
            print(f'Strategy: {self._blinding_strat}. BAO can be sampled')

            self._blind = True
            if 'DA_BLIND' in hdul[1].columns.names:
                print(f'Warning! Running on blinded data {data_path}')
                print('Using DA_BLIND column')
                self._data_vec = hdul[1].data['DA_BLIND']
            elif 'DA' in hdul[1].columns.names:
                print('Using DA column - No BAO blinding.')
                self._data_vec = hdul[1].data['DA']
            else:
                raise ValueError('No DA or DA_BLIND column found in data file.')

        elif self._blinding_strat == 'desi_y3':
            raise ValueError('Fits are forbidden on Y3 data as we do not have'
                             ' a coherent blinding strategy yet.')

        elif self._blinding_strat is None:
            self._blind = False
            self._data_vec = hdul[1].data['DA']

        else:
            self._blind = True
            raise ValueError(f"Unknown blinding strategy {self._blinding_strat}.")

        if dmat_path is None:
            if 'DM_BLIND' in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data['DM_BLIND'])
            elif 'DM' in hdul[1].columns.names:
                self._distortion_mat = csr_matrix(hdul[1].data['DM'])

        # Read the covariance matrix
        if not self.corr_item.low_mem_mode:
            if cov_path is not None:
                print(f'Reading covariance matrix file {cov_path}\n')
                with fits.open(find_file(cov_path)) as cov_hdul:
                    self._cov_mat = cov_hdul[1].data['CO']
            elif 'CO' in hdul[1].columns.names:
                self._cov_mat = hdul[1].data['CO']

            if cov_rescale is not None:
                self._cov_mat *= cov_rescale

        # Get the cosmological parameters
        if "OMEGAM" in header:
            self.cosmo_params = {}
            self.cosmo_params['Omega_m'] = header['OMEGAM']
            self.cosmo_params['Omega_k'] = header.get('OMEGAK', 0.)
            self.cosmo_params['Omega_r'] = header.get('OMEGAR', 0.)
            self.cosmo_params['wl'] = header.get('WL', -1.)

        # Get the number of pairs
        if 'NB' in hdul[1].columns.names:
            self.nb = hdul[1].data['NB']
        else:
            self.nb = None

        # Initialize the data coordinates
        self.data_coordinates = Coordinates(
            header['RPMIN'], header['RPMAX'], header['RTMAX'], header['NP'], header['NT'],
            rp_grid=hdul[1].data['RP'], rt_grid=hdul[1].data['RT'], z_grid=hdul[1].data['Z'],
        )

        if dmat_path is None:
            if len(hdul) > 2:
                rp_grid_model = hdul[2].data['DMRP']
                rt_grid_model = hdul[2].data['DMRT']
                z_grid_model = hdul[2].data['DMZ']

                # Initialize the model coordinates
                self.model_coordinates = Coordinates(
                    header['RPMIN'], header['RPMAX'], header['RTMAX'], header['NP'], header['NT'],
                    rp_grid=rp_grid_model, rt_grid=rt_grid_model, z_grid=z_grid_model
                )

            self.coeff_binning_model = 1

        hdul.close()

        # Compute the data mask
        self.data_mask = self.data_coordinates.get_mask_scale_cuts(cuts_config)

        # Read distortion matrix and initialize coordinate grids for the model
        if dmat_path is not None:
            self._read_dmat(dmat_path)

        # Check if we still need to initialize the model coordinates
        if self.model_coordinates is None:
            self.model_coordinates = self.data_coordinates
        if self.dist_model_coordinates is None:
            self.dist_model_coordinates = self.model_coordinates

        # Compute the model mask
        self.model_mask = self.dist_model_coordinates.get_mask_scale_cuts(cuts_config)

        # Compute data size
        self.data_size = len(self.masked_data_vec)
        self.full_data_size = len(self.data_vec)

        # Read the cuts we need to save for plotting
        self.r_min_cut = cuts_config.getfloat('r-min', 10.)
        self.r_max_cut = cuts_config.getfloat('r-max', 180.)

        self.mu_min_cut = cuts_config.getfloat('mu-min', -1.)
        self.mu_max_cut = cuts_config.getfloat('mu-max', +1.)

    def _check_if_blinding_matches(self, blinding_flag, dmat_path):
        if self._blinding_strat is None:
            if not (blinding_flag == 'none' or blinding_flag == 'None'):
                print(f'Warning: Data has no blinding, but distortion matrix at {dmat_path} '
                      f'has a blinding flag {blinding_flag}')
        else:
            if self._blinding_strat != blinding_flag:
                print(f'Warning: Data has a blinding flag {blinding_flag} that does not match '
                      f'the flag of the distortion matrix at {dmat_path}')

    def _read_dmat(self, dmat_path):
        print(f'Reading distortion matrix file {dmat_path}\n')
        hdul = fits.open(find_file(dmat_path))
        header = hdul[1].header

        if 'BLINDING' in header:
            self._check_if_blinding_matches(header['BLINDING'], dmat_path)

        if 'DM' in hdul[1].columns.names:
            self._distortion_mat = csr_matrix(hdul[1].data['DM'])
        elif 'DM_BLIND' in hdul[1].columns.names:
            self._distortion_mat = csr_matrix(hdul[1].data['DM_BLIND'])
        else:
            raise ValueError('No DM or DM_BLIND column found in distortion matrix file.')

        self.coeff_binning_model = header['COEFMOD']
        self.model_coordinates = Coordinates(
            header['RPMIN'], header['RPMAX'], header['RTMAX'],
            header['NP']*self.coeff_binning_model, header['NT']*self.coeff_binning_model,
            rp_grid=hdul[2].data['RP'], rt_grid=hdul[2].data['RT'], z_grid=hdul[2].data['Z']
        )

        self.dist_model_coordinates = Coordinates(
            header['RPMIN'], header['RPMAX'], header['RTMAX'], header['NP'], header['NT'])

        # Get the cosmological parameters
        if self.cosmo_params is None and "OMEGAM" in header:
            self.cosmo_params = {}
            self.cosmo_params['Omega_m'] = header['OMEGAM']
            self.cosmo_params['Omega_k'] = header.get('OMEGAK', 0.)
            self.cosmo_params['Omega_r'] = header.get('OMEGAR', 0.)
            self.cosmo_params['wl'] = header.get('WL', -1.)

        hdul.close()

    def _init_metal_tracers(self, metal_config):
        assert ('in tracer1' in metal_config) or ('in tracer2' in metal_config)

        # Read metal tracers
        metals_in_tracer1 = None
        metals_in_tracer2 = None
        if 'in tracer1' in metal_config:
            metals_in_tracer1 = metal_config.get('in tracer1').split()
        if 'in tracer2' in metal_config:
            metals_in_tracer2 = metal_config.get('in tracer2').split()

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

        return metals_in_tracer1, metals_in_tracer2, tracer_catalog

    def _init_metal_correlations(self, metal_config, metals_in_tracer1, metals_in_tracer2):
        metal_correlations = []
        if 'in tracer2' in metal_config:
            for metal in metals_in_tracer2:
                if not self._use_correlation(self.tracer1['name'], metal):
                    continue
                metal_correlations.append((self.tracer1['name'], metal))

        if 'in tracer1' in metal_config:
            for metal in metals_in_tracer1:
                if not self._use_correlation(metal, self.tracer2['name']):
                    continue
                metal_correlations.append((metal, self.tracer2['name']))

        if ('in tracer1' in metal_config) and ('in tracer2' in metal_config):
            for i, metal1 in enumerate(metals_in_tracer1):
                j0 = i if self.tracer1 == self.tracer2 else 0

                for metal2 in metals_in_tracer2[j0:]:
                    if not self._use_correlation(metal1, metal2):
                        continue
                    metal_correlations.append((metal1, metal2))

        return metal_correlations

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
        metals_in_tracer1, metals_in_tracer2, tracer_catalog = self._init_metal_tracers(metal_config)

        self.metal_mats = {}
        self.metal_coordinates = {}

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
                    tracers = (metal1, metal2)
                    name = metal1 + '_' + metal2

                    if 'RP_' + name not in metal_hdul[2].columns.names:
                        name = metal2 + '_' + metal1
                    self._read_metal_correlation(metal_hdul, tracers, name, dm_prefix)
                    metal_correlations.append(tracers)

        metal_hdul.close()

        return tracer_catalog, metal_correlations

    def _use_correlation(self, name1, name2):
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
        if 'SiII' in name1 and 'SiII' in name2 and not self.use_metal_autos:
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
        self.metal_coordinates[tracers] = Coordinates(
            metal_hdul[1].header['RPMIN'], metal_hdul[1].header['RPMAX'],
            metal_hdul[1].header['RTMAX'], metal_hdul[1].header['NP'], metal_hdul[1].header['NT'],
            rp_grid=metal_hdul[2].data['RP_' + name],
            rt_grid=metal_hdul[2].data['RT_' + name],
            z_grid=metal_hdul[2].data['Z_' + name]
        )

        metal_mat_size = self.metal_coordinates[tracers].rp_grid.size

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

    def create_monte_carlo(self, fiducial_model, scale=None, seed=None, forecast=False):
        """Create monte carlo mock of data using a fiducial model.

        Parameters
        ----------
        fiducial_model : 1D Array
            Fiducial model of the data
        scale : float, optional
            Scaling for the covariance, by default None.
        seed : int, optional
            Seed for the random number generator, by default None
        forecast : boolean, optional
            Forecast option. If true, we don't add noise to the mock,
            by default False

        Returns
        -------
        1D Array
            Monte Carlo mock of the data
        """
        if scale is None and self.corr_item.cov_rescale is None:
            scale = 1
        elif scale is None:
            scale = self.corr_item.cov_rescale

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
            if self.cholesky_masked_cov:
                masked_cov = self.cov_mat[:, self.data_mask]
                masked_cov = masked_cov[self.data_mask, :]
                self._cholesky = linalg.cholesky(self._scale * masked_cov)
            else:
                self._cholesky = linalg.cholesky(self._scale * self.cov_mat)

        # Create the mock
        if seed is not None:
            np.random.seed(seed)

        masked_fiducial = fiducial_model
        if fiducial_model.size != self.full_data_size:
            if fiducial_model.size != self.dist_model_coordinates.rp_grid.size:
                raise ValueError("Could not match fiducial model to data or model size.")
            mask = self.dist_model_coordinates.get_mask_to_other(self.data_coordinates)
            masked_fiducial = fiducial_model[mask]

        if forecast:
            self.mc_mock = masked_fiducial
        else:
            self.mc_mock = np.full(self.full_data_size, np.nan)
            if self.cholesky_masked_cov:
                ran_vec = np.random.randn(self.data_mask.sum())
                self.mc_mock[self.data_mask] = \
                    masked_fiducial[self.data_mask] + self._cholesky.dot(ran_vec)
            else:
                ran_vec = np.random.randn(self.full_data_size)
                self.mc_mock = masked_fiducial + self._cholesky.dot(ran_vec)

        self.masked_mc_mock = self.mc_mock[self.data_mask]

        return self.mc_mock
