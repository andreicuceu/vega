import numpy as np
from functools import lru_cache

from . import power_spectrum
from . import pktoxi
from . import correlation_func as corr_func
from . import utils


class Model:
    """
    Class for computing Lyman-alpha forest correlation function models.
    """

    def __init__(self, corr_item, fiducial, scale_params, data=None):
        """

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        fiducial : dict
            fiducial config
        scale_params : ScaleParameters
            ScaleParameters object
        data : Data, optional
            data object corresponding to the cf component, by default None
        """
        self._corr_item = corr_item

        assert corr_item.r_mu_grid is not None
        assert corr_item.z_grid is not None
        self._coords_grid = {}
        self._coords_grid['r'] = corr_item.r_mu_grid[0]
        self._coords_grid['mu'] = corr_item.r_mu_grid[1]
        self._coords_grid['z'] = corr_item.z_grid
        self.size = len(corr_item.r_mu_grid[0])
        self.fast_metals_safe = corr_item.config['model'].getboolean('fast_metals_safe', False)
        self.fast_metals_unsafe = corr_item.config['model'].getboolean('fast_metals_unsafe', False)

        self._data = data
        data_distortion = False
        if self._data is not None:
            data_distortion = self._data.has_distortion()
            self._corr_item.config['model']['bin_size_rp'] = str(self._data.bin_size_rp)
            self._corr_item.config['model']['bin_size_rt'] = str(self._data.bin_size_rt)
        self._has_distortion_mat = corr_item.has_distortion and data_distortion

        self.save_components = fiducial.get('save-components', False)
        if self.save_components:
            self.pk = {'peak': {}, 'smooth': {}, 'full': {}}
            self.xi = {'peak': {}, 'smooth': {}, 'full': {}}
            self.xi_distorted = {'peak': {}, 'smooth': {}, 'full': {}}

        # Initialize Broadband
        self.bb_config = None
        if 'broadband' in self._corr_item.config:
            self.bb_config = self.init_broadband(self._corr_item.config['broadband'],
                                                 self._corr_item.name, self._corr_item.bin_size_rp,
                                                 self._corr_item.coeff_binning_model)

        # Initialize main Power Spectrum object
        self.Pk_core = power_spectrum.PowerSpectrum(self._corr_item.config['model'],
                                                    fiducial, self._corr_item.tracer1,
                                                    self._corr_item.tracer2, self._corr_item.name)

        # Initialize the Pk to Xi transform
        ell_max = self._corr_item.config['model'].getint('ell_max', 6)
        self.PktoXi = pktoxi.PktoXi(self.Pk_core.k_grid, self.Pk_core.muk_grid, ell_max,
                                    self._corr_item.old_fftlog)

        # Initialize main Correlation function object
        self.Xi_core = corr_func.CorrelationFunction(self._corr_item.config['model'], fiducial,
                                                     self._coords_grid, scale_params,
                                                     self._corr_item.tracer1,
                                                     self._corr_item.tracer2, self.bb_config)

        # Initialize metals
        self.Pk_metal = {}
        self.Xi_metal = {}
        if self._corr_item.has_metals:
            for name1, name2 in self._corr_item.metal_correlations:
                # Get the tracer info
                tracer1 = self._corr_item.tracer_catalog[name1]
                tracer2 = self._corr_item.tracer_catalog[name2]

                # Read rp and rt for the metal correlation
                rp_grid = data.metal_rp_grids[(name1, name2)]
                rt_grid = data.metal_rt_grids[(name1, name2)]

                # Compute the corresponding r/mu coords
                r_grid = np.sqrt(rp_grid**2 + rt_grid**2)
                mask = r_grid != 0
                mu_grid = np.zeros(len(r_grid))
                mu_grid[mask] = rp_grid[mask] / r_grid[mask]

                # Initialize the coords grid dictionary
                coords_grid = {}
                coords_grid['r'] = r_grid
                coords_grid['mu'] = mu_grid
                coords_grid['z'] = data.metal_z_grids[(name1, name2)]

                # Get bin sizes
                if self._data is not None:
                    self._corr_item.config['metals']['bin_size_rp'] = str(self._data.bin_size_rp)
                    self._corr_item.config['metals']['bin_size_rt'] = str(self._data.bin_size_rt)

                # Initialize the metal correlation P(k)
                self.Pk_metal[(name1, name2)] = power_spectrum.PowerSpectrum(
                                    self._corr_item.config['metals'], fiducial,
                                    tracer1, tracer2, self._corr_item.name)

                assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                assert self._corr_item.config['metals'].getint('ell_max', 6) == ell_max, \
                       "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                self.Xi_metal[(name1, name2)] = corr_func.CorrelationFunction(
                                    self._corr_item.config['metals'], fiducial, coords_grid,
                                    scale_params, tracer1, tracer2, metal_corr=True)

            self._has_metal_mats = False
            if self._data is not None:
                self._has_metal_mats = self._data.metal_mats is not None

    def _compute_metals_approx(self, tracer1, tracer2, pars, pk_lin):
        bias_beta = utils.bias_beta(pars, tracer1, tracer2)
        bias1, beta1, bias2, beta2 = bias_beta

        self._temp_pk_lin = pk_lin
        self._temp_pars = pars
        pk, xi = self._approx_metals(tracer1['name'], tracer2['name'])

        return bias1 * bias2 * pk, bias1 * bias2 * xi

    @lru_cache
    def _approx_metals(self, name1, name2):
        pk = self.Pk_metal[(name1, name2)].compute(self._temp_pk_lin, self._temp_pars,
                                                   fast_metals=True)
        xi = self.Xi_metal[(name1, name2)].compute(pk, self._temp_pk_lin, self.PktoXi,
                                                   self._temp_pars)

        return pk, xi

    def _compute_metals_fast(self, tracer1, tracer2, pars, pk_lin):
        bias_beta = utils.bias_beta(pars, tracer1, tracer2)
        bias1, beta1, bias2, beta2 = bias_beta

        # If it's a QSO/DLA cross, we may have to model velocity dispersion
        # In this case we do the slow compute
        if tracer1['type'] == 'discrete' or tracer2['type'] == 'discrete':
            if 'velocity dispersion' in self._corr_item.config['metals']:
                return self._compute_metals_slow(tracer1, tracer2, pars, pk_lin)

        self._temp_pk_lin = pk_lin
        self._temp_pars = pars
        pk, xi = self._fast_metals(tracer1['name'], tracer2['name'], beta1, beta2)

        return bias1 * bias2 * pk, bias1 * bias2 * xi

    @lru_cache
    def _fast_metals(self, name1, name2, beta1, beta2):
        pk = self.Pk_metal[(name1, name2)].compute(self._temp_pk_lin, self._temp_pars,
                                                   fast_metals=True)
        xi = self.Xi_metal[(name1, name2)].compute(pk, self._temp_pk_lin, self.PktoXi,
                                                   self._temp_pars)

        return pk, xi

    def _compute_metals_slow(self, tracer1, tracer2, pars, pk_lin):
        pk = self.Pk_metal[(tracer1['name'], tracer2['name'])].compute(pk_lin, pars)
        xi = self.Xi_metal[(tracer1['name'], tracer2['name'])].compute(pk, pk_lin,
                                                                       self.PktoXi, pars)

        return pk, xi

    def compute_metals(self, pars, pk_lin, component):
        assert self._corr_item.has_metals

        xi_metals = np.zeros(self.size)
        for name1, name2, in self._corr_item.metal_correlations:
            if self.fast_metals_unsafe:
                pk, xi = self._compute_metals_approx(self._corr_item.tracer_catalog[name1],
                                                     self._corr_item.tracer_catalog[name2],
                                                     pars, pk_lin)
            elif self.fast_metals_safe:
                pk, xi = self._compute_metals_fast(self._corr_item.tracer_catalog[name1],
                                                   self._corr_item.tracer_catalog[name2],
                                                   pars, pk_lin)
            else:
                pk, xi = self._compute_metals_slow(self._corr_item.tracer_catalog[name1],
                                                   self._corr_item.tracer_catalog[name2],
                                                   pars, pk_lin)

            # Save the components
            if self.save_components:
                self.pk[component][(name1, name2)] = pk.copy()
                self.xi[component][(name1, name2)] = xi.copy()

            # Apply the metal matrix
            if self._has_metal_mats:
                xi = self._data.metal_mats[(name1, name2)].dot(xi)
                if self.save_components:
                    self.xi_distorted[component][(name1, name2)] = xi.copy()

            xi_metals += xi

        return xi_metals

    def _compute_model(self, pars, pk_lin, component='smooth'):
        """Compute a model correlation function given the input pars
        and a fiducial linear power spectrum.

        This is used internally for computing the peak and smooth
        components separately.

        Parameters
        ----------
        pars : dict
            Computation parameters
        pk_lin : 1D Array
            Linear power spectrum
        component : str, optional
            Name of pk component, used as key for dictionary of saved
            components ('peak' or 'smooth'), by default 'smooth'

        Returns
        -------
        1D Array
            Model correlation function for the specified component
        """
        # Compute core model correlation function
        pk_model = self.Pk_core.compute(pk_lin, pars)
        xi_model = self.Xi_core.compute(pk_model, pk_lin, self.PktoXi, pars)

        # Save the components
        if self.save_components:
            self.pk[component]['core'] = pk_model.copy()
            self.xi[component]['core'] = xi_model.copy()

        # Compute metal correlations
        if self._corr_item.has_metals:
            xi_model += self.compute_metals(pars, pk_lin, component)

        # Apply pre distortion broadband
        if self.bb_config is not None:
            assert self.Xi_core.has_bb
            xi_model *= self.Xi_core.compute_broadband(pars, 'pre-mul')
            xi_model += self.Xi_core.compute_broadband(pars, 'pre-add')

        # Apply the distortion matrix
        if self._has_distortion_mat:
            xi_model = self._data.distortion_mat.dot(xi_model)

        # Apply post distortion broadband
        if self.bb_config is not None:
            assert self.Xi_core.has_bb
            xi_model *= self.Xi_core.compute_broadband(pars, 'post-mul')
            xi_model += self.Xi_core.compute_broadband(pars, 'post-add')

        # Save final xi
        if self.save_components:
            self.xi_distorted[component]['core'] = xi_model.copy()

        return xi_model

    def compute(self, pars, pk_full, pk_smooth):
        """Compute full correlation function model using the input parameters,
        a fiducial linear power spectrum and its smooth component.

        Parameters
        ----------
        pars : dict
            Computation parameters
        pk_full : 1D Array
            Full fiducial linear power spectrum
        pk_smooth : 1D Array
            Smooth component of the fiducial linear power spectrum

        Returns
        -------
        1D Array
            Full correlation function
        """
        pars['peak'] = True
        xi_peak = self._compute_model(pars, pk_full - pk_smooth, 'peak')

        pars['peak'] = False
        xi_smooth = self._compute_model(pars, pk_smooth, 'smooth')

        xi_full = pars['bao_amp'] * xi_peak + xi_smooth
        return xi_full

    def compute_direct(self, pars, pk_full):
        """Compute full correlation function model directly from the full
        power spectrum.

        Parameters
        ----------
        pars : dict
            Computation parameters
        pk_full : 1D Array
            Full fiducial linear power spectrum

        Returns
        -------
        1D Array
            Full correlation function
        """
        pars['peak'] = False
        xi_full = self._compute_model(pars, pk_full, 'full')

        return xi_full

    @staticmethod
    def init_broadband(bb_input, cf_name, bin_size_rp, coeff_binning_model):
        """Read the broadband config and initialize what we need.

        Parameters
        ----------
        bb_input : ConfigParser
            broadband section from the config file
        cf_name : string
            Name of corr item
        bin_size_rp : int
            Size of r parallel bins
        coeff_binning_model : float
            Ratio of distorted coordinate grid bin size to undistorted bin size

        Returns
        -------
        list
            list with configs of broadband terms
        """
        bb_config = []
        for item, value in bb_input.items():
            value = value.split()
            config = {}
            # Check if it's additive or multiplicative
            assert value[0] == 'add' or value[0] == 'mul'
            config['type'] = value[0]

            # Check if it's pre distortion or post distortion
            assert value[1] == 'pre' or value[1] == 'post'
            config['pre'] = value[1]

            # Check if it's over rp/rt or r/mu
            assert value[2] == 'rp,rt' or value[2] == 'r,mu'
            config['rp_rt'] = value[2]

            # Check if it's normal or sky
            if len(value) == 6:
                config['func'] = value[5]
            else:
                config['func'] = 'broadband'

            # Get the coordinate configs
            r_min, r_max, dr = value[3].split(':')
            mu_min, mu_max, dmu = value[4].split(':')
            config['r_config'] = (int(r_min), int(r_max), int(dr))
            config['mu_config'] = (int(mu_min), int(mu_max), int(dmu))
            if config['pre'] == 'pre':
                config['bin_size_rp'] = bin_size_rp
            else:
                config['bin_size_rp'] = bin_size_rp / coeff_binning_model

            config['cf_name'] = cf_name
            bb_config.append(config)

        return bb_config
