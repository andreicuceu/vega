import numpy as np

# from . import new_pk, new_xi
from . import power_spectrum
from . import correlation_func as corr_func


class Model:
    """
    Class for computing Lyman-alpha forest correlation function models.
    """

    def __init__(self, corr_item, fiducial, data=None):
        """

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        fiducial : dict
            fiducial config
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

        self._data = data
        self._full_shape = fiducial.get('full-shape', False)
        self._smooth_scaling = fiducial.get('smooth-scaling', False)
        data_distortion = False
        if self._data is not None:
            data_distortion = self._data.has_distortion()
        self._has_distortion_mat = corr_item.has_distortion and data_distortion

        self.save_components = fiducial.get('save-components', False)
        if self.save_components:
            self.pk = {'peak': {}, 'smooth': {}}
            self.xi = {'peak': {}, 'smooth': {}}
            self.xi_distorted = {'peak': {}, 'smooth': {}}

        # Initialize Broadband
        self.bb_config = None
        if 'broadband' in self._corr_item.config:
            self.bb_config = self.init_broadband(
                    self._corr_item.config['broadband'], self._corr_item.name,
                    self._corr_item.bin_size_rp,
                    self._corr_item.coeff_binning_model)

        # Initialize main Power Spectrum object
        self.Pk_core = power_spectrum.PowerSpectrum(
            self._corr_item.config['model'], fiducial, self._corr_item.tracer1,
            self._corr_item.tracer2, self._corr_item.name)

        # Initialize main Correlation function object
        self.Xi_core = corr_func.CorrelationFunction(
            self._corr_item.config['model'], fiducial, self._coords_grid,
            self._corr_item.tracer1, self._corr_item.tracer2, self.bb_config)

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
                w = r_grid == 0
                r_grid[w] = 1e-6
                mu_grid = rp_grid / r_grid

                # Initialize the coords grid dictionary
                coords_grid = {}
                coords_grid['r'] = r_grid
                coords_grid['mu'] = mu_grid
                coords_grid['z'] = data.metal_z_grids[(name1, name2)]

                # Initialize the metal correlation P(k)
                self.Pk_metal[(name1, name2)] = power_spectrum.PowerSpectrum(
                                    self._corr_item.config['metals'], fiducial,
                                    tracer1, tracer2, self._corr_item.name)

                # Initialize the metal correlation Xi
                self.Xi_metal[(name1, name2)] = corr_func.CorrelationFunction(
                                    self._corr_item.config['metals'], fiducial,
                                    coords_grid, tracer1, tracer2)

            self._has_metal_mats = False
            if self._data is not None:
                self._has_metal_mats = self._data.metal_mats is not None

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
        k, muk, pk_model = self.Pk_core.compute(pk_lin, pars)
        xi_model = self.Xi_core.compute(k, muk, pk_model, pk_lin, pars)

        # Save the components
        if self.save_components:
            self.pk[component]['core'] = pk_model.copy()
            self.xi[component]['core'] = xi_model.copy()

        # Compute metal correlation function
        if self._corr_item.has_metals:
            for name1, name2, in self._corr_item.metal_correlations:
                k, muk, pk_metal = self.Pk_metal[(name1, name2)].compute(
                                                        pk_lin, pars)
                xi_metal = self.Xi_metal[(name1, name2)].compute(
                                                        k, muk, pk_metal,
                                                        pk_lin, pars)

                # Save the components
                if self.save_components:
                    self.pk[component][(name1, name2)] = pk_metal.copy()
                    self.xi[component][(name1, name2)] = xi_metal.copy()

                # Apply the metal matrix
                if self._has_metal_mats:
                    xi_metal = self._data.metal_mats[(name1, name2)].dot(
                                                        xi_metal)
                    if self.save_components:
                        self.xi_distorted[component][(name1, name2)] = \
                            xi_metal.copy()

                # Add the metal component to the full xi
                xi_model += xi_metal

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
        pars['smooth_scaling'] = self._smooth_scaling
        pars['full-shape'] = self._full_shape

        pars['peak'] = True
        xi_peak = self._compute_model(pars, pk_full - pk_smooth, 'peak')

        pars['peak'] = False
        xi_smooth = self._compute_model(pars, pk_smooth, 'smooth')

        xi_full = pars['bao_amp'] * xi_peak + xi_smooth
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
