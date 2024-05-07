from . import power_spectrum
from . import pktoxi
from . import correlation_func as corr_func
from . import metals


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
        self._model_pk = corr_item.model_pk

        assert corr_item.model_coordinates is not None

        self._data = data
        data_has_distortion = False
        if self._data is not None:
            data_has_distortion = self._data.has_distortion
        self._has_distortion_mat = corr_item.has_distortion and data_has_distortion

        self._corr_item.config['model']['bin_size_rp'] = str(corr_item.data_coordinates.rp_binsize)
        self._corr_item.config['model']['bin_size_rt'] = str(corr_item.data_coordinates.rt_binsize)

        self.save_components = fiducial.get('save-components', False)
        if self.save_components:
            self.pk = {'peak': {}, 'smooth': {}, 'full': {}}
            self.xi = {'peak': {}, 'smooth': {}, 'full': {}}
            self.xi_distorted = {'peak': {}, 'smooth': {}, 'full': {}}

        # Initialize Broadband
        bin_size_rp_data = corr_item.data_coordinates.rp_binsize
        bin_size_rp_model = corr_item.model_coordinates.rp_binsize
        self.bb_config = None
        if 'broadband' in self._corr_item.config:
            self.bb_config = self.init_broadband(
                self._corr_item.config['broadband'], self._corr_item.name, 
                bin_size_rp_data, bin_size_rp_model
            )

        # Initialize main Power Spectrum object
        self.Pk_core = power_spectrum.PowerSpectrum(
            self._corr_item.config['model'], fiducial, self._corr_item.tracer1,
            self._corr_item.tracer2, self._corr_item.name
        )

        # Initialize the Pk to Xi transform
        ell_max = self._corr_item.config['model'].getint('ell_max', 6)
        fht_lowring = self._corr_item.config['model'].getboolean('fht_lowring', True)
        fht_extrap = self._corr_item.config['model'].getboolean('fht_extrap', False)
        self.PktoXi = pktoxi.PktoXi(
            self.Pk_core.k_grid, self.Pk_core.muk_grid, ell_max,
            self._corr_item.old_fftlog, fht_lowring, fht_extrap
        )

        # Initialize main Correlation function object
        self.Xi_core = corr_func.CorrelationFunction(
            self._corr_item.config['model'], fiducial, corr_item.model_coordinates,
            scale_params, self._corr_item.tracer1, self._corr_item.tracer2, self.bb_config
        )

        # Initialize metals if needed
        self.metals = None
        if self._corr_item.has_metals:
            self.metals = metals.Metals(corr_item, fiducial, scale_params, self.PktoXi, data)

        self._instrumental_systematics_flag = corr_item.config['model'].getboolean(
            'desi-instrumental-systematics', False)

    @staticmethod
    def init_broadband(bb_input, cf_name, bin_size_rp_data, bin_size_rp_model):
        """Read the broadband config and initialize what we need.

        Parameters
        ----------
        bb_input : ConfigParser
            broadband section from the config file
        cf_name : string
            Name of corr item
        bin_size_rp_data : float
            Size of data r parallel bins
        bin_size_rp_model : float
            Size of model r parallel bins

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
                config['bin_size_rp'] = bin_size_rp_model
            else:
                config['bin_size_rp'] = bin_size_rp_data

            config['cf_name'] = cf_name
            bb_config.append(config)

        return bb_config

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
            components ('peak' or 'smooth' or 'full'), by default 'smooth'

        Returns
        -------
        1D Array
            Model correlation function for the specified component
        """
        # Compute core model correlation function
        pk_model = self.Pk_core.compute(pk_lin, pars)

        if self._model_pk:
            return self.PktoXi.compute_pk_ells(pk_model)

        # Protect against old caches that have not been cleaned
        self.PktoXi.cache_pars = None
        xi_model = self.Xi_core.compute(pk_model, pk_lin, self.PktoXi, pars)

        # Save the components
        if self.save_components:
            self.pk[component]['core'] = pk_model.copy()
            self.xi[component]['core'] = xi_model.copy()

        # Compute metal correlations
        if self._corr_item.has_metals:
            xi_model += self.metals.compute(pars, pk_lin, component)

            # Merge saved metal components into the member dictionaries
            if self.save_components:
                self.pk[component] = {**self.pk[component], **self.metals.pk[component]}
                self.xi[component] = {**self.xi[component], **self.metals.xi[component]}
                self.xi_distorted[component] = {**self.xi_distorted[component],
                                                **self.metals.xi_distorted[component]}

        # Add DESI instrumental systematics model
        if self._instrumental_systematics_flag and component != 'peak':
            xi_model += self.Xi_core.compute_desi_instrumental_systematics(
                pars, self._corr_item.data_coordinates.rp_binsize)

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

        #add dg g
        if self.

        return xi_model

    def compute(self, pars, pk_full, pk_smooth):
        """Compute correlation function model using the peak/smooth
        (wiggles/no-wiggles) decomposition.

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
