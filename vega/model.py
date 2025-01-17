from . import power_spectrum
from . import pktoxi
from . import correlation_func as corr_func
from . import metals
from . import broadband_poly


class Model:
    """
    Class for computing Lyman-alpha forest correlation function models.
    """

    def __init__(self, corr_item, fiducial, scale_params, data=None, xcf_obj=None):
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
        self.broadband = None
        if 'broadband' in self._corr_item.config:
            self.broadband = broadband_poly.BroadbandPolynomials(
                self._corr_item.config['broadband'], self._corr_item.name,
                corr_item.model_coordinates, corr_item.dist_model_coordinates
            )
    
        # Initialize main Power Spectrum object
        self.Pk_core = power_spectrum.PowerSpectrum(
            self._corr_item.config['model'], fiducial, self._corr_item.tracer1,
            self._corr_item.tracer2, self._corr_item.name
        )

        # Initialize the Pk to Xi transform
        self.PktoXi = pktoxi.PktoXi.init_from_Pk(self.Pk_core, self._corr_item.config['model'])

        # Initialize main Correlation function object
        self.Xi_core = corr_func.CorrelationFunction(
            self._corr_item.config['model'], fiducial, corr_item.model_coordinates,
            scale_params, self._corr_item.tracer1, self._corr_item.tracer2, xcf_obj, metal_corr=False
        )


        # Initialize metals if needed
        self.metals = None
        if self._corr_item.has_metals:
            self.metals = metals.Metals(corr_item, fiducial, scale_params, data)
            self.no_metal_decomp = corr_item.config['model'].getboolean('no-metal-decomp', True)

        self._instrumental_systematics_flag = corr_item.config['model'].getboolean(
            'desi-instrumental-systematics', False)
        
        self._gamma_model_flag = corr_item.config['model'].getboolean(
            'cont_dist_cross',False)
        
        self._delta_gamma_model_flag = corr_item.config['model'].getboolean(
            'cont_dist_auto',False)
        
            
        



    def _compute_model(self, pars, pk_lin, component='smooth', xi_metals=None):
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
        xi_metals : 1D Array, optional
            Metal correlation functions, by default None

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
            if self.no_metal_decomp and xi_metals is not None:
                xi_model += xi_metals
            elif not self.no_metal_decomp:
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
        if self.broadband is not None:
            xi_model *= self.broadband.compute(pars, 'pre-mul')
            xi_model += self.broadband.compute(pars, 'pre-add')

        # Apply delta gamma correction for auto
        if self._delta_gamma_model_flag:
            xi_model += self.Xi_core.compute_delta_gamma_model(pars)

        # Apply gamma correction for cross
        if self._gamma_model_flag:
            xi_model += self.Xi_core.compute_gamma_model(pars)
            
        # Apply the distortion matrix
        if self._has_distortion_mat:
            xi_model = self._data.distortion_mat.dot(xi_model)

        # Apply post distortion broadband
        if self.broadband is not None:
            xi_model *= self.broadband.compute(pars, 'post-mul')
            xi_model += self.broadband.compute(pars, 'post-add')

        # Save final xi
        if self.save_components:
            self.xi_distorted[component]['core'] = xi_model.copy()

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
        xi_metals = None
        if self._corr_item.has_metals and self.no_metal_decomp:
            xi_metals = self.metals.compute(pars, pk_full, 'full')

        xi_smooth = self._compute_model(pars, pk_smooth, 'smooth', xi_metals=xi_metals)

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
