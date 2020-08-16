import numpy as np

# from . import new_pk, new_xi
from . import power_spectrum
from . import correlation_func as corr_func


class Model:
    """
    Class for computing Lyman-alpha forest correlation function models
    """

    def __init__(self, corr_item, data, fiducial):
        """Initialize the model computation

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        data : Data
            data object corresponding to the cf component
        fiducial : dict
            fiducial config
        """
        self.corr_item = corr_item

        # ! For now we need to import r, mu, z
        # ! until I add defaults
        assert data.r_grid is not None
        assert data.mu_grid is not None
        assert data.z_grid is not None
        self._coords_grid = {}
        self._coords_grid['r'] = data.r_grid
        self._coords_grid['mu'] = data.mu_grid
        self._coords_grid['z'] = data.z_grid

        self._data = data
        self._full_shape = fiducial.get('full-shape', False)
        self._smooth_scaling = fiducial.get('smooth-scaling', False)

        self.save_components = fiducial.get('save-components', False)
        if self.save_components:
            self.pk = {'peak': {}, 'smooth': {}}
            self.xi = {'peak': {}, 'smooth': {}}
            self.xi_distorted = {'peak': {}, 'smooth': {}}

        # Initialize main Power Spectrum object
        self.Pk_core = power_spectrum.PowerSpectrum(
            self.corr_item.config['model'], fiducial, self.corr_item.tracer1,
            self.corr_item.tracer2, self.corr_item.name)

        # Initialize main Correlation function object
        self.Xi_core = corr_func.CorrelationFunction(
            self.corr_item.config['model'], fiducial, self._coords_grid,
            self.corr_item.tracer1, self.corr_item.tracer2)

        # Initialize metals
        self.Pk_metal = {}
        self.Xi_metal = {}
        if self.corr_item.has_metals:
            for name1, name2 in self.corr_item.metal_correlations:
                # Get the tracer info
                tracer1 = self.corr_item.tracer_catalog[name1]
                tracer2 = self.corr_item.tracer_catalog[name2]

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
                                    self.corr_item.config['metals'], fiducial,
                                    tracer1, tracer2, self.corr_item.name)

                # Initialize the metal correlation Xi
                self.Xi_metal[(name1, name2)] = corr_func.CorrelationFunction(
                                    self.corr_item.config['metals'], fiducial,
                                    coords_grid, tracer1, tracer2)

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
        xi_model = self.Xi_core.compute(k, muk, pk_model, pars)

        # Save the components
        if self.save_components:
            self.pk[component]['core'] = pk_model.copy()
            self.xi[component]['core'] = xi_model.copy()

        # Compute metal correlation function
        if self.corr_item.has_metals:
            for name1, name2, in self.corr_item.metal_correlations:
                k, muk, pk_metal = self.Pk_metal[(name1, name2)].compute(
                                                        pk_lin, pars)
                xi_metal = self.Xi_metal[(name1, name2)].compute(
                                                        k, muk, pk_metal, pars)

                # Save the components
                if self.save_components:
                    self.pk[component][(name1, name2)] = pk_metal.copy()
                    self.xi[component][(name1, name2)] = xi_metal.copy()

                # Apply the metal matrix
                if self._data.metal_mats is not None:
                    xi_metal = self._data.metal_mats[(name1, name2)].dot(
                                                        xi_metal)
                    if self.save_components:
                        self.xi_distorted[component][(name1, name2)] = \
                            xi_metal.copy()

                # Add the metal component to the full xi
                xi_model += xi_metal

        # Apply the distortion matrix
        if self._data.distortion_mat is not None:
            xi_model = self._data.distortion_mat.dot(xi_model)

            if self.save_components:
                self.xi_distorted[component]['core'] = xi_model.copy()

        return xi_model

    def compute(self, pars, pk_full, pk_smooth):
        """Compute full correlation function model using the input parameters,
        a fiducial linear power spectrum and its smooth component

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
