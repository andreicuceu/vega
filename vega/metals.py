import numpy as np

from . import power_spectrum
from . import correlation_func as corr_func
from . import utils


class Metals:
    """
    Class for computing metal correlations
    """
    def __init__(self, corr_item, fiducial, scale_params, PktoXi_obj, data=None):
        """Initialize metals

        Parameters
        ----------
        corr_item : CorrelationItem
            Item object with the component config
        fiducial : dict
            fiducial config
        scale_params : ScaleParameters
            ScaleParameters object
        PktoXi_obj : vega.PktoXi
            An instance of the transform object used to turn Pk into Xi
        data : Data, optional
            data object corresponding to the cf component, by default None
        """
        self._corr_item = corr_item
        self._data = data
        self.PktoXi = PktoXi_obj
        self.size = len(corr_item.r_mu_grid[0])
        ell_max = self._corr_item.config['model'].getint('ell_max', 6)

        self.fast_metals = corr_item.config['model'].getboolean('fast_metals', False)
        self.fast_metals_unsafe = corr_item.config['model'].getboolean('fast_metals_unsafe', False)
        if self.fast_metals_unsafe:
            self.fast_metals = True

        self.save_components = fiducial.get('save-components', False)

        if self.save_components and self.fast_metals:
            raise ValueError("Cannot save pk/cf components in fast_metals mode."
                             " Either turn fast_metals off, or turn off write_pk/write_cf.")

        self.pk = {'peak': {}, 'smooth': {}, 'full': {}}
        self.xi = {'peak': {}, 'smooth': {}, 'full': {}}
        self.xi_distorted = {'peak': {}, 'smooth': {}, 'full': {}}

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

                # assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                assert self._corr_item.config['metals'].getint('ell_max', 6) == ell_max, \
                       "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                self.Xi_metal[(name1, name2)] = corr_func.CorrelationFunction(
                                    self._corr_item.config['metals'], fiducial, coords_grid,
                                    scale_params, tracer1, tracer2, metal_corr=True)

            self._has_metal_mats = False
            if self._data is not None:
                self._has_metal_mats = self._data.metal_mats is not None

    def compute(self, pars, pk_lin, component):
        """Compute metal correlations for input isotropic P(k).

        Parameters
        ----------
        pars : dict
            Computation parameters
        pk_lin : 1D Array
            Linear power spectrum
        component : str
            Name of pk component, used as key for dictionary of saved
            components ('peak' or 'smooth' or 'full')

        Returns
        -------
        1D Array
            Model correlation function for the specified component
        """
        assert self._corr_item.has_metals

        xi_metals = np.zeros(self.size)
        for name1, name2, in self._corr_item.metal_correlations:
            bias1, _, bias2, __ = utils.bias_beta(pars, self._corr_item.tracer_catalog[name1],
                                                  self._corr_item.tracer_catalog[name2])

            # Fast metals mode
            if self.fast_metals:
                # Return saved distorted correlation if it exists
                if (name1, name2) in self.xi_distorted[component]:
                    xi_metals += bias1 * bias2 * self.xi_distorted[component][(name1, name2)]
                    continue
                # Return saved correlation if it exists
                elif (name1, name2) in self.xi[component]:
                    xi_metals += bias1 * bias2 * self.xi[component][(name1, name2)]
                    continue

            # No saved correlations, compute and save them
            # First try the fastest mode where we compute and save everything
            if self.fast_metals and self.fast_metals_unsafe:
                pk = self.Pk_metal[(name1, name2)].compute(pk_lin, pars, fast_metals=True)
                xi = self.Xi_metal[(name1, name2)].compute(pk, pk_lin, self.PktoXi, pars)

                # Apply the metal matrix
                if self._has_metal_mats:
                    xi = self._data.metal_mats[(name1, name2)].dot(xi)
                    self.xi_distorted[component][(name1, name2)] = xi.copy()
                else:
                    self.xi[component][(name1, name2)] = xi.copy()

                xi_metals += bias1 * bias2 * xi
                continue

            # Next try the fast safe mode where we only save Metal x Metal correlations
            elif self.fast_metals:
                # If it's a QSO/DLA cross, or LyaxMetal, we do the slow route because
                # beta_lya and beta_qso are usually sampled
                type1 = self._corr_item.tracer_catalog[name1]['type']
                type2 = self._corr_item.tracer_catalog[name2]['type']
                slow_condition = (type1 == 'discrete') or (type2 == 'discrete')
                slow_condition = slow_condition or (name1 == 'LYA') or (name2 == 'LYA')

                if not slow_condition:
                    pk = self.Pk_metal[(name1, name2)].compute(pk_lin, pars, fast_metals=True)
                    xi = self.Xi_metal[(name1, name2)].compute(pk, pk_lin, self.PktoXi, pars)

                    # Apply the metal matrix
                    if self._has_metal_mats:
                        xi = self._data.metal_mats[(name1, name2)].dot(xi)
                        self.xi_distorted[component][(name1, name2)] = xi.copy()
                    else:
                        self.xi[component][(name1, name2)] = xi.copy()

                    xi_metals += bias1 * bias2 * xi
                    continue

            # If not in fast metals mode, compute the usual way
            # Slow mode also allows the full save of components
            pk = self.Pk_metal[(name1, name2)].compute(pk_lin, pars)
            xi = self.Xi_metal[(name1, name2)].compute(pk, pk_lin, self.PktoXi, pars)

            # Save the components
            if self.save_components:
                assert not self.fast_metals
                self.pk[component][(name1, name2)] = pk.copy()
                self.xi[component][(name1, name2)] = xi.copy()

            # Apply the metal matrix
            if self._has_metal_mats:
                xi = self._data.metal_mats[(name1, name2)].dot(xi)
                if self.save_components:
                    assert not self.fast_metals
                    self.xi_distorted[component][(name1, name2)] = xi.copy()

            xi_metals += xi

        return xi_metals
