import numpy as np
import copy
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from . import power_spectrum
from . import correlation_func as corr_func
from . import utils


class Metals:
    """
    Class for computing metal correlations
    """
    cache_pk = LRUCache(128)
    cache_xi = LRUCache(128)

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

        # Build a mask for the cross-correlations with the main tracers (Lya, QSO)
        self.main_tracers = [corr_item.tracer1['name'], corr_item.tracer2['name']]
        self.main_tracer_types = [corr_item.tracer1['type'], corr_item.tracer2['type']]
        self.main_cross_mask = [tracer1 in self.main_tracers or tracer2 in self.main_tracers
                                for (tracer1, tracer2) in corr_item.metal_correlations]

        # Initialize metals
        self.Pk_metal = None
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
                if self.Pk_metal is None:
                    self.Pk_metal = power_spectrum.PowerSpectrum(self._corr_item.config['metals'],
                                                                 fiducial, tracer1, tracer2,
                                                                 self._corr_item.name)

                # assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                assert self._corr_item.config['metals'].getint('ell_max', ell_max) == ell_max, \
                       "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                # Assumes cross-corelations Lya x Metal and Metal x Lya are the same
                corr_hash = tuple(set((name1, name2)))
                self.Xi_metal[corr_hash] = corr_func.CorrelationFunction(
                                    self._corr_item.config['metals'], fiducial, coords_grid,
                                    scale_params, tracer1, tracer2, metal_corr=True)

            self._has_metal_mats = False
            if self._data is not None:
                self._has_metal_mats = self._data.metal_mats is not None

    @cached(cache=cache_pk, key=lambda self, call_pars, *cache_pars: hashkey(*cache_pars))
    def compute_pk(self, call_pars, *cache_pars):
        return self.Pk_metal.compute(*call_pars, fast_metals=True)

    @cached(cache=cache_xi,
            key=lambda self, pk_lin, pars, name1, name2, component: hashkey(name1, name2, component))
    def compute_xi_metal_metal(self, pk_lin, pars, name1, name2, component):
        pk = self.Pk_metal.compute(pk_lin, pars, fast_metals=True)

        corr_hash = tuple(set((name1, name2)))
        self.PktoXi.cache_pars = None
        xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi, pars)

        # Apply the metal matrix
        if self._has_metal_mats:
            xi = self._data.metal_mats[(name1, name2)].dot(xi)

        return xi

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
            bias1, beta1, bias2, beta2 = utils.bias_beta(pars, name1, name2)

            # Fast metals mode
            # if self.fast_metals:
            #     # Return saved distorted correlation if it exists
            #     if (name1, name2) in self.xi_distorted[component]:
            #         xi_metals += bias1 * bias2 * self.xi_distorted[component][(name1, name2)]
            #         continue
            #     # Return saved correlation if it exists
            #     elif (name1, name2) in self.xi[component]:
            #         xi_metals += bias1 * bias2 * self.xi[component][(name1, name2)]
            #         continue

            # No saved correlations, compute and save them
            # First try the fastest mode where we compute and save everything
            # if self.fast_metals and self.fast_metals_unsafe:
            #     pk = self.Pk_metal[(name1, name2)].compute(pk_lin, pars, fast_metals=True)
            #     xi = self.Xi_metal[(name1, name2)].compute(pk, pk_lin, self.PktoXi, pars)

            #     # Apply the metal matrix
            #     if self._has_metal_mats:
            #         xi = self._data.metal_mats[(name1, name2)].dot(xi)
            #         self.xi_distorted[component][(name1, name2)] = xi.copy()
            #     else:
            #         self.xi[component][(name1, name2)] = xi.copy()

            #     xi_metals += bias1 * bias2 * xi
            #     continue

            # Next try the fast safe mode where we only save Metal x Metal correlations
            # elif self.fast_metals:
            #     # If it's a QSO/DLA cross, or LyaxMetal, we do the slow route because
            #     # beta_lya and beta_qso are usually sampled
            #     type1 = self._corr_item.tracer_catalog[name1]['type']
            #     type2 = self._corr_item.tracer_catalog[name2]['type']
            #     slow_condition = (type1 == 'discrete') or (type2 == 'discrete')
            #     slow_condition = slow_condition or (name1 == 'LYA') or (name2 == 'LYA')

            #     if not slow_condition:
            #         pk = self.Pk_metal[(name1, name2)].compute(pk_lin, pars, fast_metals=True)
            #         xi = self.Xi_metal[(name1, name2)].compute(pk, pk_lin, self.PktoXi, pars)

            #         # Apply the metal matrix
            #         if self._has_metal_mats:
            #             xi = self._data.metal_mats[(name1, name2)].dot(xi)
            #             self.xi_distorted[component][(name1, name2)] = xi.copy()
            #         else:
            #             self.xi[component][(name1, name2)] = xi.copy()

            #         xi_metals += bias1 * bias2 * xi
            #         continue

            self.Pk_metal.tracer1_name = name1
            self.Pk_metal.tracer2_name = name2
            if self.fast_metals and component != 'full':
                # If its a metal x Lya or metal x QSO correlation we can only cache the Pk
                if name1 in self.main_tracers or name2 in self.main_tracers:
                    # Get beta of main tracer
                    beta_main = beta1 if name1 in self.main_tracers else beta2

                    cache_pars = None
                    # We need to separate Lya and QSO correlations
                    # because they have different parameters
                    if 'discrete' in self.main_tracer_types:
                        for par in pars.keys():
                            if 'sigma_velo_disp' in par:
                                cache_pars = (beta_main, pars[par], component)
                                break

                    if cache_pars is None:
                        cache_pars = (beta_main, component)

                    pk = self.compute_pk((pk_lin, pars), *cache_pars)

                    corr_hash = tuple(set((name1, name2)))
                    self.PktoXi.cache_pars = cache_pars
                    xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi, pars)
                    self.PktoXi.cache_pars = None

                    # Apply the metal matrix
                    if self._has_metal_mats:
                        xi = self._data.metal_mats[(name1, name2)].dot(xi)

                    xi_metals += bias1 * bias2 * xi

                else:
                    xi_metals += bias1 * bias2 * self.compute_xi_metal_metal(pk_lin, pars, name1,
                                                                             name2, component)

                continue

            # If not in fast metals mode, compute the usual way
            # Slow mode also allows the full save of components
            pk = self.Pk_metal.compute(pk_lin, pars)
            if self.save_components:
                self.pk[component][(name1, name2)] = copy.deepcopy(pk)

            corr_hash = tuple(set((name1, name2)))
            xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi, pars)

            # Save the components
            if self.save_components:
                # self.pk[component][(name1, name2)] = copy.deepcopy(pk)
                self.xi[component][(name1, name2)] = copy.deepcopy(xi)

            # Apply the metal matrix
            if self._has_metal_mats:
                xi = self._data.metal_mats[(name1, name2)].dot(xi)
                if self.save_components:
                    self.xi_distorted[component][(name1, name2)] = copy.deepcopy(xi)

            xi_metals += xi

        return xi_metals
