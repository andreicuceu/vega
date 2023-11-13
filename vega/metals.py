import numpy as np
import copy
from astropy.io import fits
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from picca import constants as picca_constants

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
        # self.fast_metals_unsafe = corr_item.config['model'].getboolean('fast_metals_unsafe', False)
        # if self.fast_metals_unsafe:
        #     self.fast_metals = True

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

        # If in new metals mode, read the stacked delta files
        self.new_metals = self._corr_item.new_metals
        if self.new_metals:
            self.metal_matrix_config = self._corr_item.config['metal-matrix']

            self.cosmo = picca_constants.Cosmo(
                corr_item.cosmo_params['Omega_m'], corr_item.cosmo_params['Omega_k'],
                corr_item.cosmo_params['Omega_r'], corr_item.cosmo_params['wl'], blinding='none'
            )

            self.stack_table_1 = fits.open(self._corr_item.tracer1['delta-stack'])[1].data

            if self._corr_item.tracer1['name'] == self._corr_item.tracer2['name']:
                self.stack_table_2 = self.stack_table_1
            else:
                self.stack_table_2 = fits.open(self._corr_item.tracer2['delta-stack'])[1].data

        # Initialize metals
        self.Pk_metal = None
        self.Xi_metal = {}
        self.rp_metal_dmats = {}
        if self._corr_item.has_metals:
            for name1, name2 in self._corr_item.metal_correlations:
                # Get the tracer info
                tracer1 = self._corr_item.tracer_catalog[name1]
                tracer2 = self._corr_item.tracer_catalog[name2]

                if self.new_metals:
                    dmat, rp_grid, rt_grid, z_grid = self.compute_fast_metal_dmat(
                        name1, name2, self.main_tracers[0], self.main_tracers[1])
                    self.rp_metal_dmats[(name1, name2)] = dmat
                else:
                    # Read rp and rt for the metal correlation
                    rp_grid = data.metal_rp_grids[(name1, name2)]
                    rt_grid = data.metal_rt_grids[(name1, name2)]
                    z_grid = data.metal_z_grids[(name1, name2)]

                # Compute the corresponding r/mu coords
                r_grid = np.sqrt(rp_grid**2 + rt_grid**2)
                mask = r_grid != 0
                mu_grid = np.zeros(len(r_grid))
                mu_grid[mask] = rp_grid[mask] / r_grid[mask]

                # Initialize the coords grid dictionary
                coords_grid = {}
                coords_grid['r'] = r_grid
                coords_grid['mu'] = mu_grid
                coords_grid['z'] = z_grid

                # Get bin sizes
                if self._data is not None:
                    self._corr_item.config['metals']['bin_size_rp'] = \
                        str(self._corr_item.bin_size_rp_data)
                    self._corr_item.config['metals']['bin_size_rt'] = \
                        str(self._corr_item.bin_size_rt_data)

                # Initialize the metal correlation P(k)
                if self.Pk_metal is None:
                    self.Pk_metal = power_spectrum.PowerSpectrum(
                        self._corr_item.config['metals'], fiducial,
                        tracer1, tracer2, self._corr_item.name
                    )

                # assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                assert self._corr_item.config['metals'].getint('ell_max', ell_max) == ell_max, \
                       "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                # Assumes cross-corelations Lya x Metal and Metal x Lya are the same
                corr_hash = tuple(set((name1, name2)))
                self.Xi_metal[corr_hash] = corr_func.CorrelationFunction(
                                    self._corr_item.config['metals'], fiducial, coords_grid,
                                    scale_params, tracer1, tracer2, metal_corr=True)

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
        if self.new_metals:
            xi = (self.rp_metal_dmats[(name1, name2)] @ xi.reshape(50, 50)).flatten()
        else:
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
                    if self.new_metals:
                        xi = (self.rp_metal_dmats[(name1, name2)] @ xi.reshape(50, 50)).flatten()
                    else:
                        xi = self._data.metal_mats[(name1, name2)].dot(xi)

                    xi_metals += bias1 * bias2 * xi

                else:
                    xi_metals += bias1 * bias2 * self.compute_xi_metal_metal(
                        pk_lin, pars, name1, name2, component)

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
            if self.new_metals:
                xi = (self.rp_metal_dmats[(name1, name2)] @ xi.reshape(50, 50)).flatten()
            else:
                xi = self._data.metal_mats[(name1, name2)].dot(xi)

            if self.save_components:
                self.xi_distorted[component][(name1, name2)] = copy.deepcopy(xi)

            xi_metals += xi

        return xi_metals

    @staticmethod
    def rebin(vector, rebin_factor):
        """Rebin a vector by a factor of rebin_factor.

        Parameters
        ----------
        vector : 1D Array
            Vector to rebin
        rebin_factor : int
            Rebinning factor

        Returns
        -------
        1D Array
            Rebinned vector
        """
        size = vector.size
        return vector[:(size // rebin_factor) * rebin_factor].reshape(
            (size // rebin_factor), rebin_factor).mean(-1)

    def get_all_rp_pairs(self, wave1, wave2, abs_1, abs_2):
        z1 = wave1 / picca_constants.ABSORBER_IGM[abs_1] - 1.
        z2 = wave2 / picca_constants.ABSORBER_IGM[abs_2] - 1.
        r1 = self.cosmo.get_r_comov(z1)
        r2 = self.cosmo.get_r_comov(z2)

        # Get all pairs
        rp_pairs = (r1[:, None] - r2[None, :]).ravel()  # same sign as line 676 of cf.py (1-2)
        if 'discrete' not in self.main_tracer_types:
            rp_pairs = np.abs(rp_pairs)

        return rp_pairs, z1, z2

    def compute_fast_metal_dmat(self, true_abs_1, true_abs_2, assumed_abs_1, assumed_abs_2):
        wave1 = 10**self.stack_table_1["LOGLAM"]
        weights1 = self.stack_table_1["WEIGHT"]
        wave2 = 10**self.stack_table_2["LOGLAM"]
        weights2 = self.stack_table_2["WEIGHT"]

        rebin_factor = self.metal_matrix_config.getint('rebin_factor', None)
        if rebin_factor is not None:
            wave1 = self.rebin(wave1, rebin_factor)
            weights1 = self.rebin(weights1, rebin_factor)
            wave2 = self.rebin(wave2, rebin_factor)
            weights2 = self.rebin(weights2, rebin_factor)

        true_rp_pairs, true_z1, true_z2 = self.get_all_rp_pairs(
            wave1, wave2, true_abs_1, true_abs_2)

        assumed_rp_pairs, assumed_z1, assumed_z2 = self.get_all_rp_pairs(
            wave1, wave2, assumed_abs_1, assumed_abs_2)

        # Weights
        # alpha_in: in (1+z)^(alpha_in-1) is a scaling used to model how the metal contribution
        # evolves with redshift (by default alpha_in=1 so that this has no effect)
        # alpha_out: (1+z)^(alpha_out-1) is applied to the delta weights in io.read_deltas and
        # used for the correlation function. It also has to be applied here.
        # we have them for both forests (1 and 2)
        true_alpha_1 = self.metal_matrix_config.getfloat(f'alpha_{true_abs_1}')
        true_alpha_2 = self.metal_matrix_config.getfloat(f'alpha_{true_abs_2}')
        assumed_alpha_1 = self.metal_matrix_config.getfloat(f'alpha_{assumed_abs_1}', 2.9)
        assumed_alpha_2 = self.metal_matrix_config.getfloat(f'alpha_{assumed_abs_2}', 2.9)

        # so here we have to apply both scalings (in the original code :
        # alpha_in is applied in cf.calc_metal_dmat and alpha_out in io.read_deltas)
        scaling_1 = (1 + true_z1)**(true_alpha_1 + assumed_alpha_1 - 2)
        scaling_2 = (1 + true_z2)**(true_alpha_2 + assumed_alpha_2 - 2)
        weights = ((weights1 * scaling_1)[:, None] * (weights2 * scaling_2)[None, :]).ravel()

        # Distortion matrix grid
        rp_bin_edges = np.linspace(self._corr_item.rp_min_model, self._corr_item.rp_max_model,
                                   self._corr_item.num_bins_rp_model + 1)

        # I checked the orientation of the matrix
        dmat, _, _ = np.histogram2d(
            assumed_rp_pairs, true_rp_pairs, bins=(rp_bin_edges, rp_bin_edges), weights=weights)

        # Normalize (sum of weights should be one for each input rp,rt)
        sum_true_weight, _ = np.histogram(true_rp_pairs, bins=rp_bin_edges, weights=weights)
        dmat *= ((sum_true_weight > 0) / (sum_true_weight + (sum_true_weight == 0)))[None, :]

        # Mean assumed weights
        sum_assumed_weight, _ = np.histogram(assumed_rp_pairs, bins=rp_bin_edges, weights=weights)
        sum_assumed_weight_rp, _ = np.histogram(
            assumed_rp_pairs, bins=rp_bin_edges,
            weights=weights * (assumed_rp_pairs[None, :].ravel())
        )

        # Return the redshift of the actual absorber, which is the average of true_z1 and true_z2
        sum_weight_z, _ = np.histogram(
            assumed_rp_pairs, bins=rp_bin_edges,
            weights=weights * ((true_z1[:, None] + true_z2[None, :]) / 2.).ravel()
        )

        rp_eff = sum_assumed_weight_rp / (sum_assumed_weight + (sum_assumed_weight == 0))
        z_eff = sum_weight_z / (sum_assumed_weight + (sum_assumed_weight == 0))

        num_bins_total = self._corr_item.num_bins_rp_model * self._corr_item.num_bins_rt_model
        full_rp_eff = np.zeros(num_bins_total)
        full_rt_eff = np.zeros(num_bins_total)
        full_z_eff = np.zeros(num_bins_total)

        rp_indices = np.arange(self._corr_item.num_bins_rp_model)
        rt_bins = np.arange(
            self._corr_item.bin_size_rt_model / 2, self._corr_item.rt_max_model,
            self._corr_item.bin_size_rt_model
        )

        for j in range(self._corr_item.num_bins_rt_model):
            indices = j + self._corr_item.num_bins_rt_model * rp_indices

            full_rp_eff[indices] = rp_eff
            full_rt_eff[indices] = rt_bins[j]
            full_z_eff[indices] = z_eff

        return dmat, full_rp_eff, full_rt_eff, full_z_eff
