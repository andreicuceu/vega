import numpy as np
import copy
from astropy.io import fits
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from picca import constants as picca_constants

from . import power_spectrum
from . import correlation_func as corr_func
from . import coordinates
from . import utils


class Metals:
    """
    Class for computing metal correlations
    """
    cache_pk = LRUCache(128)
    cache_xi = LRUCache(128)

    metal_growth_rate = None
    par_sigma_smooth = None
    per_sigma_smooth = None
    fast_metals = False

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
        self.size = corr_item.model_coordinates.rp_grid.size
        ell_max = self._corr_item.config['model'].getint('ell_max', 6)
        self._coordinates = corr_item.model_coordinates
        self.fast_metals = corr_item.config['model'].getboolean('fast_metals', False)

        # Read the growth rate and sigma_smooth from the fiducial config
        if 'metal-growth_rate' in fiducial:
            self.metal_growth_rate = fiducial['metal-growth_rate']
        if 'par_sigma_smooth' in fiducial:
            self.par_sigma_smooth = fiducial['par_sigma_smooth']
        if 'per_sigma_smooth' in fiducial:
            self.per_sigma_smooth = fiducial['per_sigma_smooth']

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
            self.rp_nbins = self._coordinates.rp_nbins
            self.rt_nbins = self._coordinates.rt_nbins

            self.cosmo = picca_constants.Cosmo(
                Om=corr_item.cosmo_params['Omega_m'], Ok=corr_item.cosmo_params['Omega_k'],
                Or=corr_item.cosmo_params['Omega_r'], wl=corr_item.cosmo_params['wl'],
                blinding='none'
            )

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
                    dmat, rp_grid, rt_grid, z_grid = self.compute_fast_metal_dmat(name1, name2)

                    self.rp_metal_dmats[(name1, name2)] = dmat
                    metal_coordinates = coordinates.Coordinates.init_from_grids(
                        self._coordinates, rp_grid, rt_grid, z_grid)
                else:
                    # Read rp and rt for the metal correlation
                    metal_coordinates = data.metal_coordinates[(name1, name2)]

                # Get bin sizes
                if self._data is not None:
                    self._corr_item.config['metals']['bin_size_rp'] = \
                        str(self._corr_item.data_coordinates.rp_binsize)
                    self._corr_item.config['metals']['bin_size_rt'] = \
                        str(self._corr_item.data_coordinates.rt_binsize)

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
                    self._corr_item.config['metals'], fiducial, metal_coordinates,
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
            xi = (self.rp_metal_dmats[(name1, name2)]
                  @ xi.reshape(self.rp_nbins, self.rt_nbins)).flatten()
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
        local_pars = copy.deepcopy(pars)

        # TODO Check growth rate and sigma_smooth exist. They should be in the fiducial config.
        if self.fast_metals:
            if 'growth_rate' in local_pars and self.metal_growth_rate is not None:
                local_pars['growth_rate'] = self.metal_growth_rate
            if 'sigma_smooth_par' in local_pars and self.par_sigma_smooth is not None:
                local_pars['sigma_smooth_par'] = self.par_sigma_smooth
            if 'sigma_smooth_per' in local_pars and self.per_sigma_smooth is not None:
                local_pars['sigma_smooth_per'] = self.per_sigma_smooth

        xi_metals = np.zeros(self.size)
        for name1, name2, in self._corr_item.metal_correlations:
            bias1, beta1, bias2, beta2 = utils.bias_beta(local_pars, name1, name2)

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
                        for par in local_pars.keys():
                            if 'sigma_velo_disp' in par:
                                cache_pars = (beta_main, local_pars[par], component)
                                break

                    if cache_pars is None:
                        cache_pars = (beta_main, component)

                    pk = self.compute_pk((pk_lin, local_pars), *cache_pars)

                    corr_hash = tuple(set((name1, name2)))
                    self.PktoXi.cache_pars = cache_pars
                    xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi, local_pars)
                    self.PktoXi.cache_pars = None

                    # Apply the metal matrix
                    if self.new_metals:
                        xi = (self.rp_metal_dmats[(name1, name2)]
                              @ xi.reshape(self.rp_nbins, self.rt_nbins)).flatten()
                    else:
                        xi = self._data.metal_mats[(name1, name2)].dot(xi)

                    xi_metals += bias1 * bias2 * xi

                else:
                    xi_metals += bias1 * bias2 * self.compute_xi_metal_metal(
                        pk_lin, local_pars, name1, name2, component)

                continue

            # If not in fast metals mode, compute the usual way
            # Slow mode also allows the full save of components
            pk = self.Pk_metal.compute(pk_lin, local_pars)
            if self.save_components:
                self.pk[component][(name1, name2)] = copy.deepcopy(pk)

            corr_hash = tuple(set((name1, name2)))
            xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi, local_pars)

            # Save the components
            if self.save_components:
                # self.pk[component][(name1, name2)] = copy.deepcopy(pk)
                self.xi[component][(name1, name2)] = copy.deepcopy(xi)

            # Apply the metal matrix
            if self.new_metals:
                xi = (self.rp_metal_dmats[(name1, name2)]
                      @ xi.reshape(self.rp_nbins, self.rt_nbins)).flatten()
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

    def get_forest_weights(self, main_tracer):
        assert main_tracer['type'] == 'continuous'
        with fits.open(main_tracer['weights-path']) as hdul:
            stack_table = hdul[1].data

        wave = 10**stack_table["LOGLAM"]
        weights = stack_table["WEIGHT"]

        rebin_factor = self.metal_matrix_config.getint('rebin_factor', None)
        if rebin_factor is not None:
            wave = self.rebin(wave, rebin_factor)
            weights = self.rebin(weights, rebin_factor)

        return wave, weights

    def get_qso_weights(self, tracer):
        assert tracer['type'] == 'discrete'
        with fits.open(tracer['weights-path']) as hdul:
            z_qso_cat = hdul[1].data['Z']

        z_ref = self.metal_matrix_config.getfloat('z_ref_objects', 2.25)
        z_evol = self.metal_matrix_config.getfloat('z_evol_objects', 1.44)
        qso_z_bins = self.metal_matrix_config.getint('z_bins_objects', 1000)
        weights_qso_cat = ((1. + z_qso_cat) / (1. + z_ref))**(z_evol - 1.)

        zbins = qso_z_bins
        histo_w, zbins = np.histogram(z_qso_cat, bins=zbins, weights=weights_qso_cat)
        histo_wz, _ = np.histogram(z_qso_cat, bins=zbins, weights=weights_qso_cat*z_qso_cat)
        selection = histo_w > 0
        z_qso = histo_wz[selection] / histo_w[selection]  # weighted mean in bins
        weights_qso = histo_w[selection]

        return z_qso, weights_qso

    def get_rp_pairs(self, z1, z2):
        r1 = self.cosmo.get_r_comov(z1)
        r2 = self.cosmo.get_r_comov(z2)

        # Get all pairs
        rp_pairs = (r1[:, None] - r2[None, :]).ravel()  # same sign as line 676 of cf.py (1-2)
        if 'discrete' not in self.main_tracer_types:
            rp_pairs = np.abs(rp_pairs)

        return rp_pairs

    def get_forest_weight_scaling(self, z, true_abs, assumed_abs):
        true_alpha = self.metal_matrix_config.getfloat(f'alpha_{true_abs}')
        assumed_alpha = self.metal_matrix_config.getfloat(f'alpha_{assumed_abs}', 2.9)
        scaling = (1 + z)**(true_alpha + assumed_alpha - 2)
        return scaling

    def compute_fast_metal_dmat(self, true_abs_1, true_abs_2):
        # Initialize tracer 1 redshift and weights
        if self.main_tracer_types[0] == 'continuous':
            wave1, weights1 = self.get_forest_weights(self._corr_item.tracer1)
            true_z1 = wave1 / picca_constants.ABSORBER_IGM[true_abs_1] - 1.
            assumed_z1 = wave1 / picca_constants.ABSORBER_IGM[self.main_tracers[0]] - 1.
            scaling_1 = self.get_forest_weight_scaling(true_z1, true_abs_1, self.main_tracers[0])
        else:
            true_z1, weights1 = self.get_qso_weights(self._corr_item.tracer1)
            assumed_z1 = true_z1
            scaling_1 = 1.

        # Initialize tracer 2 redshift and weights
        if self.main_tracer_types[1] == 'continuous':
            wave2, weights2 = self.get_forest_weights(self._corr_item.tracer2)
            true_z2 = wave2 / picca_constants.ABSORBER_IGM[true_abs_2] - 1.
            assumed_z2 = wave2 / picca_constants.ABSORBER_IGM[self.main_tracers[1]] - 1.
            scaling_2 = self.get_forest_weight_scaling(true_z2, true_abs_2, self.main_tracers[1])
        else:
            true_z2, weights2 = self.get_qso_weights(self._corr_item.tracer2)
            assumed_z2 = true_z2
            scaling_2 = 1.

        # Compute rp pairs
        true_rp_pairs = self.get_rp_pairs(true_z1, true_z2)
        assumed_rp_pairs = self.get_rp_pairs(assumed_z1, assumed_z2)

        # Compute weights
        weights = ((weights1 * scaling_1)[:, None] * (weights2 * scaling_2)[None, :]).ravel()

        # Distortion matrix grid
        rp_bin_edges = np.linspace(
            self._coordinates.rp_min, self._coordinates.rp_max, self.rp_nbins + 1)

        # Compute the distortion matrix
        dmat, _, __ = np.histogram2d(
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

        num_bins_total = self.rp_nbins * self.rt_nbins
        full_rp_eff = np.zeros(num_bins_total)
        full_rt_eff = np.zeros(num_bins_total)
        full_z_eff = np.zeros(num_bins_total)

        rp_indices = np.arange(self.rp_nbins)
        rt_bins = np.arange(
            self._coordinates.rt_binsize / 2, self._coordinates.rt_max,
            self._coordinates.rt_binsize
        )

        for j in range(self.rt_nbins):
            indices = j + self.rt_nbins * rp_indices

            full_rp_eff[indices] = rp_eff
            full_rt_eff[indices] = rt_bins[j]
            full_z_eff[indices] = z_eff

        return dmat, full_rp_eff, full_rt_eff, full_z_eff
