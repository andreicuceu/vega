import copy

import numpy as np
from astropy.io import fits
from picca import constants as picca_constants
from scipy.sparse import csr_matrix

from . import coordinates
from . import correlation_func as corr_func
from . import pktoxi, power_spectrum, utils


class Metals:
    """
    Class for computing metal correlations
    """
    # cache_pk = LRUCache(128)
    # cache_xi = LRUCache(128)
    cache_xi = {}

    growth_rate = None
    par_sigma_smooth = None
    per_sigma_smooth = None
    fast_metals = False

    def __init__(self, corr_item, fiducial, scale_params, data=None):
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
        self.cosmo = corr_item.cosmo
        self._data = data
        # self.PktoXi = PktoXi_obj
        self.size = corr_item.model_coordinates.rp_grid.size
        # ell_max = self._corr_item.config['model'].getint('ell_max', 6)
        self._coordinates = corr_item.model_coordinates
        self.fast_metals = corr_item.config['model'].getboolean('fast_metals', False)
        self.fast_metal_bias = corr_item.config['model'].getboolean('fast_metal_bias', True)
        self.rp_only_metal_mats = corr_item.config['model'].getboolean('rp_only_metal_mats', False)

        # Read the growth rate and sigma_smooth from the fiducial config
        if 'growth_rate' in fiducial:
            self.growth_rate = fiducial['growth_rate']
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
        self.new_metals = corr_item.new_metals
        if self.new_metals:
            self.metal_matrix_config = corr_item.config['metal-matrix']
            self.rp_nbins = self._coordinates.rp_nbins
            self.rt_nbins = self._coordinates.rt_nbins

        # Initialize metals
        self.Pk_metal = {}
        self.PktoXi = {}
        self.Xi_metal = {}
        self.rp_metal_dmats = {}
        if corr_item.has_metals:
            for name1, name2 in corr_item.metal_correlations:
                # Get the tracer info
                tracer1 = corr_item.tracer_catalog[name1]
                tracer2 = corr_item.tracer_catalog[name2]

                if self.new_metals:
                    if self.rp_only_metal_mats:
                        dmat, rp_grid, rt_grid, z_grid = self.compute_metal_rp_dmat(name1, name2)
                    else:
                        dmat, rp_grid, rt_grid, z_grid = self.compute_metal_dmat(name1, name2)

                    self.rp_metal_dmats[(name1, name2)] = dmat
                    metal_coordinates = coordinates.Coordinates.init_from_grids(
                        self._coordinates, rp_grid, rt_grid, z_grid)
                else:
                    # Read rp and rt for the metal correlation
                    metal_coordinates = data.metal_coordinates[(name1, name2)]

                # Get bin sizes
                if self._data is not None:
                    corr_item.config['metals']['bin_size_rp'] = \
                        str(corr_item.data_coordinates.rp_binsize)
                    corr_item.config['metals']['bin_size_rt'] = \
                        str(corr_item.data_coordinates.rt_binsize)

                # Assumes cross-corelations Lya x Metal and Metal x Lya are the same
                corr_hash = tuple(set((name1, name2)))

                # Initialize the metal correlation P(k)
                self.Pk_metal[corr_hash] = power_spectrum.PowerSpectrum(
                        self._corr_item.config['metals'], fiducial,
                        tracer1, tracer2, self._corr_item.name
                    )

                self.PktoXi[corr_hash] = pktoxi.PktoXi.init_from_Pk(
                    self.Pk_metal[corr_hash], corr_item.config['model'])

                # assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                # assert self._corr_item.config['metals'].getint('ell_max', ell_max) == ell_max, \
                #        "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                self.Xi_metal[corr_hash] = corr_func.CorrelationFunction(
                    self._corr_item.config['metals'], fiducial, metal_coordinates,
                    scale_params, tracer1, tracer2, metal_corr=True, cosmo=self.cosmo
                )

    def compute_xi_metal_metal(self, pk_lin, pars, name1, name2):
        corr_hash = tuple(set((name1, name2)))

        if corr_hash in self.cache_xi:
            return self.cache_xi[corr_hash]

        pk = self.Pk_metal[corr_hash].compute(pk_lin, pars, fast_metals=True)
        self.PktoXi[corr_hash].cache_pars = None
        xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi[corr_hash], pars)

        # Apply the metal matrix
        if self.new_metals:
            if self.rp_only_metal_mats:
                xi = (self.rp_metal_dmats[(name1, name2)]
                      @ xi.reshape(self.rp_nbins, self.rt_nbins)).flatten()
            else:
                xi = self.rp_metal_dmats[(name1, name2)] @ xi
        else:
            xi = self._data.metal_mats[(name1, name2)].dot(xi)

        self.cache_xi[corr_hash] = xi

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
            if 'growth_rate' in local_pars and self.growth_rate is not None:
                local_pars['growth_rate'] = self.growth_rate
        xi_metals = np.zeros(self.size)
        for name1, name2, in self._corr_item.metal_correlations:
            bias1, beta1, bias2, beta2 = utils.bias_beta(local_pars, name1, name2)
            corr_hash = tuple(set((name1, name2)))

            if (self.fast_metals and (name1 not in self.main_tracers)
                    and (name2 not in self.main_tracers)):
                xi_metals += bias1 * bias2 * self.compute_xi_metal_metal(
                    pk_lin, local_pars, name1, name2)

                continue

            # If not in fast metals mode, compute the usual way
            # Slow mode also allows the full save of components
            pk = self.Pk_metal[corr_hash].compute(
                pk_lin, local_pars, fast_metals=self.fast_metal_bias)
            if self.save_components:
                assert not self.fast_metal_bias, 'You need to set fast_metal_bias=False.'
                self.pk[component][(name1, name2)] = copy.deepcopy(pk)

            xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi[corr_hash], local_pars)

            # Save the components
            if self.save_components:
                # self.pk[component][(name1, name2)] = copy.deepcopy(pk)
                self.xi[component][(name1, name2)] = copy.deepcopy(xi)

            # Apply the metal matrix
            if self.new_metals:
                if self.rp_only_metal_mats:
                    xi = (self.rp_metal_dmats[(name1, name2)]
                          @ xi.reshape(self.rp_nbins, self.rt_nbins)).flatten()
                else:
                    xi = self.rp_metal_dmats[(name1, name2)] @ xi
            else:
                xi = self._data.metal_mats[(name1, name2)].dot(xi)

            if self.save_components:
                self.xi_distorted[component][(name1, name2)] = copy.deepcopy(xi)

            if self.fast_metal_bias:
                xi_metals += bias1 * bias2 * xi

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

        mean_distance = ((r1[:, None] + r2[None, :]) / 2).ravel()
        return rp_pairs, mean_distance

    def get_forest_weight_scaling(self, z, true_abs, assumed_abs):
        true_alpha = self.metal_matrix_config.getfloat(f'alpha_{true_abs}')
        assumed_alpha = self.metal_matrix_config.getfloat(f'alpha_{assumed_abs}', 2.9)
        scaling = (1 + z)**(true_alpha + assumed_alpha - 2)
        return scaling

    def compute_metal_dmat(self, true_abs_1, true_abs_2):
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
        true_rp_pairs, true_mean_distance = self.get_rp_pairs(true_z1, true_z2)
        assumed_rp_pairs, assumed_mean_distance = self.get_rp_pairs(assumed_z1, assumed_z2)

        # Compute weights
        weights = ((weights1 * scaling_1)[:, None] * (weights2 * scaling_2)[None, :]).ravel()

        # Distortion matrix grid
        rp_bin_edges = np.linspace(
            self._coordinates.rp_min, self._coordinates.rp_max, self.rp_nbins + 1)

        # Compute the distortion matrix
        rp_1d_dmat, _, __ = np.histogram2d(
            assumed_rp_pairs, true_rp_pairs, bins=(rp_bin_edges, rp_bin_edges), weights=weights)

        # Normalize (sum of weights should be one for each input rp,rt)
        sum_rp_1d_dmat = np.sum(rp_1d_dmat, axis=0)
        rp_1d_dmat /= (sum_rp_1d_dmat + (sum_rp_1d_dmat==0))

        # independently, we compute the r_trans distortion matrix
        rt_bin_edges = np.linspace(0, self._coordinates.rt_max, self.rt_nbins + 1)

        # we have input_dist , output_dist and weight.
        # we don't need to store the absolute comoving distances
        # but the ratio between output and input.
        # we rebin that to compute the rest faster.
        # histogram of distance scaling with proper weights:
        # dist*theta = r_trans
        # theta_max  = r_trans_max/dist
        # solid angle contibuting for each distance propto
        # theta_max**2 = (r_trans_max/dist)**2 propto 1/dist**2
        # we weight the distances with this additional factor
        # using the input or the output distance in the solid angle weight
        # gives virtually the same result
        # distance_ratio_weights,distance_ratio_bins =
        # np.histogram(output_dist/input_dist,bins=4*rtbins.size,
        # weights=weights/input_dist**2*(input_rp<cf.r_par_max)*(input_rp>cf.r_par_min))
        # we also select only distance ratio for which the input_rp
        # (that of the true separation of the absorbers) is small, so that this
        # fast matrix calculation is accurate where it matters the most
        distance_ratio_weights, distance_ratio_bins = np.histogram(
            assumed_mean_distance / true_mean_distance, bins=4*rt_bin_edges.size,
            weights=weights/true_mean_distance**2*(np.abs(true_rp_pairs) < 20.)
        )
        distance_ratios = (distance_ratio_bins[1:] + distance_ratio_bins[:-1]) / 2

        # now we need to scan as a function of separation angles, or equivalently rt.
        rt_bin_centers = (rt_bin_edges[:-1] + rt_bin_edges[1:]) / 2
        rt_bin_half_size = self._coordinates.rt_binsize / 2

        # we are oversampling the correlation function rt grid to correctly compute bin migration.
        oversample = 7
        # the -2/oversample term is needed to get a even-spaced grid
        delta_rt = np.linspace(
            -rt_bin_half_size, rt_bin_half_size*(1 - 2 / oversample), oversample)[None, :]
        rt_1d_dmat = np.zeros((self.rt_nbins, self.rt_nbins))

        for i, rt in enumerate(rt_bin_centers):
            # the weight is proportional to rt+delta_rt to get the correct solid angle effect
            # inside the bin (but it's almost a negligible effect)
            rt_1d_dmat[:, i], _ = np.histogram(
                (distance_ratios[:, None] * (rt + delta_rt)[None, :]).ravel(), bins=rt_bin_edges,
                weights=(distance_ratio_weights[:, None] * (rt + delta_rt)[None, :]).ravel()
            )

        # normalize
        sum_rt_1d_dmat = np.sum(rt_1d_dmat, axis=0)
        rt_1d_dmat /= (sum_rt_1d_dmat + (sum_rt_1d_dmat == 0))

        # now that we have both distortion along r_par and r_trans, we have to combine them
        # we just multiply the two matrices, with indices splitted for rt and rp
        # full_index = rt_index + cf.num_bins_r_trans * rp_index
        # rt_index   = full_index%cf.num_bins_r_trans
        # rp_index  = full_index//cf.num_bins_r_trans
        num_bins_total = self.rp_nbins * self.rt_nbins
        dmat = csr_matrix(
            np.einsum('ij,kl->ikjl', rp_1d_dmat, rt_1d_dmat).reshape(num_bins_total, num_bins_total)
        )

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
        r_par_eff_1d = sum_assumed_weight_rp / (sum_assumed_weight + (sum_assumed_weight == 0))
        z_eff_1d = sum_weight_z / (sum_assumed_weight + (sum_assumed_weight == 0))

        # r_trans has no weights here
        r1 = np.arange(self.rt_nbins) * self._coordinates.rt_max / self.rt_nbins
        r2 = (1 + np.arange(self.rt_nbins)) * self._coordinates.rt_max / self.rt_nbins

        # this is to account for the solid angle effect on the mean
        r_trans_eff_1d = (2 * (r2**3 - r1**3)) / (3 * (r2**2 - r1**2))

        full_index = np.arange(num_bins_total)
        rt_index = full_index % self.rt_nbins
        rp_index = full_index // self.rt_nbins

        full_rp_eff = r_par_eff_1d[rp_index]
        full_rt_eff = r_trans_eff_1d[rt_index]
        full_z_eff = z_eff_1d[rp_index]

        return dmat, full_rp_eff, full_rt_eff, full_z_eff

    def compute_metal_rp_dmat(self, true_abs_1, true_abs_2):
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
        true_rp_pairs, _ = self.get_rp_pairs(true_z1, true_z2)
        assumed_rp_pairs, _ = self.get_rp_pairs(assumed_z1, assumed_z2)

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
