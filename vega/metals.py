import copy

import numpy as np
from picca import constants as picca_constants
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, kron

from . import coordinates, pktoxi, power_spectrum, redshift_weights, utils
from . import correlation_func as corr_func


class Metals:
    """
    Class for computing metal correlations

    With ``new_metals = True`` the metal distortion matrices are built at initialization from
    the stacked delta files (``compute_metal_dmat``, ``compute_metal_rp_dmat``). The
    ``metal_matrix_convention`` option in ``[model]`` selects how:

    - ``estimator`` (default): the matrix is the expectation of the pair estimator,
      ``M_AB = sum_{pairs in A, true in B} W E / sum_{pairs in A} W``, with ``W`` the estimator
      weight of the pair and ``E`` the product of the amplitude evolutions of the two legs,
      normalized at the reference redshift ``z0 = metal_z_ref`` (the effective redshift of the
      correlation). The undistorted metal correlation is evaluated at ``z0`` and ``Xi_metal``
      applies no further evolution, since ``((1 + z0) / (1 + z_eff))**alpha = 1``. The
      evolution is therefore counted exactly once, and ``z0`` is used both to normalize the
      matrix and as the redshift of the undistorted correlation, so that the two cannot
      differ. Consequence: the metal terms do not respond to runtime ``alpha_*``, which are
      frozen into the matrices at initialization (varying them is refused by
      ``vega.interface_utils.check_metal_model_options``).
    - ``legacy``: upstream behaviour (columns normalized to unit sum, undistorted correlation
      evaluated at the effective redshift of the observed bin with the runtime evolution). It
      counts the redshift evolution twice (review finding F01) and uses the coordinates of the
      assumed bin (F02). It is kept to reproduce previous results.

    With ``new_metals = False`` the metal matrices and the coordinates are read from the
    picca files. This path is legacy: picca's pair-based matrix already contains the amplitude
    evolution (normalized at picca's own ``--z-ref``, default 2.25), but the undistorted metal
    correlation is evaluated at the ``Z_<name>`` grid of the file with the evolution applied
    again, so the evolution is counted twice (F01). A warning is printed at initialization.

    ``fast_metals = True`` speeds up the metal term with three approximations (about 10x for
    Lya x Lya):

    1. Each metal x metal correlation is computed once, at the first model evaluation, and
       afterwards only rescaled by the metal biases, so its shape is frozen. This is exact
       only if no sampled parameter changes it. ``vega.interface_utils.check_metal_model_options``
       therefore refuses fast metals together with varied ``beta_metals``, ``beta_<metal>``
       or ``alpha_<metal>`` (where ``alpha_<metal>`` is not refused anyway, see above),
       ``no-metal-decomp = False`` and ``metal-scaling = True``.
    2. Main x metal pairs with equal (beta1, beta2) share one undistorted correlation within
       an evaluation (the cache key is (beta1, beta2, all parameter values), not the pair).
       Under the ``estimator`` convention every pair is evaluated at the same ``z0``, so this
       is exact apart from species-dependent differences of the effective r_par of
       <~0.2-0.4 Mpc/h (Delta chi2 <= 0.004, review finding F03). Under ``legacy`` and with
       ``new_metals = False`` the shared correlation also carries the z grid and the
       evolution exponent of the pair that filled the cache (Delta chi2 ~ 0.19 for
       Lya x Lya in the DESI baseline).
    3. The growth rate is replaced by its fiducial value in all metal terms.
    """

    # cache_pk = LRUCache(128)
    # cache_xi = LRUCache(128)
    growth_rate = None
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
        self._rmu_binning = self._data is not None and self._data._rmu_binning

        # self.PktoXi = PktoXi_obj
        self.size = corr_item.model_coordinates.rp_grid.size
        if self._rmu_binning:
            ups = corr_item.config["model"].getint("rmu_metal_grid_factor", 1)
            self._coordinates = copy.deepcopy(corr_item.model_coordinates)
            self._coordinates.rp_binsize /= ups
            self._coordinates.rt_binsize /= ups
            self._coordinates.rp_nbins *= ups
            self._coordinates.rt_nbins *= ups
        else:
            self._coordinates = corr_item.model_coordinates

        self.rp_only_metal_mats = corr_item.config["model"].getboolean("rp_only_metal_mats", False)

        # Redshift bins
        self.zmin = corr_item.config["data"].getfloat("zmin", 0.0)
        self.zmax = corr_item.config["data"].getfloat("zmax", 10.0)

        self.separate_metal_auto_biases = corr_item.config["model"].getboolean(
            "separate-metal-auto-biases", False
        )
        self.single_metal_beta = corr_item.config["model"].getboolean("single-metal-beta", False)

        self.fast_metals = corr_item.config["model"].getboolean("fast_metals", False)
        self.fast_metal_bias = corr_item.config["model"].getboolean("fast_metal_bias", True)
        if self.fast_metals or self.separate_metal_auto_biases:
            self.fast_metal_bias = True

        self.cache_xi_metal_metal = {}
        self.cache_xi_metal_cross_main = {}

        # Read the growth rate and sigma_smooth from the fiducial config
        if "growth_rate" in fiducial:
            self.growth_rate = fiducial["growth_rate"]

        self.save_components = fiducial.get("save-components", False)

        if self.save_components and (self.fast_metals or self.separate_metal_auto_biases):
            raise ValueError(
                "Cannot save pk/cf components in fast_metals mode."
                " Either turn fast_metals off, or turn off write_pk/write_cf."
            )

        self.pk = {"peak": {}, "smooth": {}, "full": {}}
        self.xi = {"peak": {}, "smooth": {}, "full": {}}
        self.xi_distorted = {"peak": {}, "smooth": {}, "full": {}}

        # Build a mask for the cross-correlations with the main tracers (Lya, QSO)
        self.main_tracers = [corr_item.tracer1["name"], corr_item.tracer2["name"]]
        self.is_auto_correlation = self.main_tracers[0] == self.main_tracers[1]
        self.main_tracer_types = [corr_item.tracer1["type"], corr_item.tracer2["type"]]
        self.main_cross_mask = [
            tracer1 in self.main_tracers or tracer2 in self.main_tracers
            for (tracer1, tracer2) in corr_item.metal_correlations
        ]

        self._interp_coords = None
        self._interp = None

        # Convention used to build the new-metal matrices. It is only used if new_metals is True,
        # but it is validated and stored in any case.
        self.metal_matrix_convention = corr_item.config["model"].get(
            "metal_matrix_convention", "estimator"
        )
        if self.metal_matrix_convention not in ("estimator", "legacy"):
            raise ValueError(
                f"Invalid metal_matrix_convention = '{self.metal_matrix_convention}' in [model] "
                f"of '{corr_item.name}'. Use 'estimator' or 'legacy'."
            )

        # Exponents entering the new-metal matrices, filled by _resolve_matrix_exponents
        self._weight_evol_exponents = {}
        self._amplitude_exponents = {}

        # If in new metals mode, read the stacked delta files
        self.new_metals = corr_item.new_metals

        # The picca-matrix path is legacy: warn once per correlation (no change of the model)
        if not self.new_metals:
            warning = (
                f"WARNING: '{corr_item.name}' uses the metal matrices from picca "
                "(new_metals = False), a legacy path. The picca pair-based matrix already "
                "contains the amplitude evolution of the metals, normalized at picca's own "
                "--z-ref (default 2.25), but Vega evaluates the undistorted metal correlation "
                "at the Z_<name> grid of the file and applies the redshift evolution again, so "
                "the evolution is counted twice (review finding F01). Use new_metals = True."
            )
            if self.fast_metals:
                warning += (
                    " With fast_metals = True, main x metal pairs with equal (beta1, beta2) share "
                    "one undistorted correlation within an evaluation although their Z_<name> "
                    "grids and evolution exponents differ (review finding F03; "
                    "Delta chi2 ~ 0.19 for Lya x Lya in the DESI baseline)."
                )
            print(warning)

        if self.new_metals:
            self.metal_matrix_config = corr_item.config["metal-matrix"]

            # Reference redshift z0 of the metal matrices. It normalizes the amplitude evolution
            # inside the matrix (estimator convention) and is the redshift at which the
            # undistorted metal correlation is evaluated, so that both always use the same value
            # (any z0 cancels if they agree). The cosmology gives the growth factor D(z) / D(z0)
            # of the amplitude evolution.
            if corr_item.z_eff is None:
                raise ValueError(
                    f"The reference redshift of the metal matrices of '{corr_item.name}' is "
                    "z_eff, but it is not set. Set 'zeff' in [data sets] of the main config."
                )
            self.metal_z_ref = corr_item.z_eff
            self._Omega_m = fiducial.get("Omega_m")
            self._Omega_de = fiducial.get("Omega_de")

            if corr_item.has_metals:
                # The matrices are built below, so the exponents must be resolved first
                self._resolve_matrix_exponents()
            self.rp_nbins = self._coordinates.rp_nbins
            self.rt_nbins = self._coordinates.rt_nbins
            self.size = self.rp_nbins * self.rt_nbins

            if self._rmu_binning:
                rpg = np.linspace(
                    self._coordinates.rp_min, self._coordinates.rp_max, self.rp_nbins + 1
                )
                rpg = (rpg[1:] + rpg[:-1]) / 2
                rtg = np.arange(
                    self._coordinates.rt_binsize / 2,
                    self._coordinates.rt_max,
                    self._coordinates.rt_binsize,
                )
                self._interp_coords = np.vstack(
                    [corr_item.model_coordinates.rp_grid, corr_item.model_coordinates.rt_grid]
                ).T
                self._interp = RegularGridInterpolator(
                    (rpg, rtg),
                    np.zeros((rpg.size, rtg.size)),
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )

        # Initialize metals
        self.Pk_metal = {}
        self.PktoXi = {}
        self.Xi_metal = {}
        self.rp_metal_dmats = {}
        if corr_item.has_metals:
            if self.new_metals and self.metal_matrix_convention == "estimator":
                # The matrices already include the separate evolution of both tracer legs.
                corr_item.config["metals"]["new-bias-evolution"] = "False"

            for corr_hash in corr_item.metal_correlations:
                name1, name2 = corr_hash

                # Get the tracer info
                tracer1 = corr_item.tracer_catalog[name1]
                tracer2 = corr_item.tracer_catalog[name2]

                if self.new_metals:
                    if self.rp_only_metal_mats:
                        dmat, rp_grid, rt_grid, z_grid = self.compute_metal_rp_dmat(name1, name2)
                    else:
                        dmat, rp_grid, rt_grid, z_grid = self.compute_metal_dmat(name1, name2)

                    self.rp_metal_dmats[corr_hash] = dmat
                    metal_coordinates = coordinates.RtRpCoordinates.init_from_grids(
                        self._coordinates, rp_grid, rt_grid, z_grid
                    )
                else:
                    # Read rp and rt for the metal correlation
                    if corr_hash in data.metal_coordinates:
                        metal_coordinates = data.metal_coordinates[corr_hash]
                    else:
                        metal_coordinates = data.metal_coordinates[corr_hash[::-1]]

                # Get bin sizes
                if self._data is not None:
                    corr_item.config["metals"]["bin_size_rp"] = str(
                        corr_item.data_coordinates.rp_binsize
                    )
                    corr_item.config["metals"]["bin_size_rt"] = str(
                        corr_item.data_coordinates.rt_binsize
                    )

                # Initialize the metal correlation P(k)
                self.Pk_metal[corr_hash] = power_spectrum.PowerSpectrum(
                    self._corr_item.config["metals"],
                    fiducial,
                    tracer1,
                    tracer2,
                    self._corr_item.name,
                )

                self.PktoXi[corr_hash] = pktoxi.PktoXi.init_from_Pk(
                    self.Pk_metal[corr_hash], corr_item.config["model"]
                )

                # assert len(self.Pk_metal[(name1, name2)].muk_grid) == len(self.Pk_core.muk_grid)
                # assert self._corr_item.config['metals'].getint('ell_max', ell_max) == ell_max, \
                #        "Core and metals must have the same ell_max"

                # Initialize the metal correlation Xi
                self.Xi_metal[corr_hash] = corr_func.CorrelationFunction(
                    self._corr_item.config["metals"],
                    fiducial,
                    metal_coordinates,
                    scale_params,
                    tracer1,
                    tracer2,
                    metal_corr=True,
                    cosmo=self.cosmo,
                )

    def compute_xi_metal_metal(self, pk_lin, pars, corr_hash):
        """Compute M_1 x M_2 metal cross-correlations with caching.

        This is the fast-metals approximation of the metal x metal terms: the correlation is
        computed once, at the first evaluation, and cached by tracer names only. Afterwards
        only the metal biases, which are applied later in ``compute``, can change; the shape of
        the correlation is frozen. Everything else must stay fixed, in particular ``beta``,
        ``alpha`` of the metals, ``no-metal-decomp`` and ``metal-scaling``, and
        ``vega.interface_utils.check_metal_model_options`` refuses fast metals if any of them is
        varied or set. The growth rate is replaced by its fiducial value.

        Parameters
        ----------
        pk_lin : Array
            linear power spectrum
        pars : dict
            parameters
        corr_hash : tuple
            tuple of the two metal tracer names

        Returns
        -------
        1D Array
            metal cross-correlation function
        """
        if corr_hash in self.cache_xi_metal_metal:
            return self.cache_xi_metal_metal[corr_hash]

        self.cache_xi_metal_metal[corr_hash] = self.compute_metal_corr_slow(
            pars, pk_lin, corr_hash, fast_metals=True
        )

        return self.cache_xi_metal_metal[corr_hash]

    def compute_xi_metal_cross_main(self, pk_lin, pars, corr_hash, beta1, beta2):
        """Compute M x main tracer metal cross-correlations with caching.

        This is the fast-metals approximation of the metal x main-tracer terms. The cache key
        is (beta1, beta2, all parameter values), not the pair of tracers: pairs with equal
        (beta1, beta2) share one undistorted correlation within an evaluation, computed for
        the pair that first fills the cache, and only the metal matrix of each pair is applied
        afterwards. Under the ``estimator`` convention all pairs are evaluated at the same
        ``z0``, so this is exact apart from species-dependent differences of the effective
        r_par of <~0.2-0.4 Mpc/h (Delta chi2 <= 0.004, review finding F03). Under ``legacy``
        and with ``new_metals = False`` the shared correlation also carries the z grid and the
        evolution exponent of the pair that filled the cache (Delta chi2 ~ 0.19 for Lya x Lya
        in the DESI baseline). Metal biases are added later and the growth rate is replaced by
        its fiducial value.

        Parameters
        ----------
        pk_lin : Array
            linear power spectrum
        pars : dict
            parameters
        corr_hash : tuple
            tuple of the two metal tracer names
        beta1 : float
            beta parameter for first tracer
        beta2 : float
            beta parameter for second tracer

        Returns
        -------
        1D Array
            metal cross-correlation function
        """
        par_array = np.array([pars[key] for key in sorted(pars.keys())])
        xi_hash = (beta1, beta2, *tuple(par_array))

        if xi_hash in self.cache_xi_metal_cross_main:
            xi = self.cache_xi_metal_cross_main[xi_hash]
        else:
            xi = self.compute_metal_corr_slow(
                pars, pk_lin, corr_hash, fast_metals=True, add_metal_dmat=False
            )
            self.cache_xi_metal_cross_main[xi_hash] = xi

        # Add the correct metal dmats for each correlation
        dmat_xi = self.apply_metal_matrix(xi, corr_hash)

        return dmat_xi

    def compute_metal_corr_slow(
        self, pars, pk_lin, corr_hash, fast_metals, add_metal_dmat=True, component=None
    ):
        """Compute a single metal correlation function, optionally applying the metal matrix.

        Parameters
        ----------
        pars : dict
            Computation parameters
        pk_lin : array
            Linear power spectrum
        corr_hash : tuple
            (name1, name2) tracer name pair
        fast_metals : bool
            If True, use the fast metals approximation (no bias factors in P(k))
        add_metal_dmat : bool, optional
            Whether to apply the metal distortion matrix, by default True
        component : str, optional
            Component key for saving ('peak', 'smooth', or 'full'), by default None

        Returns
        -------
        1D Array
            Metal correlation function, optionally distorted
        """
        pk = self.Pk_metal[corr_hash].compute(pk_lin, pars, fast_metals=fast_metals)
        self.PktoXi[corr_hash].cache_pars = None
        xi = self.Xi_metal[corr_hash].compute(pk, pk_lin, self.PktoXi[corr_hash], pars)
        # If cross-metal correlation, multiply by 2 to account for symmetry
        if self.is_auto_correlation and corr_hash[0] != corr_hash[1]:
            xi *= 2

        if self.save_components:
            assert not fast_metals, "You need to set fast_metal_bias=False."
            assert component is not None, "You need to provide component name."
            self.pk[component][corr_hash] = copy.deepcopy(pk)
            self.xi[component][corr_hash] = copy.deepcopy(xi)

        if not add_metal_dmat:
            return xi

        # Apply the metal matrix
        dmat_xi = self.apply_metal_matrix(xi, corr_hash)

        if self.save_components:
            self.xi_distorted[component][corr_hash] = copy.deepcopy(dmat_xi)

        return dmat_xi

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
            if "growth_rate" in local_pars and self.growth_rate is not None:
                local_pars["growth_rate"] = self.growth_rate

        xi_metals = np.zeros(self.size)
        self.cache_xi_metal_cross_main = {}  # clear cache each time compute is called
        for corr_hash in self._corr_item.metal_correlations:
            name1, name2 = corr_hash

            if self.single_metal_beta:
                if name1 not in self.main_tracers:
                    local_pars[f"beta_{name1}"] = local_pars["beta_metals"]
                if name2 not in self.main_tracers:
                    local_pars[f"beta_{name2}"] = local_pars["beta_metals"]

            bias1, beta1, bias2, beta2 = utils.bias_beta(local_pars, name1, name2)

            is_cross_with_main_tracer = name1 in self.main_tracers or name2 in self.main_tracers

            if is_cross_with_main_tracer:
                bias_product = bias1 * bias2
            elif self.separate_metal_auto_biases and name1 != name2:
                if f"bias_{name1}_{name2}" in local_pars:
                    bias_auto_factor = local_pars.get(f"bias_{name1}_{name2}", 1.0)
                elif f"bias_{name2}_{name1}" in local_pars:
                    bias_auto_factor = local_pars.get(f"bias_{name2}_{name1}", 1.0)
                else:
                    raise ValueError(
                        f"Separate metal auto biases is on, but no bias_{name1}_{name2}"
                        f" or bias_{name2}_{name1} parameter found for {corr_hash}."
                    )
                bias_product = bias1 * bias2 * bias_auto_factor
            else:
                bias_product = bias1 * bias2

            if self.fast_metals and is_cross_with_main_tracer:
                xi_metals += bias_product * self.compute_xi_metal_cross_main(
                    pk_lin, local_pars, corr_hash, beta1, beta2
                )

            elif self.fast_metals:
                xi_metals += bias_product * self.compute_xi_metal_metal(
                    pk_lin, local_pars, corr_hash
                )

            else:
                # If not in fast metals mode, compute the usual way
                # Slow mode also allows the full save of components
                xi = self.compute_metal_corr_slow(
                    local_pars,
                    pk_lin,
                    corr_hash,
                    fast_metals=self.fast_metal_bias,
                    component=component,
                )
                if self.fast_metal_bias:
                    xi_metals += bias_product * xi
                else:
                    xi_metals += xi

        if self._rmu_binning:
            self._interp.values = xi_metals.reshape(self.rp_nbins, self.rt_nbins)
            xi_metals = self._interp(self._interp_coords).ravel()

        return xi_metals

    def apply_metal_matrix(self, xi, corr_hash):
        """Apply metal distortion matrices to input correlation function.

        Parameters
        ----------
        xi : 1D Array
            correlation function to apply metal distortion matrices to
        corr_hash : tuple
            tuple of the two metal tracer names

        Returns
        -------
        1D Array
            distorted correlation function
        """
        if self.new_metals:
            if self.rp_only_metal_mats:
                dmat_xi = (
                    self.rp_metal_dmats[corr_hash] @ xi.reshape(self.rp_nbins, self.rt_nbins)
                ).flatten()
            else:
                dmat_xi = self.rp_metal_dmats[corr_hash] @ xi
        else:
            if corr_hash in self._data.metal_mats:
                dmat_xi = self._data.metal_mats[corr_hash].dot(xi)
            else:
                dmat_xi = self._data.metal_mats[corr_hash[::-1]].dot(xi)

        return dmat_xi

    @staticmethod
    def rebin(vector, rebin_factor):
        """Rebin a vector by a factor of rebin_factor.

        Parameters
        ----------
        vector : 1D array
            Vector to rebin
        rebin_factor : int
            Rebinning factor

        Returns
        -------
        1D array
            Rebinned vector
        """
        return redshift_weights.rebin(vector, rebin_factor)

    def get_forest_weights(self, main_tracer):
        """Read wavelength and weight arrays from the stacked delta file for a forest tracer.

        Parameters
        ----------
        main_tracer : dict
            Tracer config dict with at least 'type' and 'weights-path' keys

        Returns
        -------
        array, array
            Wavelength array and corresponding weight array
        """
        assert main_tracer["type"] == "continuous", (
            f"get_forest_weights expects a continuous tracer, got '{main_tracer['type']}'"
        )
        rebin_factor = self.metal_matrix_config.getint("rebin_factor", fallback=None)
        return redshift_weights.get_forest_weights(
            main_tracer["weights-path"], rebin_factor=rebin_factor
        )

    def _resolve_matrix_exponents(self):
        """Resolve the redshift-evolution exponents used to build the new-metal matrices.

        Two different kinds of exponent enter the matrices, and they are stored separately.

        The *weighting exponent* of a main tracer describes how the correlation-function
        estimator weighted the data: forest pixels with weight ``(1 + z)**(gamma - 1)``
        (``gamma`` = 2.9 by default) and objects with ``((1 + z) / (1 + z_ref))**(gamma - 1)``
        (``gamma`` = 1.44 by default). It is read from ``weight_evol_<tracer name>`` in
        ``[metal-matrix]``. The deprecated keys are ``alpha_<tracer name>`` for a forest tracer
        and ``z_evol_objects`` for a discrete tracer; they are still accepted with a warning.
        If a new and a deprecated key are both present with different values, a ValueError
        is raised. The Lyb forest has tracer name ``LYA`` and uses ``weight_evol_LYA``.

        The *amplitude exponent* of a true absorber X (a metal, or the main tracer of the
        corresponding leg) describes the redshift evolution of the bias of X,
        ``b_X(z) ~ (1 + z)**alpha_X``. It is read from

        - ``[parameters]`` (``alpha_<X>``) in the ``estimator`` convention. An
          ``alpha_<metal>`` still present in ``[metal-matrix]`` is ignored with a warning;
        - ``[metal-matrix]`` (``alpha_<X>``) in the ``legacy`` convention if present (deprecated,
          with a warning), otherwise ``[parameters]``. This keeps old configurations unchanged.

        The parameter values are taken from ``corr_item.parameters``, set by VegaInterface,
        because the matrices are built once at initialization. They are therefore frozen:
        see ``vega.interface_utils.check_metal_model_options``.

        Warnings are printed once, here. The results are stored in
        ``self._weight_evol_exponents`` (main tracer name -> weighting exponent) and
        ``self._amplitude_exponents`` (true absorber name -> amplitude exponent).

        Raises
        ------
        ValueError
            If a deprecated and a new weighting key disagree, or if an amplitude exponent is
            needed but ``alpha_<X>`` is not in ``[parameters]``.
        """
        corr_item = self._corr_item
        corr_name = corr_item.name
        config = self.metal_matrix_config
        legacy = self.metal_matrix_convention == "legacy"

        # Estimator weighting exponent of each main tracer
        for tracer in (corr_item.tracer1, corr_item.tracer2):
            name = tracer["name"]
            if name in self._weight_evol_exponents:
                continue

            if tracer["type"] == "continuous":
                deprecated_key, default_value = f"alpha_{name}", 2.9
            else:
                deprecated_key, default_value = "z_evol_objects", 1.44

            new_key = f"weight_evol_{name}"
            new_value = config.getfloat(new_key)
            deprecated_value = config.getfloat(deprecated_key)

            if new_value is not None and deprecated_value is not None:
                if new_value != deprecated_value:
                    raise ValueError(
                        f"[metal-matrix] of '{corr_name}' has both {new_key} = {new_value} and "
                        f"the deprecated {deprecated_key} = {deprecated_value}. {deprecated_key} "
                        f"was renamed to {new_key}; remove it or give both the same value."
                    )
                value = new_value
            elif new_value is not None:
                value = new_value
            elif deprecated_value is not None:
                value = deprecated_value
                message = (
                    f"WARNING: [metal-matrix] of '{corr_name}': {deprecated_key} is deprecated "
                    f"as the estimator weighting exponent, it was renamed to {new_key}."
                )
                if tracer["type"] == "continuous":
                    if legacy:
                        message += (
                            f" In the legacy convention {deprecated_key} still also sets the "
                            f"amplitude exponent of the {name} leg."
                        )
                    else:
                        message += (
                            f" The amplitude exponent of the {name} leg is taken from "
                            f"[parameters] {deprecated_key}."
                        )
                print(message)
            else:
                value = default_value

            self._weight_evol_exponents[name] = value

        # Amplitude exponent of each true absorber appearing in the metal correlations
        absorbers = []
        for corr_hash in corr_item.metal_correlations:
            for absorber in corr_hash:
                if absorber not in absorbers:
                    absorbers.append(absorber)

        parameters = getattr(corr_item, "parameters", None)
        legacy_keys = []
        ignored_keys = []
        for absorber in absorbers:
            key = f"alpha_{absorber}"
            matrix_value = config.getfloat(key)
            is_main_tracer = absorber in self.main_tracers
            is_discrete_main = (
                is_main_tracer and corr_item.tracer_catalog[absorber]["type"] == "discrete"
            )

            if legacy:
                # The legacy matrices have no amplitude evolution for the discrete-tracer leg
                if is_discrete_main:
                    continue

                if matrix_value is not None:
                    self._amplitude_exponents[absorber] = matrix_value
                    if not is_main_tracer:
                        legacy_keys.append(key)
                    continue
            elif matrix_value is not None and not is_main_tracer:
                parameter_value = None if parameters is None else parameters.get(key)
                if parameter_value is not None and parameter_value != matrix_value:
                    ignored_keys.append(
                        f"{key} ([metal-matrix]: {matrix_value}, [parameters]: {parameter_value})"
                    )
                else:
                    ignored_keys.append(key)

            if parameters is None or key not in parameters:
                raise ValueError(
                    f"The amplitude exponent {key} of the true absorber {absorber} is needed to "
                    f"build the metal matrices of '{corr_name}', but {key} is not in "
                    "[parameters]. Add it to the [parameters] section of the main config."
                )
            self._amplitude_exponents[absorber] = parameters[key]

        if legacy_keys:
            print(
                f"WARNING: [metal-matrix] of '{corr_name}': the amplitude exponents "
                f"{', '.join(legacy_keys)} are deprecated. They are only used in the legacy "
                "convention; the estimator convention reads alpha_<absorber> from [parameters]."
            )
        if ignored_keys:
            print(
                f"WARNING: [metal-matrix] of '{corr_name}': {'; '.join(ignored_keys)} are ignored "
                "in the estimator convention. The amplitude exponents are taken from [parameters]."
                " Remove them from [metal-matrix]."
            )

    def get_qso_weights(self, tracer):
        """Read QSO redshifts and compute weighted redshift bins from the catalog file.

        The catalog weights follow the estimator weighting exponent of the tracer
        (``weight_evol_<tracer name>``, resolved by ``_resolve_matrix_exponents``).

        Parameters
        ----------
        tracer : dict
            Tracer config dict with at least 'type' and 'weights-path' keys

        Returns
        -------
        array, array
            Weighted mean redshifts per bin and corresponding weight sums
        """
        assert tracer["type"] == "discrete", (
            f"get_qso_weights expects a discrete tracer, got '{tracer['type']}'"
        )
        return redshift_weights.get_qso_weights(
            tracer["weights-path"],
            z_ref=self.metal_matrix_config.getfloat("z_ref_objects", 2.25),
            z_evol=self._weight_evol_exponents[tracer["name"]],
            z_bins=self.metal_matrix_config.getint("z_bins_objects", 1000),
        )

    def get_rp_pairs(self, z1, z2):
        """Compute line-of-sight separation pairs and mean comoving distances.

        Parameters
        ----------
        z1 : array
            Redshifts of tracer 1
        z2 : array
            Redshifts of tracer 2

        Returns
        -------
        array, array
            rp_pairs (all z1-z2 pair separations) and mean_distance (mean comoving distances)
        """
        if np.any(z1 < 0) or np.any(z2 < 0):
            raise ValueError("Attempting to compute distance to a negative redshift")
        r1 = self.cosmo.get_r_comov(z1)
        r2 = self.cosmo.get_r_comov(z2)

        # Get all pairs
        rp_pairs = (r1[:, None] - r2[None, :]).ravel()  # same sign as line 676 of cf.py (1-2)
        if "discrete" not in self.main_tracer_types:
            rp_pairs = np.abs(rp_pairs)

        mean_distance = ((r1[:, None] + r2[None, :]) / 2).ravel()
        return rp_pairs, mean_distance

    def get_forest_weight_scaling(self, z, true_abs, assumed_abs):
        """Compute the weight scaling factor due to assuming a wrong absorber identity.

        The scaling is ``(1 + z)**(true_alpha + assumed_alpha - 2)``, where ``true_alpha`` is the
        amplitude exponent of the true absorber (redshift evolution of its bias,
        ``b ~ (1 + z)**alpha``) and ``assumed_alpha`` is the weighting exponent of the assumed
        main tracer (the estimator weighted its pixels with ``(1 + z)**(assumed_alpha - 1)``).
        Both are resolved at initialization by ``_resolve_matrix_exponents``.

        Parameters
        ----------
        z : array
            Redshift at the true absorber wavelength
        true_abs : str
            Name of the true absorber (e.g. 'SiIII(1207)')
        assumed_abs : str
            Name of the assumed absorber (e.g. 'LYA')

        Returns
        -------
        array
            Weight scaling factor at each redshift
        """
        true_alpha = self._amplitude_exponents[true_abs]
        assumed_alpha = self._weight_evol_exponents[assumed_abs]
        scaling = (1 + z) ** (true_alpha + assumed_alpha - 2)
        return scaling

    def _metal_pair_catalogue(self, true_abs_1, true_abs_2):
        """Build the stacked-weight pair catalogue of the new-metal matrices.

        Every pair of wavelength pixels of the stacked forest weights (or of redshift bins
        of the discrete-tracer catalogue) is one pair. Separations are computed twice: with
        the true absorber identities and with the assumed ones (the main tracers, as in the
        estimator). Both matrix builders share this construction.

        In the ``legacy`` convention the numerator and denominator weights are the same
        array: the stacked weights times ``get_forest_weight_scaling`` for each forest leg.

        In the ``estimator`` convention, leg k carries the estimator weight ``w_k`` (the
        stacked weight times ``(1 + z^t_k)**(gamma - 1)`` for a forest leg, with ``gamma`` the
        weighting exponent of the main tracer; the histogram weights of
        ``get_qso_weights`` for a discrete leg) and the amplitude evolution ``e_k`` of the true
        absorber, see ``_amplitude_evolution``. The factor ``(1 + z^t)`` of the forest weight
        is evaluated at the true instead of the assumed redshift. The two differ by the
        constant ratio ``(1 + z^t) / (1 + z^a) = lambda_Lya / lambda_X``, which cancels in the
        normalization of the matrix rows. The numerator weights are ``w_1 e_1 w_2 e_2`` and the
        denominator weights ``w_1 w_2``. ``e_k`` is applied per leg, before the pair product.

        In both conventions pairs whose assumed mean redshift lies outside
        ``[zmin, zmax]`` get zero weight.

        Parameters
        ----------
        true_abs_1 : str
            Name of the true absorber for tracer 1
        true_abs_2 : str
            Name of the true absorber for tracer 2

        Returns
        -------
        dict
            Flattened arrays of shape (N_1 * N_2,), tracer 1 being the slow index:

            - ``true_rp_pairs``, ``assumed_rp_pairs``: line-of-sight separation in Mpc/h
              (absolute value if no tracer is discrete);
            - ``true_mean_distance``, ``assumed_mean_distance``: mean comoving distance of the
              pair in Mpc/h;
            - ``true_pair_z``: mean true redshift of the pair;
            - ``numerator_weights``, ``denominator_weights``: see above (the same array object
              in the ``legacy`` convention).
        """
        legacy = self.metal_matrix_convention == "legacy"
        leg_tracers = (self._corr_item.tracer1, self._corr_item.tracer2)
        leg_absorbers = (true_abs_1, true_abs_2)

        true_redshifts = []
        assumed_redshifts = []
        numerator_leg_weights = []
        denominator_leg_weights = []
        for leg, (tracer, true_abs) in enumerate(zip(leg_tracers, leg_absorbers)):
            main_name = self.main_tracers[leg]

            if self.main_tracer_types[leg] == "continuous":
                wave, stacked_weights = self.get_forest_weights(tracer)
                true_z = wave / picca_constants.ABSORBER_IGM[true_abs] - 1.0
                assumed_z = wave / picca_constants.ABSORBER_IGM[main_name] - 1.0

                if legacy:
                    leg_weights = stacked_weights * self.get_forest_weight_scaling(
                        true_z, true_abs, main_name
                    )
                else:
                    weighting_exponent = self._weight_evol_exponents[main_name]
                    leg_weights = stacked_weights * (1 + true_z) ** (weighting_exponent - 1)
            else:
                # The histogram weights of the catalogue are already the estimator weights
                true_z, leg_weights = self.get_qso_weights(tracer)
                assumed_z = true_z

            true_redshifts.append(true_z)
            assumed_redshifts.append(assumed_z)
            denominator_leg_weights.append(leg_weights)
            if legacy:
                numerator_leg_weights.append(leg_weights)
            else:
                numerator_leg_weights.append(
                    leg_weights * self._amplitude_evolution(true_abs, true_z)
                )

        true_z1, true_z2 = true_redshifts
        assumed_z1, assumed_z2 = assumed_redshifts

        # Compute rp pairs
        true_rp_pairs, true_mean_distance = self.get_rp_pairs(true_z1, true_z2)
        assumed_rp_pairs, assumed_mean_distance = self.get_rp_pairs(assumed_z1, assumed_z2)

        # Pair weights, restricted to the redshift range of the data
        numerator_weights = (
            numerator_leg_weights[0][:, None] * numerator_leg_weights[1][None, :]
        ).ravel()
        pair_z = (assumed_z1[:, None] + assumed_z2[None, :]) / 2.0
        z_mask = ((pair_z >= self.zmin) & (pair_z <= self.zmax)).ravel()
        numerator_weights *= z_mask

        if legacy:
            denominator_weights = numerator_weights
        else:
            denominator_weights = (
                denominator_leg_weights[0][:, None] * denominator_leg_weights[1][None, :]
            ).ravel()
            denominator_weights *= z_mask

        return {
            "true_rp_pairs": true_rp_pairs,
            "assumed_rp_pairs": assumed_rp_pairs,
            "true_mean_distance": true_mean_distance,
            "assumed_mean_distance": assumed_mean_distance,
            "true_pair_z": ((true_z1[:, None] + true_z2[None, :]) / 2.0).ravel(),
            "numerator_weights": numerator_weights,
            "denominator_weights": denominator_weights,
        }

    def _amplitude_evolution(self, true_abs, true_z):
        """Compute the amplitude evolution of one leg of a metal pair, normalized at z0.

        ``e(z) = ((1 + z) / (1 + z0))**alpha * D(z) / D(z0)``: the redshift evolution of the
        bias of the true absorber, ``b ~ (1 + z)**alpha``, times that of the linear growth of
        the matter perturbations. The exponent is ``alpha`` and not ``alpha - 1`` because the
        growth is explicit. This is the same factor that ``CorrelationFunction`` applies to
        the main correlation (``rel_z_evol**alpha`` times the growth), here evaluated pixel by
        pixel at the true redshift instead of once at the effective redshift. The reference
        redshift ``z0 = self.metal_z_ref`` is the one at which the undistorted metal
        correlation is evaluated, see the class docstring.

        Parameters
        ----------
        true_abs : str
            Name of the true absorber of the leg (a metal, or the main tracer itself).
        true_z : array
            Redshift of each pixel (or redshift bin) of the leg for the true absorber.

        Returns
        -------
        array
            Dimensionless amplitude evolution, with the shape of ``true_z``.
        """
        exponent = self._amplitude_exponents[true_abs]
        bias_evolution = ((1 + true_z) / (1 + self.metal_z_ref)) ** exponent
        growth = utils.normalized_growth_factor(
            true_z, self.metal_z_ref, self._Omega_m, self._Omega_de
        )
        return bias_evolution * growth

    def _effective_rp_true_bin(self, true_rp_pairs, numerator_weights, rp_bin_edges):
        """Compute the effective r_par of each true bin of a metal matrix (estimator).

        The undistorted metal correlation multiplies the matrix column of the *true* bin, so
        the separation at which it is evaluated is the mean true separation of the pairs that
        populate that column, weighted with the numerator weights of the matrix. Bins without
        pairs fall back to the bin centre; their matrix columns vanish.

        Parameters
        ----------
        true_rp_pairs : array, shape (N_pairs,)
            True line-of-sight separation of each pair, in Mpc/h.
        numerator_weights : array, shape (N_pairs,)
            Numerator weights of the pairs.
        rp_bin_edges : array, shape (N_rp + 1,)
            Bin edges of the r_par grid, in Mpc/h.

        Returns
        -------
        array, shape (N_rp,)
            Effective r_par of each true bin, in Mpc/h.
        """
        sum_true_weight, _ = np.histogram(
            true_rp_pairs, bins=rp_bin_edges, weights=numerator_weights
        )
        sum_true_weight_rp, _ = np.histogram(
            true_rp_pairs, bins=rp_bin_edges, weights=numerator_weights * true_rp_pairs
        )

        empty_bins = sum_true_weight == 0
        rp_bin_centers = (rp_bin_edges[1:] + rp_bin_edges[:-1]) / 2
        mean_true_rp = sum_true_weight_rp / (sum_true_weight + empty_bins)

        return np.where(empty_bins, rp_bin_centers, mean_true_rp)

    def compute_metal_dmat(self, true_abs_1, true_abs_2):
        """Compute the 2D metal distortion matrix for a given absorber pair.

        Builds a full (rp, rt) distortion matrix as the Kronecker product of separate 1D
        rp and rt distortion matrices computed from the stacked delta files. The pair
        construction is shared with ``compute_metal_rp_dmat``, see ``_metal_pair_catalogue``.
        Rows are the observed (assumed) bins A, columns the true bins B.

        The convention is set by ``metal_matrix_convention`` in ``[model]``.

        ``estimator``: the matrix is the expectation of the pair estimator,
        ``M_AB = sum_{pairs in A, true in B} W E / sum_{pairs in A} W``, with ``W`` the
        estimator weight of the pair and ``E`` the product of the amplitude evolutions of the
        two legs normalized at ``z0 = self.metal_z_ref``. The rows are normalized by the weight
        of *all* pairs in the observed bin, including those whose true separation lies outside
        the grid, so that the row sums measure the average amplitude evolution and the weight
        leaving the grid. For r_trans the rows are normalized by their sum over the grid. The
        support of the r_trans matrix is therefore incomplete in the last observed bins
        (true separations beyond ``rt_max`` are not histogrammed); the metal correlation is
        negligible there. The effective r_par of a column is the mean true separation of its
        pairs, and the effective redshift is ``z0`` everywhere: the undistorted metal
        correlation is evaluated at ``z0`` and ``Xi_metal`` applies no further evolution, since
        ``((1 + z0) / (1 + z_eff))**alpha = 1`` with ``z0 = z_eff``. The amplitude exponents are
        frozen into the matrix at initialization, so the metal terms do not respond to
        ``alpha_*`` at runtime (varying them is refused by
        ``vega.interface_utils.check_metal_model_options``).

        ``legacy``: upstream behaviour. The columns are normalized to unit sum, the pairs are
        weighted with ``get_forest_weight_scaling`` and the undistorted metal correlation is
        evaluated at the effective redshift of the observed bin (r_par of the assumed bin,
        mean true pair redshift) with the runtime redshift evolution. The evolution is thus
        counted both in the matrix and in ``Xi_metal`` (review finding F01), and the effective
        coordinates are those of the assumed bin (F02).

        Parameters
        ----------
        true_abs_1 : str
            Name of the true absorber for tracer 1
        true_abs_2 : str
            Name of the true absorber for tracer 2

        Returns
        -------
        csr_matrix, array, array, array
            Distortion matrix of shape (N_rp N_rt, N_rp N_rt), and the effective rp, rt (Mpc/h)
            and z grids of the undistorted metal correlation, each of shape (N_rp N_rt,).
        """
        legacy = self.metal_matrix_convention == "legacy"

        pairs = self._metal_pair_catalogue(true_abs_1, true_abs_2)
        true_rp_pairs = pairs["true_rp_pairs"]
        assumed_rp_pairs = pairs["assumed_rp_pairs"]
        true_mean_distance = pairs["true_mean_distance"]
        assumed_mean_distance = pairs["assumed_mean_distance"]
        weights = pairs["numerator_weights"]

        # Distortion matrix grid
        rp_bin_edges = np.linspace(
            self._coordinates.rp_min, self._coordinates.rp_max, self.rp_nbins + 1
        )

        # Compute the distortion matrix
        rp_1d_dmat, _, __ = np.histogram2d(
            assumed_rp_pairs, true_rp_pairs, bins=(rp_bin_edges, rp_bin_edges), weights=weights
        )

        if legacy:
            # Normalize (sum of weights should be one for each input rp,rt)
            sum_rp_1d_dmat = np.sum(rp_1d_dmat, axis=0)
            rp_1d_dmat /= sum_rp_1d_dmat + (sum_rp_1d_dmat == 0)
        else:
            # Normalize each observed bin by the weight of all its pairs, also those whose
            # true separation is off the grid
            sum_assumed_weight, _ = np.histogram(
                assumed_rp_pairs, bins=rp_bin_edges, weights=pairs["denominator_weights"]
            )
            rp_1d_dmat /= (sum_assumed_weight + (sum_assumed_weight == 0))[:, None]

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
            assumed_mean_distance / true_mean_distance,
            bins=4 * rt_bin_edges.size,
            weights=weights / true_mean_distance**2 * (np.abs(true_rp_pairs) < 20.0),
        )
        distance_ratios = (distance_ratio_bins[1:] + distance_ratio_bins[:-1]) / 2

        # now we need to scan as a function of separation angles, or equivalently rt.
        rt_bin_centers = (rt_bin_edges[:-1] + rt_bin_edges[1:]) / 2
        rt_bin_half_size = self._coordinates.rt_binsize / 2

        # we are oversampling the correlation function rt grid to correctly compute bin migration.
        oversample = 7
        # the -2/oversample term is needed to get a even-spaced grid
        delta_rt = np.linspace(
            -rt_bin_half_size, rt_bin_half_size * (1 - 2 / oversample), oversample
        )[None, :]
        rt_1d_dmat = np.zeros((self.rt_nbins, self.rt_nbins))

        for i, rt in enumerate(rt_bin_centers):
            # the weight is proportional to rt+delta_rt to get the correct solid angle effect
            # inside the bin (but it's almost a negligible effect)
            rt_1d_dmat[:, i], _ = np.histogram(
                (distance_ratios[:, None] * (rt + delta_rt)[None, :]).ravel(),
                bins=rt_bin_edges,
                weights=(distance_ratio_weights[:, None] * (rt + delta_rt)[None, :]).ravel(),
            )

        # normalize
        if legacy:
            sum_rt_1d_dmat = np.sum(rt_1d_dmat, axis=0)
            rt_1d_dmat /= sum_rt_1d_dmat + (sum_rt_1d_dmat == 0)
        else:
            # Rows are observed bins; true separations beyond rt_max are not histogrammed,
            # so the last observed bins are normalized over an incomplete support
            sum_rt_rows = np.sum(rt_1d_dmat, axis=1)
            rt_1d_dmat /= (sum_rt_rows + (sum_rt_rows == 0))[:, None]

        # now that we have both distortion along r_par and r_trans, we have to combine them
        # we just multiply the two matrices, with indices splitted for rt and rp
        # full_index = rt_index + cf.num_bins_r_trans * rp_index
        # rt_index   = full_index%cf.num_bins_r_trans
        # rp_index  = full_index//cf.num_bins_r_trans
        # kron(A, B)[i * n_rt + k, j * n_rt + l] = A[i, j] B[k, l] gives the same products as
        # einsum("ij,kl->ikjl") reshaped to (N, N), without the dense (N_rp N_rt)^2 temporary
        # (about 7 GB per species for a 300 x 100 grid) and much faster.
        num_bins_total = self.rp_nbins * self.rt_nbins
        dmat = kron(csr_matrix(rp_1d_dmat), csr_matrix(rt_1d_dmat), format="csr")
        dmat.eliminate_zeros()
        dmat.sort_indices()

        if legacy:
            # Mean assumed weights
            sum_assumed_weight, _ = np.histogram(
                assumed_rp_pairs, bins=rp_bin_edges, weights=weights
            )
            sum_assumed_weight_rp, _ = np.histogram(
                assumed_rp_pairs,
                bins=rp_bin_edges,
                weights=weights * (assumed_rp_pairs[None, :].ravel()),
            )

            # Return the redshift of the actual absorber, which is the average of true_z1
            # and true_z2
            sum_weight_z, _ = np.histogram(
                assumed_rp_pairs,
                bins=rp_bin_edges,
                weights=weights * pairs["true_pair_z"],
            )
            r_par_eff_1d = sum_assumed_weight_rp / (sum_assumed_weight + (sum_assumed_weight == 0))
            z_eff_1d = sum_weight_z / (sum_assumed_weight + (sum_assumed_weight == 0))
        else:
            # The undistorted correlation multiplies the column of the true bin, so it is
            # evaluated at the mean true separation of the pairs of that bin, at z0
            r_par_eff_1d = self._effective_rp_true_bin(true_rp_pairs, weights, rp_bin_edges)
            z_eff_1d = np.full(self.rp_nbins, self.metal_z_ref)

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
        """Compute the rp-only metal distortion matrix for a given absorber pair.

        Builds a 1D rp distortion matrix only (no rt mixing), applied separately
        for each rt bin. Faster than the full 2D compute_metal_dmat, and with the same
        conventions (``metal_matrix_convention`` in ``[model]``), see there for details.
        Rows are the observed (assumed) bins A, columns the true bins B.

        ``estimator``: ``M_AB = sum_{pairs in A, true in B} W E / sum_{pairs in A} W`` with
        ``E`` the product of the per-leg amplitude evolutions normalized at
        ``z0 = self.metal_z_ref``; the effective r_par of a column is the mean true separation
        of its pairs and the effective redshift is ``z0`` everywhere, so ``Xi_metal`` applies
        no further evolution. The metal terms do not respond to runtime ``alpha_*``.

        ``legacy``: upstream behaviour. Columns are normalized by the weight of the pairs in
        the true bin, and the effective r_par and redshift are those of the assumed bin
        (review findings F01 and F02).

        Parameters
        ----------
        true_abs_1 : str
            Name of the true absorber for tracer 1
        true_abs_2 : str
            Name of the true absorber for tracer 2

        Returns
        -------
        array, array, array, array
            rp distortion matrix of shape (N_rp, N_rp), and the effective rp, rt (Mpc/h) and z
            grids, each of shape (N_rp N_rt,).
        """
        legacy = self.metal_matrix_convention == "legacy"

        pairs = self._metal_pair_catalogue(true_abs_1, true_abs_2)
        true_rp_pairs = pairs["true_rp_pairs"]
        assumed_rp_pairs = pairs["assumed_rp_pairs"]
        weights = pairs["numerator_weights"]

        # Distortion matrix grid
        rp_bin_edges = np.linspace(
            self._coordinates.rp_min, self._coordinates.rp_max, self.rp_nbins + 1
        )

        # Compute the distortion matrix
        dmat, _, __ = np.histogram2d(
            assumed_rp_pairs, true_rp_pairs, bins=(rp_bin_edges, rp_bin_edges), weights=weights
        )

        if legacy:
            # Normalize (sum of weights should be one for each input rp,rt)
            sum_true_weight, _ = np.histogram(true_rp_pairs, bins=rp_bin_edges, weights=weights)
            dmat *= ((sum_true_weight > 0) / (sum_true_weight + (sum_true_weight == 0)))[None, :]

            # Mean assumed weights
            sum_assumed_weight, _ = np.histogram(
                assumed_rp_pairs, bins=rp_bin_edges, weights=weights
            )
            sum_assumed_weight_rp, _ = np.histogram(
                assumed_rp_pairs,
                bins=rp_bin_edges,
                weights=weights * (assumed_rp_pairs[None, :].ravel()),
            )

            # Return the redshift of the actual absorber, which is the average of true_z1
            # and true_z2
            sum_weight_z, _ = np.histogram(
                assumed_rp_pairs,
                bins=rp_bin_edges,
                weights=weights * pairs["true_pair_z"],
            )

            rp_eff = sum_assumed_weight_rp / (sum_assumed_weight + (sum_assumed_weight == 0))
            z_eff = sum_weight_z / (sum_assumed_weight + (sum_assumed_weight == 0))
        else:
            # Normalize each observed bin by the weight of all its pairs, also those whose
            # true separation is off the grid
            sum_assumed_weight, _ = np.histogram(
                assumed_rp_pairs, bins=rp_bin_edges, weights=pairs["denominator_weights"]
            )
            dmat /= (sum_assumed_weight + (sum_assumed_weight == 0))[:, None]

            # The undistorted correlation multiplies the column of the true bin, so it is
            # evaluated at the mean true separation of the pairs of that bin, at z0
            rp_eff = self._effective_rp_true_bin(true_rp_pairs, weights, rp_bin_edges)
            z_eff = np.full(self.rp_nbins, self.metal_z_ref)

        num_bins_total = self.rp_nbins * self.rt_nbins
        full_rp_eff = np.zeros(num_bins_total)
        full_rt_eff = np.zeros(num_bins_total)
        full_z_eff = np.zeros(num_bins_total)

        rp_indices = np.arange(self.rp_nbins)
        rt_bins = np.arange(
            self._coordinates.rt_binsize / 2, self._coordinates.rt_max, self._coordinates.rt_binsize
        )

        for j in range(self.rt_nbins):
            indices = j + self.rt_nbins * rp_indices

            full_rp_eff[indices] = rp_eff
            full_rt_eff[indices] = rt_bins[j]
            full_z_eff[indices] = z_eff

        return dmat, full_rp_eff, full_rt_eff, full_z_eff
