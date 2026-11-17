"""Analytic marginalization of linear model parameters.

The data vector of each correlation is modelled as

    data = model(theta) + templates @ coefficients + noise,

where the templates do not depend on the sampled parameters ``theta``. The
coefficients of templates with a Gaussian prior of covariance ``Sigma`` are
marginalized exactly by the covariance update

    C -> C + templates @ Sigma @ templates.T,

which is applied once at initialization. The chi2 computed with the updated
covariance equals the chi2 at the best-fit coefficients including their prior
penalty, and the updated determinant normalizes the marginal likelihood, so
likelihood evaluations cost the same as without marginalization.

Coefficients with a flat prior are the limit of an infinitely wide Gaussian.
With ``F`` their templates and ``P`` the precision after the Gaussian update,
they are marginalized by the projection

    P -> P - P @ F @ inv(F.T @ P @ F) @ F.T @ P,

also applied once at initialization. The flat-prior templates must be linearly
independent within the fitted bins, and the marginal likelihood is then defined
up to a constant that depends on the units of the coefficients.

This module keeps track of the templates of each correlation, propagates them
through the distortion matrix and the data masks, builds the covariance update,
and recovers the best-fit coefficients from a residual when they are requested.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_array, hstack

from vega.broadband_poly import BroadbandPolynomials

# Singular values below this fraction of the largest one are treated as
# degenerate template combinations and dropped from the covariance update.
DEGENERATE_MODE_THRESHOLD = 1e-8

# Flat-prior templates are rejected as linearly dependent when the condition number of
# their normal matrix, after scaling each template to unit norm, exceeds this value.
MAX_FLAT_CONDITION_NUMBER = 1e12


@dataclass
class TemplateFamily:
    """Set of linear templates of one correlation sharing a prior width.

    Attributes
    ----------
    name : str
        Family identifier, e.g. ``"small-scale"``.
    ids : tuple of str
        Identifier of each coefficient, in column order.
    templates : scipy.sparse.csr_array
        Response of the distorted model to a unit coefficient, with shape
        ``(model_size, num_templates)`` on the full (unmasked) output grid.
    prior_sigma : float
        Width of the zero-mean Gaussian prior on each coefficient, in the
        units of the coefficient. ``np.inf`` denotes a flat prior.
    num_modes : int
        Number of non-degenerate template combinations within the fitted bins.
        Set when the covariance update of the family is computed; equal to the
        number of templates for a flat prior.
    amplitude_factor : float
        Factor already included in ``templates`` that is absent when the model is
        computed directly from the full power spectrum, by default 1.
    """

    name: str
    ids: tuple
    templates: csr_array
    prior_sigma: float
    num_modes: int = 0
    amplitude_factor: float = 1.0

    @property
    def has_flat_prior(self):
        """Whether the coefficients of this family have a flat prior.

        Returns
        -------
        bool
            True if ``prior_sigma`` is infinite.
        """
        return np.isinf(self.prior_sigma)


def include_marginalized_scales_in_fit(data):
    """Extend the data and model masks to the marginalized small-scale bins.

    Used with ``fit-marginalized-scales``: the bins excluded by the minimum
    separation cuts are kept in the fit, and their information is removed by
    the small-scale templates instead.

    Parameters
    ----------
    data : vega.data.Data
        Data of one correlation. Its ``data_mask``, ``model_mask`` and masked
        data vector are updated in place.

    Raises
    ------
    ValueError
        If the extended data and model masks select different numbers of bins.
    """
    corr_item = data.corr_item
    cuts_config = corr_item.config["cuts"]

    data.data_mask |= data.data_coordinates.get_mask_marginalization_scales(
        cuts_config, corr_item.marginalize_small_scales
    )
    data.model_mask |= data.dist_model_coordinates.get_mask_marginalization_scales(
        cuts_config, corr_item.marginalize_small_scales
    )

    if data.data_mask.sum() != data.model_mask.sum():
        raise ValueError(
            "Data and model masks should be the same after marginalization scale cuts."
            " The most likely reason is a mismatch in rp-min between the data and"
            " the model coordinates. Check that 'rp-min = -300' for cross-correlations, "
            " or set it to the smallest rp in your distortion matrix."
        )

    # Recompute masked data vector and size
    data._masked_data_vec = None
    _ = data.masked_data_vec


def build_small_scale_family(data):
    """Build the small-scale templates of one correlation.

    Each template is one bin of the undistorted correlation function at small
    separation, propagated through the distortion matrix.

    Parameters
    ----------
    data : vega.data.Data
        Data of one correlation with small-scale marginalization configured.

    Returns
    -------
    TemplateFamily
        Distorted small-scale templates with the ``marginalize-prior-sigma``
        Gaussian prior.

    Raises
    ------
    ValueError
        If the correlation has no distortion matrix.
    """
    corr_item = data.corr_item
    if not data.has_distortion:
        raise ValueError("Distortion matrix required for marginalization")

    undistorted_templates = corr_item.get_undist_xi_marg_templates()
    templates = data.distortion_mat.dot(undistorted_templates)

    ids = tuple(f"small-scale-{corr_item.name} ({i})" for i in range(templates.shape[1]))
    return TemplateFamily(
        name="small-scale",
        ids=ids,
        templates=templates,
        prior_sigma=corr_item.marginalize_small_scales_prior_sigma,
    )


def build_broadband_family(data):
    """Build the additive broadband polynomial templates of one correlation.

    Every coefficient of every additive polynomial term (before or after
    distortion) gives one template: the response of the distorted model to a unit
    coefficient. Terms added before distortion are propagated through the
    distortion matrix. Multiplicative broadbands are not applied to the templates.

    In the peak/smooth decomposition the additive polynomials enter both
    components, so a coefficient ``a`` contributes ``(1 + bao_amp) * a * polynomial``
    to the model. The templates include this factor, evaluated at the ``bao_amp``
    value of ``[parameters]``, so that marginalized coefficients have the same
    meaning as sampled ones.

    Parameters
    ----------
    data : vega.data.Data
        Data of one correlation with ``marginalize-broadbands`` enabled.

    Returns
    -------
    TemplateFamily
        Distorted broadband templates, with the ``broadband-prior-sigma`` Gaussian
        prior or a flat prior when that option is absent.

    Raises
    ------
    ValueError
        If the data are not on an rp/rt grid, if the correlation has no additive
        polynomial broadband, or if the prior width is not positive.
    """
    corr_item = data.corr_item
    if data.use_multipoles or data.is_direct_multipoles or data._rmu_binning:
        raise ValueError(
            f"marginalize-broadbands is only supported for rp/rt data ({corr_item.name})."
        )

    broadband_config = corr_item.config["broadband"] if "broadband" in corr_item.config else {}
    broadband = BroadbandPolynomials(
        broadband_config,
        corr_item.name,
        corr_item.model_coordinates,
        corr_item.dist_model_coordinates,
    )

    ids = []
    columns = []
    for position in ("pre-add", "post-add"):
        coordinates = (
            corr_item.model_coordinates
            if position == "pre-add"
            else corr_item.dist_model_coordinates
        )

        for term in broadband.bb_terms[position]:
            # Sky broadbands are not polynomials in their parameters
            if term["func"] != "broadband":
                continue

            term_ids = broadband.get_coefficient_names(term)

            for coefficient_id in term_ids:
                # Response of the model evaluator itself to a unit coefficient
                unit_coefficients = dict.fromkeys(term_ids, 0.0)
                unit_coefficients[coefficient_id] = 1.0
                column = broadband._compute_broadband(term, unit_coefficients, coordinates)

                if position == "pre-add" and data.has_distortion:
                    column = data.distortion_mat.dot(column)

                ids.append(coefficient_id)
                columns.append(column)

    if not ids:
        raise ValueError(
            f"marginalize-broadbands is set for {corr_item.name}, but it has no additive "
            "polynomial broadband."
        )

    prior_sigma = corr_item.config["model"].getfloat("broadband-prior-sigma", np.inf)
    if not prior_sigma > 0:
        raise ValueError(f"broadband-prior-sigma must be positive ({corr_item.name}).")

    amplitude_factor = 1.0 + corr_item.parameters["bao_amp"]
    templates = csr_array(amplitude_factor * np.column_stack(columns))

    return TemplateFamily(
        name="broadband",
        ids=tuple(ids),
        templates=templates,
        prior_sigma=prior_sigma,
        num_modes=len(ids) if np.isinf(prior_sigma) else 0,
        amplitude_factor=amplitude_factor,
    )


def project_flat_templates(precision, flat_templates):
    """Marginalize flat-prior coefficients by projecting their templates out of a precision.

    Parameters
    ----------
    precision : 2D array
        Precision matrix of the fitted bins, shape ``(data_size, data_size)``.
    flat_templates : 2D array
        Templates with a flat prior restricted to the fitted bins, shape
        ``(data_size, num_flat)``.

    Returns
    -------
    projected_precision : 2D array
        ``P - P F inv(F.T P F) F.T P``, with shape ``(data_size, data_size)``.
    log_normalization : float
        ``log det(F.T P F) - num_flat * log(2 pi)``, the term that integrating
        over the coefficients adds to minus twice the log-likelihood.

    Raises
    ------
    ValueError
        If the templates are linearly dependent within the fitted bins.
    """
    # A precision obtained by numerical inversion is symmetric only to round-off, and
    # for an ill-conditioned covariance the asymmetry is not negligible. The projection
    # removes the templates from both sides of the quadratic form only if it is symmetric.
    precision = 0.5 * (precision + precision.T)

    # Scale each template to unit norm: the projection is unchanged, and the
    # conditioning of the normal matrix no longer depends on the template units.
    weighted_templates = precision.dot(flat_templates)
    template_norms = np.sqrt(np.sum(flat_templates * weighted_templates, axis=0))
    if np.any(~np.isfinite(template_norms)) or np.any(template_norms <= 0):
        raise ValueError("A flat-prior template vanishes within the fitted bins.")

    scaled_templates = flat_templates / template_norms
    scaled_weighted_templates = weighted_templates / template_norms
    normal_matrix = scaled_templates.T.dot(scaled_weighted_templates)
    normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)

    eigenvalues = np.linalg.eigvalsh(normal_matrix)
    if eigenvalues[0] <= eigenvalues[-1] / MAX_FLAT_CONDITION_NUMBER:
        raise ValueError(
            "Templates with a flat prior are linearly dependent within the fitted bins. "
            "Remove the redundant broadband terms or set a Gaussian prior."
        )

    cholesky_factor = cho_factor(normal_matrix)
    scaled_coefficient_matrix = cho_solve(cholesky_factor, scaled_weighted_templates.T)
    projected_precision = precision - scaled_weighted_templates.dot(scaled_coefficient_matrix)

    log_det_normal_matrix = np.sum(np.log(eigenvalues)) + 2 * np.sum(np.log(template_norms))
    log_normalization = log_det_normal_matrix - flat_templates.shape[1] * np.log(2 * np.pi)

    return projected_precision, log_normalization


def solve_with_covariance(covariance, right_hand_side):
    """Solve ``covariance @ solution = right_hand_side`` for a symmetric covariance.

    Parameters
    ----------
    covariance : 2D array
        Covariance matrix, shape ``(data_size, data_size)``. It may be overwritten.
    right_hand_side : 2D array
        Right-hand sides, shape ``(data_size, num_columns)``.

    Returns
    -------
    2D array
        Solution with shape ``(data_size, num_columns)``.
    """
    try:
        cholesky_factor = cho_factor(covariance, overwrite_a=True)
    except np.linalg.LinAlgError:
        # Covariances that are not positive definite are accepted elsewhere in Vega
        return np.linalg.solve(covariance, right_hand_side)

    return cho_solve(cholesky_factor, right_hand_side)


def solve_coefficients(masked_templates, weighted_templates, prior_variances):
    """Build the solution for the best-fit coefficients of linear templates.

    With ``C`` the covariance before any marginalization, ``T`` the templates and
    ``Lambda`` the diagonal prior precision (zero for flat priors), the best-fit
    coefficients for a residual ``r`` are

        inv(T.T inv(C) T + Lambda) @ T.T @ inv(C) @ r,

    and the inverse of the normal matrix is their covariance at fixed sampled
    parameters. The original covariance is used on purpose: the precision after
    the Gaussian update is ill-conditioned along the marginalized directions and
    is not accurate enough to recover the coefficients from it.

    Parameters
    ----------
    masked_templates : 2D array
        Templates within the fitted bins, shape ``(data_size, num_coefficients)``.
    weighted_templates : 2D array
        ``inv(C) @ T``, with shape ``(data_size, num_coefficients)``.
    prior_variances : 1D array
        Prior variance of each coefficient; infinite for flat priors.

    Returns
    -------
    coefficient_matrix : 2D array
        Matrix giving the coefficients from a residual, shape
        ``(num_coefficients, data_size)``.
    coefficient_errors : 1D array
        Standard deviation of each coefficient at fixed sampled parameters.
    """
    normal_matrix = masked_templates.T.dot(weighted_templates)
    normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
    normal_matrix[np.diag_indices_from(normal_matrix)] += 1 / prior_variances

    inverse_normal_matrix = np.linalg.inv(normal_matrix)
    coefficient_matrix = inverse_normal_matrix.dot(weighted_templates.T)

    # Round-off can leave tiny negative variances for unconstrained coefficients
    coefficient_errors = np.sqrt(np.clip(np.diag(inverse_normal_matrix), 0.0, None))

    return coefficient_matrix, coefficient_errors


class ComponentMarginalization:
    """Linear templates of one correlation and their covariance update.

    Attributes
    ----------
    families : list of TemplateFamily
        Template families of this correlation; empty when nothing is marginalized.
    cov_update : 2D array or None
        Sum of ``templates @ Sigma @ templates.T`` over the families, restricted
        to the fitted bins, with shape ``(data_size, data_size)``.
    num_small_scale_modes : int
        Number of non-degenerate small-scale modes, subtracted from the number
        of data points when counting degrees of freedom.
    flat_log_normalization : float
        Term added to the log-determinant of the covariance by the flat-prior
        coefficients; zero without them. Set when the precision of the
        correlation is first computed.
    """

    def __init__(self, data):
        """Build the templates of one correlation and update its covariance.

        Parameters
        ----------
        data : vega.data.Data
            Data of one correlation, with its covariance matrix, distortion matrix
            and masks already read. With ``fit-marginalized-scales`` its masks are
            extended, and its covariance matrix is updated in place.
        """
        self._data = data
        corr_item = data.corr_item

        self.families = []
        self.cov_update = None
        self.num_small_scale_modes = 0
        self.flat_log_normalization = 0.0

        if corr_item.marginalize_small_scales:
            self.families.append(build_small_scale_family(data))

            # The masks must be final before the templates are restricted to the fitted bins
            if corr_item.fit_marg_scales:
                include_marginalized_scales_in_fit(data)

        if corr_item.config["model"].getboolean("marginalize-broadbands", False):
            self.families.append(build_broadband_family(data))

        # The families are fixed from here on: stack their templates and priors once
        self._templates = None
        self._prior_variances = np.empty(0)
        self._amplitude_factors = np.empty(0)
        if len(self.families) == 1:
            self._templates = self.families[0].templates
        elif self.families:
            self._templates = hstack([family.templates for family in self.families], format="csr")
        if self.families:
            self._prior_variances = np.concatenate(
                [np.full(len(family.ids), family.prior_sigma**2) for family in self.families]
            )
            self._amplitude_factors = np.concatenate(
                [np.full(len(family.ids), family.amplitude_factor) for family in self.families]
            )

        for family in self.families:
            # Flat priors are handled by a projection of the precision matrix
            if family.has_flat_prior:
                continue

            family_cov_update = self._compute_cov_update(family)
            if self.cov_update is None:
                self.cov_update = family_cov_update
            else:
                self.cov_update += family_cov_update

            if family.name == "small-scale":
                self.num_small_scale_modes = family.num_modes

        if self.cov_update is not None:
            # Any inverse or determinant computed before the update is no longer valid
            data._inv_masked_cov = None
            data._log_cov_det = None
            data._cov_mat[np.ix_(data.data_mask, data.data_mask)] += self.cov_update

    @property
    def num_coefficients(self):
        """Total number of marginalized coefficients of this correlation.

        Returns
        -------
        int
            Number of template columns summed over the families.
        """
        return sum(len(family.ids) for family in self.families)

    @property
    def coefficient_ids(self):
        """Identifiers of the marginalized coefficients of this correlation.

        Returns
        -------
        tuple of str
            Coefficient identifiers in the column order of ``templates``.
        """
        return tuple(coefficient_id for family in self.families for coefficient_id in family.ids)

    @property
    def templates(self):
        """Templates of all families on the full output grid.

        Returns
        -------
        scipy.sparse.csr_array
            Template matrix with shape ``(model_size, num_coefficients)``.
        """
        return self._templates

    @property
    def prior_variances(self):
        """Prior variance of each marginalized coefficient.

        Returns
        -------
        1D array
            Prior variances with shape ``(num_coefficients,)``; infinite for flat priors.
        """
        return self._prior_variances

    @property
    def amplitude_factors(self):
        """Factor included in each template that is absent for a direct power spectrum.

        Returns
        -------
        1D array
            Amplitude factors with shape ``(num_coefficients,)``.
        """
        return self._amplitude_factors

    @property
    def num_other_coefficients(self):
        """Number of marginalized coefficients that are not small-scale modes.

        Returns
        -------
        int
            Number of coefficients counted as model parameters in the degrees of freedom.
        """
        return sum(len(family.ids) for family in self.families if family.name != "small-scale")

    @property
    def has_flat_templates(self):
        """Whether any coefficient of this correlation has a flat prior.

        Returns
        -------
        bool
            True if at least one family has a flat prior.
        """
        return any(family.has_flat_prior for family in self.families)

    def get_masked_flat_templates(self):
        """Templates with a flat prior restricted to the fitted bins.

        Returns
        -------
        2D array
            Dense templates with shape ``(data_size, num_flat)``.
        """
        flat_mask = np.isinf(self.prior_variances)
        return self.templates[self._data.model_mask, :][:, flat_mask].toarray()

    def project_precision(self, precision):
        """Marginalize the flat-prior coefficients of this correlation.

        Parameters
        ----------
        precision : 2D array
            Inverse of the masked covariance of this correlation, including the
            Gaussian-prior update, shape ``(data_size, data_size)``.

        Returns
        -------
        2D array
            Precision with the flat-prior templates projected out. Unchanged
            when no coefficient has a flat prior.
        """
        if not self.has_flat_templates:
            return precision

        precision, self.flat_log_normalization = project_flat_templates(
            precision, self.get_masked_flat_templates()
        )

        return precision

    def get_base_values(self, params):
        """Values of the marginalized coefficients used in the model itself.

        Additive broadband coefficients keep their parameter values in the model,
        and the marginalization is over deviations from them. Small-scale
        templates are not part of the model, so their base values are zero.

        Parameters
        ----------
        params : dict
            Model parameters.

        Returns
        -------
        1D array
            Base values with shape ``(num_coefficients,)``.
        """
        return np.array(
            [params.get(coefficient_id, 0.0) for coefficient_id in self.coefficient_ids]
        )

    def _compute_cov_update(self, family):
        """Compute ``templates @ Sigma @ templates.T`` within the fitted bins.

        The templates scaled by the prior width are decomposed with an SVD, and
        combinations with negligible singular values are dropped. These are
        degenerate within the fitted bins and would only add numerical noise.

        Parameters
        ----------
        family : TemplateFamily
            Template family; its ``num_modes`` is set to the number of retained modes.

        Returns
        -------
        2D array
            Covariance update with shape ``(data_size, data_size)``.
        """
        scaled_templates = family.templates * family.prior_sigma
        scaled_templates = scaled_templates[self._data.model_mask, :].toarray()

        print(
            f"  There are {family.templates.shape[1]} templates. "
            "SVD of template matrix to remove degenerate modes."
        )
        left_vectors, singular_values, _ = np.linalg.svd(scaled_templates, full_matrices=False)
        mode_mask = singular_values > DEGENERATE_MODE_THRESHOLD * singular_values[0]
        left_vectors = left_vectors[:, mode_mask]
        singular_values = singular_values[mode_mask]

        print(f"  There are {mode_mask.sum()} remaining modes for marginalization.")
        family.num_modes = mode_mask.sum()

        return np.dot(left_vectors * singular_values**2, left_vectors.T)


class AnalyticMarginalization:
    """Joint handling of the marginalized linear coefficients of all correlations.

    The likelihood needs nothing from this class: the covariance of each
    correlation, and the global covariance if present, already contain the
    covariance updates. This class recovers the best-fit coefficients from a
    residual, either once after a fit or at every likelihood evaluation.

    The coefficients are the solution of the joint normal equations of all
    templates with their priors (see ``solve_coefficients``), so one matrix, built
    on first use, maps a residual to all coefficients.
    """

    def __init__(self, interface):
        """Collect the correlations that have marginalized templates.

        Parameters
        ----------
        interface : vega.vega_interface.VegaInterface
            Initialized interface, with data and (optionally) the global covariance read.
        """
        self._interface = interface
        self.components = {
            name: data.marginalization
            for name, data in interface.data.items()
            if data is not None and data.marginalization.families
        }

        self._coefficient_matrices = None
        self._global_coefficient_matrix = None
        self._coefficient_errors = None
        self._global_weighted_templates = None

    def validate_configuration(self):
        """Check the marginalized coefficients against the sampled and scanned parameters.

        Raises
        ------
        ValueError
            If a marginalized broadband coefficient is sampled, scanned or has an entry
            in ``[priors]``, or if ``bao_amp`` is sampled or scanned while broadbands
            are marginalized.
        """
        interface = self._interface

        varied_parameters = set(interface.sample_params["limits"])
        if interface.mc_config is not None:
            varied_parameters |= set(interface.mc_config["sample"]["limits"])
        if "chi2 scan" in interface.main_config:
            varied_parameters |= set(interface.main_config["chi2 scan"])

        broadband_ids = set()
        for name, component in self.components.items():
            for family in component.families:
                if family.name == "broadband":
                    broadband_ids |= set(family.ids)

                    model = interface.models[name]
                    has_multiplicative = model.broadband is not None and (
                        model.broadband.bb_terms["pre-mul"] or model.broadband.bb_terms["post-mul"]
                    )
                    if has_multiplicative:
                        print(
                            f"WARNING: {name} has multiplicative broadbands. They are not "
                            "applied to the analytically marginalized additive broadbands."
                        )

        if not broadband_ids:
            return

        conflicting_ids = broadband_ids & (varied_parameters | set(interface.priors))
        if conflicting_ids:
            raise ValueError(
                "Analytically marginalized broadband coefficients cannot appear in [sample], "
                f"[chi2 scan] or [priors]: {sorted(conflicting_ids)}. Use broadband-prior-sigma "
                "in the component config to set their prior."
            )

        if "bao_amp" in varied_parameters:
            raise ValueError(
                "bao_amp cannot be sampled or scanned while additive broadbands are "
                "analytically marginalized: their templates are built with a fixed bao_amp."
            )

    def update_global_covariance(self):
        """Add the covariance updates of all correlations to the global covariance.

        Marginalizes the Gaussian-prior coefficients of every template family in
        the block of each correlation of ``interface.global_cov``, in place. The
        action of the original inverse covariance on the templates is stored
        first, for the solution of the best-fit coefficients.
        """
        interface = self._interface
        if not self.components:
            return

        # The best-fit coefficients are solved with the covariance before the
        # update, which is overwritten below: keep its action on the templates.
        original_masked_covariance = interface.global_cov[
            np.ix_(interface.full_data_mask, interface.full_data_mask)
        ]
        self._global_weighted_templates = solve_with_covariance(
            original_masked_covariance, self._get_joint_masked_templates()
        )
        del original_masked_covariance

        if all(component.cov_update is None for component in self.components.values()):
            return

        print("Updating global covariance with marginalization templates.")
        block_start = 0
        for name in interface.corr_items:
            data = interface.data[name]
            block_stop = block_start + data.full_data_size

            if name in self.components and self.components[name].cov_update is not None:
                component = self.components[name]
                covariance_block = interface.global_cov[
                    block_start:block_stop, block_start:block_stop
                ]
                covariance_block[np.ix_(data.data_mask, data.data_mask)] += component.cov_update

                if interface.low_mem_mode:
                    component.cov_update = None

            block_start = block_stop

    def check_direct_model(self):
        """Refuse a direct power spectrum model with Gaussian broadband priors.

        The broadband templates and their covariance update include the
        ``(1 + bao_amp)`` factor of the peak/smooth decomposition. A model computed
        directly from the full power spectrum does not have this factor, so the
        prior would act with a width ``(1 + bao_amp)`` times the configured one.
        Flat priors are not affected.

        Raises
        ------
        ValueError
            If any correlation has marginalized broadbands with a Gaussian prior
            and an amplitude factor different from one.
        """
        for name, component in self.components.items():
            for family in component.families:
                if (
                    family.name == "broadband"
                    and not family.has_flat_prior
                    and family.amplitude_factor != 1.0
                ):
                    raise ValueError(
                        f"{name}: analytically marginalized broadbands with a Gaussian prior "
                        "(broadband-prior-sigma) cannot be used with a model computed "
                        "directly from the full power spectrum. Use a flat prior."
                    )

    def project_global_precision(self):
        """Marginalize the flat-prior coefficients of all correlations in the global precision.

        Updates ``masked_global_invcov`` and ``masked_global_log_cov_det`` of the
        interface in place. Does nothing when no coefficient has a flat prior.
        """
        interface = self._interface
        if not any(component.has_flat_templates for component in self.components.values()):
            return

        flat_template_blocks = []
        for name in interface.corr_items:
            data = interface.data[name]
            if name in self.components and self.components[name].has_flat_templates:
                flat_template_blocks.append(self.components[name].get_masked_flat_templates())
            else:
                flat_template_blocks.append(np.zeros((data.data_size, 0)))

        joint_flat_templates = dense_block_diag(*flat_template_blocks)
        projected_precision, log_normalization = project_flat_templates(
            interface.masked_global_invcov, joint_flat_templates
        )

        interface.masked_global_invcov = projected_precision
        interface.masked_global_log_cov_det += log_normalization

    @property
    def num_other_coefficients(self):
        """Number of marginalized coefficients that are not small-scale modes.

        Returns
        -------
        dict
            Number of such coefficients keyed by correlation name, for all correlations.
        """
        return {
            name: self.components[name].num_other_coefficients if name in self.components else 0
            for name in self._interface.corr_items
        }

    @property
    def num_coefficients(self):
        """Number of marginalized coefficients of each correlation.

        Returns
        -------
        dict
            Number of coefficients keyed by correlation name, for all correlations.
        """
        return {
            name: self.components[name].num_coefficients if name in self.components else 0
            for name in self._interface.corr_items
        }

    def _get_joint_masked_templates(self):
        """Templates of all correlations within the fitted bins, as one matrix.

        Returns
        -------
        2D array
            Dense block-diagonal template matrix with shape
            ``(total_data_size, total_num_coefficients)``. Correlations without
            marginalized templates contribute rows and no columns.
        """
        interface = self._interface

        template_blocks = []
        for name in interface.corr_items:
            data = interface.data[name]
            if name in self.components:
                masked_templates = self.components[name].templates[data.model_mask, :]
                template_blocks.append(masked_templates.toarray())
            else:
                template_blocks.append(np.zeros((data.data_size, 0)))

        return dense_block_diag(*template_blocks)

    def _build_coefficient_matrices(self):
        """Build the matrices mapping a masked residual to best-fit coefficients.

        Without a global covariance each correlation has its own matrix with shape
        ``(num_coefficients, data_size)``. With a global covariance one joint matrix
        with shape ``(total_num_coefficients, total_data_size)`` accounts for the
        covariance between correlations.
        """
        interface = self._interface

        if interface._use_global_cov:
            prior_variances = np.concatenate(
                [component.prior_variances for component in self.components.values()]
            )
            self._global_coefficient_matrix, joint_errors = solve_coefficients(
                self._get_joint_masked_templates(),
                self._global_weighted_templates,
                prior_variances,
            )

            self._coefficient_errors = {}
            start = 0
            for name, component in self.components.items():
                stop = start + component.num_coefficients
                self._coefficient_errors[name] = joint_errors[start:stop]
                start = stop
            return

        self._coefficient_matrices = {}
        self._coefficient_errors = {}
        for name, component in self.components.items():
            data = interface.data[name]
            if data.cov_mat_org is None:
                raise ValueError(
                    "The marginalized coefficients need the original covariance, "
                    "which is not kept in low-memory mode without a global covariance."
                )

            masked_templates = component.templates[data.model_mask, :].toarray()
            original_masked_covariance = data.cov_mat_org[np.ix_(data.data_mask, data.data_mask)]
            weighted_templates = solve_with_covariance(original_masked_covariance, masked_templates)

            self._coefficient_matrices[name], self._coefficient_errors[name] = solve_coefficients(
                masked_templates, weighted_templates, component.prior_variances
            )

    def get_coefficient_errors(self):
        """Uncertainties of the marginalized coefficients at fixed sampled parameters.

        They do not include the uncertainty of the sampled parameters.

        Returns
        -------
        dict
            Standard deviations keyed by correlation name, each with shape
            ``(num_coefficients,)``.
        """
        if not self.components:
            return {}
        if self._coefficient_matrices is None and self._global_coefficient_matrix is None:
            self._build_coefficient_matrices()

        return self._coefficient_errors

    def compute_coefficients(self, model_cf, direct=False):
        """Compute the best-fit marginalized coefficients for a given model.

        For additive broadbands the coefficients are deviations from the
        ``[parameters]`` values used in the model.

        Parameters
        ----------
        model_cf : dict
            Model correlation functions keyed by correlation name, on the full
            output grid and without any template contribution.
        direct : bool, optional
            True if the model was computed directly from the full power spectrum
            instead of the peak/smooth decomposition, by default False. Broadband
            coefficients are then rescaled to the normalization of that model.

        Returns
        -------
        dict
            Best-fit coefficients keyed by correlation name, each with shape
            ``(num_coefficients,)``, for the correlations with marginalized templates.
        """
        if not self.components:
            return {}

        interface = self._interface
        if self._coefficient_matrices is None and self._global_coefficient_matrix is None:
            self._build_coefficient_matrices()

        # A common rescaling of the updated covariance, as applied to Monte Carlo
        # mocks, leaves the conditional mean of the coefficients unchanged.
        if interface._use_global_cov:
            masked_model = np.concatenate(
                [model_cf[name][interface.data[name].model_mask] for name in interface.corr_items]
            )
            if interface.monte_carlo:
                masked_data = interface.analysis.current_mc_mock
            else:
                masked_data = np.concatenate(
                    [interface.data[name].masked_data_vec for name in interface.corr_items]
                )

            joint_coefficients = self._global_coefficient_matrix.dot(masked_data - masked_model)

            coefficients = {}
            start = 0
            for name, component in self.components.items():
                coefficients[name] = joint_coefficients[start : start + component.num_coefficients]
                start += component.num_coefficients
        else:
            coefficients = {}
            for name in self.components:
                data = interface.data[name]
                masked_data = data.masked_mc_mock if interface.monte_carlo else data.masked_data_vec
                residual = masked_data - model_cf[name][data.model_mask]
                coefficients[name] = self._coefficient_matrices[name].dot(residual)

        if direct:
            for name, component in self.components.items():
                coefficients[name] = coefficients[name] * component.amplitude_factors

        return coefficients

    def add_templates(self, model_cf, coefficients, direct=False):
        """Add the template contribution to the model correlation functions.

        Parameters
        ----------
        model_cf : dict
            Model correlation functions keyed by correlation name. Updated in place.
        coefficients : dict
            Coefficients keyed by correlation name, as returned by ``compute_coefficients``.
        direct : bool, optional
            Value of ``direct`` used when the coefficients were computed, by default False.

        Returns
        -------
        dict
            The updated ``model_cf``.
        """
        for name, component in self.components.items():
            if name in coefficients:
                template_coefficients = np.asarray(coefficients[name])
                if direct:
                    template_coefficients = template_coefficients / component.amplitude_factors
                model_cf[name] += component.templates.dot(template_coefficients)

        return model_cf
