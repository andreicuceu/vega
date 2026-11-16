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
from scipy.sparse import block_diag, csr_array, hstack

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

            r1_min, r1_max, r1_step = term["r1_config"]
            r2_min, r2_max, r2_step = term["r2_config"]
            term_ids = [
                f"{term['name']} ({i},{j})"
                for i in np.arange(r1_min, r1_max + 1, r1_step)
                for j in np.arange(r2_min, r2_max + 1, r2_step)
            ]

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
    coefficient_matrix : 2D array
        Matrix ``inv(F.T P F) F.T P`` giving the best-fit flat-prior coefficients
        from a residual, with shape ``(num_flat, data_size)``.
    log_normalization : float
        ``log det(F.T P F) - num_flat * log(2 pi)``, the term that integrating
        over the coefficients adds to minus twice the log-likelihood.
    coefficient_variances : 1D array
        Diagonal of ``inv(F.T P F)``, the variance of each flat-prior coefficient
        at fixed sampled parameters, with shape ``(num_flat,)``.

    Raises
    ------
    ValueError
        If the templates are linearly dependent within the fitted bins.
    """
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
    coefficient_matrix = scaled_coefficient_matrix / template_norms[:, None]

    log_det_normal_matrix = np.sum(np.log(eigenvalues)) + 2 * np.sum(np.log(template_norms))
    log_normalization = log_det_normal_matrix - flat_templates.shape[1] * np.log(2 * np.pi)

    inverse_normal_matrix = cho_solve(cholesky_factor, np.eye(flat_templates.shape[1]))
    coefficient_variances = np.diag(inverse_normal_matrix) / template_norms**2

    return projected_precision, coefficient_matrix, log_normalization, coefficient_variances


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
    flat_coefficient_matrix : 2D array or None
        Matrix giving the best-fit flat-prior coefficients of this correlation from
        its residual, shape ``(num_flat, data_size)``. Set when the precision of
        the correlation is first computed.
    flat_log_normalization : float
        Term added to the log-determinant of the covariance by the flat-prior
        coefficients; zero without them.
    flat_coefficient_variances : 1D array or None
        Variance of each flat-prior coefficient at fixed sampled parameters,
        ignoring the covariance between correlations.
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
        self.flat_coefficient_matrix = None
        self.flat_log_normalization = 0.0
        self.flat_coefficient_variances = None

        if corr_item.marginalize_small_scales:
            self.families.append(build_small_scale_family(data))

            # The masks must be final before the templates are restricted to the fitted bins
            if corr_item.fit_marg_scales:
                include_marginalized_scales_in_fit(data)

        if corr_item.config["model"].getboolean("marginalize-broadbands", False):
            self.families.append(build_broadband_family(data))

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
        if len(self.families) == 1:
            return self.families[0].templates

        return hstack([family.templates for family in self.families], format="csr")

    @property
    def prior_variances(self):
        """Prior variance of each marginalized coefficient.

        Returns
        -------
        1D array
            Prior variances with shape ``(num_coefficients,)``; infinite for flat priors.
        """
        return np.concatenate(
            [np.full(len(family.ids), family.prior_sigma**2) for family in self.families]
        )

    @property
    def amplitude_factors(self):
        """Factor included in each template that is absent for a direct power spectrum.

        Returns
        -------
        1D array
            Amplitude factors with shape ``(num_coefficients,)``.
        """
        return np.concatenate(
            [np.full(len(family.ids), family.amplitude_factor) for family in self.families]
        )

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

        (
            precision,
            self.flat_coefficient_matrix,
            self.flat_log_normalization,
            self.flat_coefficient_variances,
        ) = project_flat_templates(precision, self.get_masked_flat_templates())

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

    For Gaussian priors the best-fit coefficients are the conditional mean

        coefficients = Sigma @ templates.T @ updated_precision @ (data - model),

    so one matrix, built on first use, maps a residual to all coefficients.
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
        self._global_flat_coefficient_matrix = None
        self._global_flat_coefficient_variances = None
        self._coefficient_errors = None

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
        projected_precision, coefficient_matrix, log_normalization, coefficient_variances = (
            project_flat_templates(interface.masked_global_invcov, joint_flat_templates)
        )

        interface.masked_global_invcov = projected_precision
        interface.masked_global_log_cov_det += log_normalization
        self._global_flat_coefficient_matrix = coefficient_matrix
        self._global_flat_coefficient_variances = coefficient_variances

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

    def _build_coefficient_matrices(self):
        """Build the matrices mapping a masked residual to best-fit coefficients.

        Without a global covariance each correlation has its own matrix with shape
        ``(num_coefficients, data_size)``. With a global covariance one joint matrix
        with shape ``(total_num_coefficients, total_data_size)`` accounts for the
        covariance between correlations.
        """
        interface = self._interface

        if interface._use_global_cov:
            # Templates of correlations without marginalization have zero columns
            masked_templates = []
            prior_variances = []
            for name in interface.corr_items:
                data = interface.data[name]
                if name in self.components:
                    component = self.components[name]
                    masked_templates.append(component.templates[data.model_mask, :])
                    prior_variances.append(component.prior_variances)
                else:
                    masked_templates.append(csr_array((data.data_size, 0)))

            joint_templates = block_diag(masked_templates, format="csr")
            prior_variances = np.concatenate(prior_variances)
            self._global_coefficient_matrix = self._conditional_mean_matrix(
                joint_templates,
                prior_variances,
                interface.masked_global_invcov,
                self._global_flat_coefficient_matrix,
            )
            joint_errors = self._conditional_errors(
                joint_templates,
                prior_variances,
                self._global_coefficient_matrix,
                self._global_flat_coefficient_variances,
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
            # Computing the precision also sets the flat-prior coefficient matrix
            precision = data.inv_masked_cov
            masked_templates = component.templates[data.model_mask, :]
            self._coefficient_matrices[name] = self._conditional_mean_matrix(
                masked_templates,
                component.prior_variances,
                precision,
                component.flat_coefficient_matrix,
            )
            self._coefficient_errors[name] = self._conditional_errors(
                masked_templates,
                component.prior_variances,
                self._coefficient_matrices[name],
                component.flat_coefficient_variances,
            )

    @staticmethod
    def _conditional_errors(
        masked_templates, prior_variances, coefficient_matrix, flat_coefficient_variances
    ):
        """Compute the uncertainty of each coefficient at fixed sampled parameters.

        For a Gaussian prior the posterior variance of coefficient ``i`` is
        ``sigma_i**2 * (1 - (coefficient_matrix @ templates)[i, i])``. For a flat
        prior it is the diagonal of ``inv(F.T P F)``.

        Parameters
        ----------
        masked_templates : scipy.sparse.csr_array
            Templates within the fitted bins, shape ``(data_size, num_coefficients)``.
        prior_variances : 1D array
            Prior variance of each coefficient; infinite for flat priors.
        coefficient_matrix : 2D array
            Matrix from ``_conditional_mean_matrix``, shape ``(num_coefficients, data_size)``.
        flat_coefficient_variances : 1D array or None
            Variances of the flat-prior coefficients, shape ``(num_flat,)``.

        Returns
        -------
        1D array
            Standard deviation of each coefficient, shape ``(num_coefficients,)``.
        """
        flat_mask = np.isinf(prior_variances)

        # Fraction of the prior variance removed by the data, per coefficient
        data_fraction = np.asarray(
            masked_templates.multiply(coefficient_matrix.T).sum(axis=0)
        ).ravel()
        variances = np.where(flat_mask, 0.0, prior_variances) * (1 - data_fraction)
        if flat_mask.any():
            variances[flat_mask] = flat_coefficient_variances

        # Round-off can leave tiny negative variances for unconstrained coefficients
        return np.sqrt(np.clip(variances, 0.0, None))

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

    @staticmethod
    def _conditional_mean_matrix(
        masked_templates, prior_variances, precision, flat_coefficient_matrix
    ):
        """Build the matrix mapping a residual to the best-fit coefficients.

        Gaussian-prior coefficients are the conditional mean
        ``Sigma @ templates.T @ precision @ residual``, where the precision already
        has the flat-prior templates projected out. Flat-prior coefficients are the
        generalized least-squares solution stored when the projection was done.

        Parameters
        ----------
        masked_templates : scipy.sparse.csr_array
            Templates within the fitted bins, shape ``(data_size, num_coefficients)``.
        prior_variances : 1D array
            Prior variance of each coefficient; infinite for flat priors.
        precision : 2D array
            Marginalized precision matrix, shape ``(data_size, data_size)``.
        flat_coefficient_matrix : 2D array or None
            Rows for the flat-prior coefficients, shape ``(num_flat, data_size)``.

        Returns
        -------
        2D array
            Coefficient matrix with shape ``(num_coefficients, data_size)``.
        """
        flat_mask = np.isinf(prior_variances)
        gaussian_variances = np.where(flat_mask, 0.0, prior_variances)

        coefficient_matrix = gaussian_variances[:, None] * masked_templates.T.dot(precision)
        if flat_mask.any():
            coefficient_matrix[flat_mask, :] = flat_coefficient_matrix

        return coefficient_matrix

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
