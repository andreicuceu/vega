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

This module keeps track of the templates of each correlation, propagates them
through the distortion matrix and the data masks, builds the covariance update,
and recovers the best-fit coefficients from a residual when they are requested.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import block_diag, csr_array

# Singular values below this fraction of the largest one are treated as
# degenerate template combinations and dropped from the covariance update.
DEGENERATE_MODE_THRESHOLD = 1e-8


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
        units of the correlation function.
    num_modes : int
        Number of non-degenerate template combinations within the fitted bins.
        Set when the covariance update of the family is computed.
    """

    name: str
    ids: tuple
    templates: csr_array
    prior_sigma: float
    num_modes: int = 0


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

        if corr_item.marginalize_small_scales:
            self.families.append(build_small_scale_family(data))

            # The masks must be final before the templates are restricted to the fitted bins
            if corr_item.fit_marg_scales:
                include_marginalized_scales_in_fit(data)

        if not self.families:
            return

        for family in self.families:
            family_cov_update = self._compute_cov_update(family)
            if self.cov_update is None:
                self.cov_update = family_cov_update
            else:
                self.cov_update += family_cov_update

            if family.name == "small-scale":
                self.num_small_scale_modes = family.num_modes

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

        return csr_array(np.hstack([family.templates.toarray() for family in self.families]))

    @property
    def prior_variances(self):
        """Prior variance of each marginalized coefficient.

        Returns
        -------
        1D array
            Prior variances with shape ``(num_coefficients,)``.
        """
        return np.concatenate(
            [np.full(len(family.ids), family.prior_sigma**2) for family in self.families]
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
            projection = joint_templates.T.dot(interface.masked_global_invcov)
            self._global_coefficient_matrix = prior_variances[:, None] * projection
            return

        self._coefficient_matrices = {}
        for name, component in self.components.items():
            data = interface.data[name]
            masked_templates = component.templates[data.model_mask, :]
            projection = masked_templates.T.dot(data.inv_masked_cov)
            self._coefficient_matrices[name] = component.prior_variances[:, None] * projection

    def compute_coefficients(self, model_cf):
        """Compute the best-fit marginalized coefficients for a given model.

        Parameters
        ----------
        model_cf : dict
            Model correlation functions keyed by correlation name, on the full
            output grid and without any template contribution.

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

            return coefficients

        coefficients = {}
        for name in self.components:
            data = interface.data[name]
            masked_data = data.masked_mc_mock if interface.monte_carlo else data.masked_data_vec
            residual = masked_data - model_cf[name][data.model_mask]
            coefficients[name] = self._coefficient_matrices[name].dot(residual)

        return coefficients

    def add_templates(self, model_cf, coefficients):
        """Add the template contribution to the model correlation functions.

        Parameters
        ----------
        model_cf : dict
            Model correlation functions keyed by correlation name. Updated in place.
        coefficients : dict
            Coefficients keyed by correlation name, as returned by ``compute_coefficients``.

        Returns
        -------
        dict
            The updated ``model_cf``.
        """
        for name, component in self.components.items():
            if name in coefficients:
                model_cf[name] += component.templates.dot(coefficients[name])

        return model_cf
