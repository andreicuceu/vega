"""Opt-in linear nuisance elimination, independent of Vega's legacy covariance update.

The model convention is ``data = model + templates @ coefficients + noise``.
Flat priors use unit Lebesgue measure in the supplied coefficient coordinates;
their integrated likelihood is consequently not an absolute evidence.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag


@dataclass
class LinearResult:
    """Conditional coefficient posterior and the selected -2 log objective.

    ``profile`` includes the Gaussian prior penalty. ``objective`` additionally
    includes the integration volume when mode is ``integrate``. Data likelihood
    normalization is left to the caller. ``covariance`` is conditional on the
    nonlinear parameters, not their propagated uncertainty.
    """

    ids: tuple
    coefficients: np.ndarray
    covariance: np.ndarray
    profile: float
    objective: float
    logdet: float


def solve_linear(
    residual,
    templates,
    precision,
    ids=None,
    means=None,
    sigmas=None,
    mode="profile",
    rtol=None,
    whitener=None,
):
    """Profile or integrate jointly over linear coefficients using an augmented SVD.

    ``precision`` must be symmetric positive definite. A positive finite sigma
    specifies a normalized Gaussian prior; ``np.inf`` specifies a flat prior.
    No pseudoinverse or silent mode truncation is used: unidentifiable flat
    directions raise ValueError. Proper priors regularize degenerate templates.
    ``rtol`` is the relative singular-value rank threshold (machine precision
    times the largest matrix dimension by default).
    """
    if mode not in ("profile", "integrate"):
        raise ValueError("Linear nuisance mode must be profile or integrate")
    residual = np.asarray(residual, dtype=float)
    templates = np.asarray(templates, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if residual.ndim != 1 or templates.ndim != 2 or templates.shape[0] != residual.size:
        raise ValueError("Residual and template dimensions do not match")
    n = templates.shape[1]
    ids = tuple(range(n)) if ids is None else tuple(ids)
    means = np.zeros(n) if means is None else np.asarray(means, dtype=float)
    sigmas = np.full(n, np.inf) if sigmas is None else np.asarray(sigmas, dtype=float)
    if len(ids) != n or len(set(ids)) != n or means.shape != (n,) or sigmas.shape != (n,):
        raise ValueError("Coefficient IDs must be unique and priors must match the columns")
    if not np.all(np.isfinite(means)) or np.any(np.isnan(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Prior means must be finite and sigmas positive (or inf for flat)")
    if precision.shape != (residual.size, residual.size):
        raise ValueError("Precision dimensions do not match residual")
    if not all(np.all(np.isfinite(x)) for x in (residual, templates, precision)):
        raise ValueError("Residual, templates and precision must be finite")
    if not np.allclose(precision, precision.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Precision must be symmetric positive definite")
    if rtol is not None and (not np.isfinite(rtol) or not 0 < rtol < 1):
        raise ValueError("Rank rtol must be between zero and one")
    if whitener is None:
        whitener = np.linalg.cholesky(precision).T
    y = whitener @ (residual - templates @ means)
    proper = np.isfinite(sigmas)
    weights = 1 / sigmas
    design = np.vstack((whitener @ templates, np.diag(weights)))
    target = np.concatenate((y, np.zeros(n)))
    if not n:
        value = float(y @ y)
        return LinearResult(ids, np.empty(0), np.empty((0, 0)), value, value, 0.0)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    threshold = (np.finfo(float).eps * max(design.shape) if rtol is None else rtol) * s[0]
    if np.any(s <= threshold):
        raise ValueError(
            "Unidentifiable linear nuisance modes: remove redundant flat templates "
            "or specify finite Gaussian priors"
        )
    delta = vt.T @ ((u.T @ target) / s)
    covariance = (vt.T / s**2) @ vt
    error = target - design @ delta
    profile = float(error @ error)
    logdet = float(2 * np.log(s).sum())
    objective = profile
    if mode == "integrate":
        objective += logdet + 2 * np.log(sigmas[proper]).sum()
        objective -= (~proper).sum() * np.log(2 * np.pi)
    return LinearResult(ids, means + delta, covariance, profile, float(objective), logdet)


class LinearNuisance:
    """Phase 1 adapter for small-bin modes and additive polynomial broadbands.

    Each local column has a stable string ID; aliases can deliberately share a
    coefficient across components. All columns are assembled before applying
    the full joint precision. No physics parameter is selected automatically.
    """

    def __init__(self, interface):
        self.interface = interface
        config = interface.main_config
        settings = config["linear nuisance"]
        self.mode = settings.get("mode", "profile")
        if self.mode not in ("profile", "integrate"):
            raise ValueError("Linear nuisance mode must be profile or integrate")
        if not interface._has_data or interface.model_pk:
            raise ValueError("Linear nuisance elimination requires correlation-function data")
        self.rtol = settings.getfloat("rank-rtol", None)
        self.aliases = (
            dict(config["linear nuisance ids"]) if "linear nuisance ids" in config else {}
        )
        self.priors = {}
        if "linear nuisance priors" in config:
            for key, value in config["linear nuisance priors"].items():
                mean, sigma = map(float, value.split())
                self.priors[key] = (mean, sigma)
        self.bb_names = {}
        self.defaults = {}
        self.small = {}
        local_ids = set()
        for name, model in interface.models.items():
            item = interface.corr_items[name]
            data = interface.data[name]
            if item.marginalize_small_scales:
                # The established selection and mask expansion were performed by Data.
                raw = item.get_undist_xi_marg_templates().toarray()
                if raw.shape[1] == 0:
                    raise ValueError(f"No small-scale templates selected for {name}")
                self.small[name] = raw
                for i in range(raw.shape[1]):
                    key = f"small-scale-{name} ({i})"
                    local_ids.add(key)
                    self.defaults[key] = (0.0, item.marginalize_small_scales_prior_sigma)
            self.bb_names[name] = []
            if settings.getboolean("broadbands", True) and model.broadband is not None:
                for position in ("pre-add", "post-add"):
                    for term in model.broadband.bb_terms[position]:
                        if term["func"] != "broadband":
                            continue
                        lo1, hi1, step1 = term["r1_config"]
                        lo2, hi2, step2 = term["r2_config"]
                        # The legacy evaluator reshapes this axis assuming step=1.
                        if step1 != 1:
                            raise ValueError(
                                "Phase 1 additive broadband requires first-axis step=1"
                            )
                        for i in range(lo1, hi1 + 1, step1):
                            for j in range(lo2, hi2 + 1, step2):
                                key = f"{term['name']} ({i},{j})"
                                self.bb_names[name].append((key, position))
                                local_ids.add(key)
                                self.defaults[key] = (0.0, np.inf)
            if (
                data.is_direct_multipoles
                or data.use_multipoles
                or getattr(data, "_rmu_binning", False)
            ):
                raise ValueError("Phase 1 linear nuisance adapter supports rp/rt data only")
        unknown = set(self.aliases) - local_ids
        if unknown:
            raise ValueError(f"Unknown local nuisance IDs: {sorted(unknown)}")
        self.ids = tuple(sorted({self.aliases.get(key, key) for key in local_ids}))
        if not self.ids:
            raise ValueError("Linear nuisance enabled but no phase 1 templates selected")
        if set(self.priors) - set(self.ids):
            raise ValueError("Linear nuisance priors must refer to final (possibly aliased) IDs")
        sampled = set(interface.sample_params["limits"])
        if interface.mc_config is not None:
            sampled |= set(interface.mc_config["sample"]["limits"])
        eliminated = local_ids | set(self.ids)
        scanned = set(config["chi2 scan"]) if "chi2 scan" in config else set()
        if (
            sampled.intersection(eliminated)
            or set(interface.priors).intersection(eliminated)
            or scanned.intersection(eliminated)
        ):
            raise ValueError(
                "Eliminated coefficients cannot appear in [sample], [priors] or [chi2 scan]; "
                "use [linear nuisance priors]"
            )
        defaults = {}
        for key, prior in self.defaults.items():
            shared = self.aliases.get(key, key)
            if shared in defaults and defaults[shared] != prior and shared not in self.priors:
                raise ValueError(f"Shared coefficient {shared} requires an explicit common prior")
            defaults[shared] = prior
        self.means, self.sigmas = np.array(
            [self.priors.get(key, defaults[key]) for key in self.ids]
        ).T
        if (
            not np.all(np.isfinite(self.means))
            or np.any(np.isnan(self.sigmas))
            or np.any(self.sigmas <= 0)
        ):
            raise ValueError("Linear nuisance priors require finite means and positive sigmas")
        self.last_result = None
        self._precision = None
        self._whitener = None

    def base_params(self, params):
        """Remove analytically eliminated additive coefficients from the base model."""
        params = dict(params)
        for terms in self.bb_names.values():
            for key, _ in terms:
                params[key] = 0.0
        return params

    def templates(self, params, direct=False):
        """Build full output-grid responses through the current distortion path."""
        output = {}
        for name, model in self.interface.models.items():
            data = self.interface.data[name]
            columns = np.zeros((data.model_mask.size, len(self.ids)))
            bb = model.broadband
            post = 1.0 if bb is None else bb.compute(params, "post-mul")

            def distort(column, current_model=model, current_data=data):
                if current_model._has_distortion_mat:
                    return current_data.distortion_mat @ column
                return column

            for i, column in enumerate(self.small.get(name, np.empty((0, 0))).T):
                key = f"small-scale-{name} ({i})"
                columns[:, self.ids.index(self.aliases.get(key, key))] += distort(column) * post
            for key, position in self.bb_names[name]:
                unit = self.base_params(params)
                unit[key] = 1.0
                term = next(
                    term for term in bb.bb_terms[position] if key.startswith(term["name"] + " (")
                )
                coordinates = (
                    bb.model_coordinates if position == "pre-add" else bb.dist_model_coordinates
                )
                column = bb._compute_broadband(term, unit, coordinates)
                if position == "pre-add":
                    column = distort(column) * post
                column = column * (1.0 if direct else 1.0 + params["bao_amp"])
                columns[:, self.ids.index(self.aliases.get(key, key))] += column
            output[name] = columns
        return output

    def evaluate(self, model, params, direct=False):
        """Solve against the same masked data and precision as the Vega likelihood."""
        interface = self.interface
        templates = self.templates(params, direct)
        residuals, columns, precisions = [], [], []
        for name in interface.corr_items:
            data = interface.data[name]
            observed = (
                data.masked_mc_mock
                if interface.monte_carlo and not interface._use_global_cov
                else data.masked_data_vec
            )
            residuals.append(observed - model[name][data.model_mask])
            columns.append(templates[name][data.model_mask])
            if not interface._use_global_cov:
                precisions.append(
                    data.scaled_inv_masked_cov if interface.monte_carlo else data.inv_masked_cov
                )
        residual = np.concatenate(residuals)
        if interface._use_global_cov:
            precision = interface.masked_global_invcov
            if interface.monte_carlo:
                residual = interface.analysis.current_mc_mock - np.concatenate(
                    [model[name][interface.data[name].model_mask] for name in interface.corr_items]
                )
        else:
            precision = block_diag(*precisions)
        if self._precision is None or not np.array_equal(precision, self._precision):
            self._precision = precision.copy()
            self._whitener = np.linalg.cholesky(precision).T
        result = solve_linear(
            residual,
            np.vstack(columns),
            precision,
            self.ids,
            self.means,
            self.sigmas,
            self.mode,
            self.rtol,
            self._whitener,
        )
        self.last_result = result
        self.last_templates = templates
        return result
