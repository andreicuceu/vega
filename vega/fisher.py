"""Small fixed-covariance, configuration-space Fisher forecasts for Vega.

No fitting, mock generation, covariance derivatives, or analytic nuisance elimination
is performed here. All derivatives use Vega's final model and blinding machinery.
"""

import json
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from scipy.linalg import cho_solve, cholesky, solve_triangular

from vega.utils import find_file


@dataclass
class FisherResult:
    """Forecast in ``parameters`` order; covariance/sigma are None if rank deficient.

    ``jacobian`` has shape (selected data bins, parameters). ``null_directions``
    contains unit-length parameter-space columns. Rank and condition number use
    the diagonally normalized Fisher matrix to reduce parameter-unit dependence.
    ``step_changes`` compares h and h/2 derivatives in the covariance metric.
    """

    parameters: tuple
    fiducial: np.ndarray
    steps: np.ndarray
    model: np.ndarray
    jacobian: np.ndarray
    fisher_data: np.ndarray
    fisher_prior: np.ndarray
    fisher: np.ndarray
    covariance: object
    sigma: object
    rank: int
    eigenvalues: np.ndarray
    null_directions: np.ndarray
    condition_number: float
    step_changes: np.ndarray
    metadata: dict

    def save(self, filename):
        """Save a pickle-free NPZ, refusing to overwrite an existing file.

        Missing covariance/sigma are stored as empty arrays; metadata is JSON.
        """
        payload = dict(vars(self))
        payload["parameters"] = np.asarray(self.parameters)
        payload["metadata"] = json.dumps(self.metadata)
        for name in ("covariance", "sigma"):
            if payload[name] is None:
                payload[name] = np.array([])
        with open(filename, "xb") as stream:
            np.savez_compressed(stream, **payload)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _matrix(value, size, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0):
        raise ValueError(f"{name} must be symmetric.")
    try:
        factor = cholesky((matrix + matrix.T) / 2, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite; no pseudoinverse is used.") from exc
    return factor


def _has_supplied_covariance(data):
    # Data inserts an identity matrix when CO is absent, so has_cov_mat alone
    # cannot distinguish a supplied covariance from that plotting convenience.
    config = data.corr_item.config["data"]
    if config.get("covariance-file"):
        return True
    if data.is_direct_multipoles:
        return False
    with fits.open(find_file(config.get("filename"))) as hdul:
        return "CO" in hdul[1].columns.names


def _whitener(vega, covariance, size):
    if covariance is not None:
        factor = _matrix(covariance, size, "Supplied selected covariance")
        source = "explicit selected covariance (no additional rescaling)"
    elif vega._use_global_cov:
        source = "configured global covariance (including Vega corrections)"
        if vega.low_mem_mode:
            # Vega has discarded C in low-memory mode. Factor its existing
            # precision: if P=L L^T, L^T J has Gram matrix J^T P J.
            factor = _matrix(vega.masked_global_invcov, size, "Global precision")
            return lambda values: factor.T @ values, source
        mask = vega.full_data_mask
        factor = _matrix(vega.global_cov[np.ix_(mask, mask)], size, "Global covariance")
    else:
        factors = []
        counts = []
        for name in vega.corr_items:
            data = vega.data[name]
            if not _has_supplied_covariance(data):
                raise ValueError(
                    f"No supplied covariance for {name}; refusing Data's identity fallback."
                )
            mask = data.data_mask
            count = int(mask.sum())
            counts.append(count)
            factors.append(_matrix(data.cov_mat[np.ix_(mask, mask)], count, f"Covariance {name}"))
        source = "configured independent covariance blocks (including Vega corrections)"

        def whiten_blocks(values):
            blocks = np.split(values, np.cumsum(counts)[:-1], axis=0)
            return np.concatenate(
                [
                    solve_triangular(factor, block, lower=True)
                    for factor, block in zip(factors, blocks)
                ],
                axis=0,
            )

        return whiten_blocks, source
    return lambda values: solve_triangular(factor, values, lower=True), source


def compute_fisher(
    vega,
    parameters=None,
    *,
    steps,
    fiducial=None,
    covariance=None,
    priors=None,
    check_steps=True,
    step_tolerance=0.05,
    rcond=1e-10,
):
    """Compute F = J.T C^-1 J plus independent Gaussian prior precision.

    Parameters
    ----------
    vega : VegaInterface
        Initialized from ordinary Vega INIs; configuration-space data required.
    parameters : sequence of str, optional
        Forecast order. Defaults to the sampled parameters. Others remain fixed.
        On blind data only parameters already validated in [sample] are allowed.
    steps : mapping
        Explicit positive, absolute central-difference step for every parameter.
        Both endpoints must lie inside any configured [sample] limits.
    fiducial : dict, optional
        Overrides to vega.params in the usual input (possibly blinded) coordinates.
        [sample] starting values and Monte Carlo settings are not used.
    covariance : array, optional
        Already selected joint covariance, ordered by corr_items, then data_mask
        within each component. No scaling or correction is added to this array.
        Otherwise use the configured global covariance, or independent data blocks.
    priors : dict, optional
        Mapping parameter -> (mean, sigma). Defaults to vega.priors; {} disables
        priors. Means do not affect curvature. Priors on fixed parameters contribute
        no curvature. Vega's parameter blinding is additive, so precision is unchanged.
    check_steps : bool
        Compare derivatives at h and h/2 and warn above step_tolerance. The h
        derivatives define the returned Fisher matrix in either case.
    step_tolerance : float
        Relative tolerance in the covariance-weighted derivative norm.
    rcond : float
        Relative eigenvalue cutoff on the diagonally normalized Fisher matrix.

    Returns
    -------
    FisherResult
        Covariance is returned only for full numerical rank. No Percival correction
        is applied to parameter covariance. No marginalize-in-fit support is provided.
    """
    if vega.model_pk or not vega._has_data:
        raise ValueError("Fisher forecasts require configuration-space data and models.")
    if vega.monte_carlo:
        raise ValueError("Use a VegaInterface with monte_carlo disabled for fixed covariance.")
    if vega.marginalize_in_fit:
        raise ValueError("Fisher does not yet support marginalize-in-fit.")
    names = tuple(vega.sample_params["limits"] if parameters is None else parameters)
    if not names or len(set(names)) != len(names):
        raise ValueError("Forecast parameters must be nonempty and unique.")
    if set(steps) != set(names):
        raise ValueError("steps must contain exactly the forecast parameter names.")
    step_values = np.array([_positive(steps[name], f"Step for {name}") for name in names])
    step_tolerance = _positive(step_tolerance, "step_tolerance")
    rcond = _positive(rcond, "rcond")
    if rcond >= 1:
        raise ValueError("rcond must be less than one.")
    base = dict(vega.params)
    if fiducial is not None:
        unknown = set(fiducial) - set(base)
        if unknown:
            raise ValueError(f"Unknown fiducial parameters: {sorted(unknown)}")
        base.update(fiducial)
    for name, step in zip(names, step_values):
        if name not in base or not np.isfinite(base[name]):
            raise ValueError(f"Missing or nonfinite fiducial parameter: {name}")
        if vega._blind and name not in vega.sample_params["limits"]:
            raise ValueError(f"Blind forecast parameter {name} must be validated in [sample].")
        value = base[name]
        if not (value - step < value < value + step):
            raise ValueError(f"Step for {name} is too small to change its fiducial value.")
        lower, upper = vega.sample_params["limits"].get(name, (None, None))
        if (lower is not None and value - step < lower) or (
            upper is not None and value + step > upper
        ):
            raise ValueError(f"Central difference for {name} crosses configured sample limits.")

    components = tuple(vega.corr_items)
    model_masks = [np.asarray(vega.data[name].model_mask) for name in components]
    counts = [int(np.count_nonzero(vega.data[name].data_mask)) for name in components]
    if any(mask.dtype != bool or mask.ndim != 1 for mask in model_masks):
        raise ValueError("Model masks must be one-dimensional boolean arrays.")
    if any(mask.sum() != count for mask, count in zip(model_masks, counts)) or not sum(counts):
        raise ValueError("Selected data and model sizes must agree and be nonempty.")
    whiten, source = _whitener(vega, covariance, sum(counts))

    prior_matrix = np.zeros((len(names), len(names)))
    selected_priors = {}
    for name, prior in (vega.priors if priors is None else priors).items():
        if name not in base:
            raise ValueError(f"Unknown prior parameter: {name}")
        mean, sigma = prior
        sigma = _positive(sigma, f"Prior sigma for {name}")
        if not np.isfinite(mean):
            raise ValueError(f"Nonfinite prior mean for {name}")
        if name in names:
            prior_matrix[names.index(name), names.index(name)] = sigma**-2
            selected_priors[name] = [float(mean), sigma]

    def evaluate(params):
        models = vega.compute_model(params, run_init=False)
        selected = []
        for name, mask in zip(components, model_masks):
            model = np.asarray(models[name])
            if model.shape != mask.shape:
                raise ValueError(f"Final model and model_mask shapes disagree for {name}.")
            selected.append(model[mask])
        vector = np.concatenate(selected)
        if not np.all(np.isfinite(vector)):
            raise ValueError("Selected model contains nonfinite values.")
        return vector

    nominal = evaluate(base)
    jacobian = np.empty((nominal.size, len(names)))
    step_changes = np.full(len(names), np.nan)
    try:
        for index, (name, step) in enumerate(zip(names, step_values)):

            def derivative(h, name=name):
                plus, minus = dict(base), dict(base)
                plus[name], minus[name] = base[name] + h, base[name] - h
                if not minus[name] < base[name] < plus[name]:
                    raise ValueError(f"Finite difference step for {name} is unresolvable.")
                return (evaluate(plus) - evaluate(minus)) / (2 * h)

            jacobian[:, index] = derivative(step)
            if check_steps:
                half = whiten(derivative(step / 2))
                delta = whiten(jacobian[:, index]) - half
                denominator = max(np.linalg.norm(half), np.finfo(float).tiny)
                step_changes[index] = np.linalg.norm(delta) / denominator
                if step_changes[index] > step_tolerance:
                    warnings.warn(
                        f"Fisher derivative for {name} changes by {step_changes[index]:.3g} "
                        "between h and h/2; check the step size.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
    finally:
        # Leave model caches/components at the requested fiducial, never at p-h.
        evaluate(base)

    weighted = whiten(jacobian)
    fisher_data = weighted.T @ weighted
    fisher = fisher_data + prior_matrix
    if not np.all(np.isfinite(fisher)):
        raise ValueError("Nonfinite Fisher matrix; check model derivatives and prior widths.")
    scales = np.ones(len(names))
    positive = np.diag(fisher) > 0
    scales[positive] = 1 / np.sqrt(np.diag(fisher)[positive])
    normalized = fisher * scales[:, None] * scales[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    identified = eigenvalues > rcond * max(eigenvalues[-1], 0)
    rank = int(identified.sum())
    null = scales[:, None] * eigenvectors[:, ~identified]
    if null.size:
        null /= np.linalg.norm(null, axis=0)
    covariance_out, sigma_out = None, None
    condition = float("inf")
    if rank == len(names):
        condition = float(eigenvalues[-1] / eigenvalues[0])
        factor = cholesky(normalized, lower=True)
        covariance_out = cho_solve((factor, True), np.eye(len(names)))
        covariance_out *= scales[:, None] * scales[None, :]
        sigma_out = np.sqrt(np.diag(covariance_out))
    else:
        warnings.warn(
            f"Fisher rank {rank}/{len(names)}; covariance and sigma are unavailable. "
            "Inspect null_directions or add justified priors.",
            RuntimeWarning,
            stacklevel=2,
        )
    return FisherResult(
        names,
        np.array([base[name] for name in names]),
        step_values,
        nominal,
        jacobian,
        fisher_data,
        prior_matrix,
        fisher,
        covariance_out,
        sigma_out,
        rank,
        eigenvalues,
        null,
        condition,
        step_changes,
        {
            "components": list(components),
            "selected_bin_counts": counts,
            "covariance_source": source,
            "priors": selected_priors,
            "blind": bool(vega._blind),
            "check_steps": bool(check_steps),
            "step_tolerance": step_tolerance,
            "rcond": rcond,
            "percival_correction_applied": False,
            "configured_percival_correction": float(vega._percival_corr),
        },
    )
