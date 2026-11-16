"""Tests of the analytic marginalization of linear templates on a small synthetic data set."""

import configparser

import numpy as np
import pytest
from astropy.io import fits

from vega.utils import find_file
from vega.vega_interface import VegaInterface

PRIOR_SIGMA = 0.3


def write_configs(tmp_path, global_cov=False, marginalize_in_fit=False, marginalize=True):
    """Write a 12-bin correlation, its component configs and a main config.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory receiving the FITS and INI files.
    global_cov : bool, optional
        If True, use two correlations with a correlated global covariance, by default False.
    marginalize_in_fit : bool, optional
        Value of the ``marginalize-in-fit`` control option, by default False.
    marginalize : bool, optional
        If True, marginalize the bins below ``rt = 40``, by default True.

    Returns
    -------
    pathlib.Path
        Path to the main config.
    """
    tmp_path.mkdir(exist_ok=True)

    rp_grid = np.repeat([25.0, 75.0, 125.0, 175.0], 3)
    rt_grid = np.tile([20.0, 60.0, 100.0], 4)
    num_bins = rp_grid.size

    covariance = np.eye(num_bins) * 0.01 + 0.001
    distortion = np.eye(num_bins) + 0.02
    data_vector = np.linspace(0.0, 0.01, num_bins)

    columns = [
        fits.Column(name="RP", format="D", array=rp_grid),
        fits.Column(name="RT", format="D", array=rt_grid),
        fits.Column(name="Z", format="D", array=np.full(num_bins, 2.33)),
        fits.Column(name="DA", format="D", array=data_vector),
        fits.Column(name="CO", format=f"{num_bins}D", array=covariance),
        fits.Column(name="DM", format=f"{num_bins}D", array=distortion),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    for key, value in {"RPMIN": 0.0, "RPMAX": 200.0, "RTMAX": 120.0, "NP": 4, "NT": 3}.items():
        hdu.header[key] = value
    data_path = tmp_path / "data.fits"
    hdu.writeto(data_path)

    component_paths = []
    for name in ("first", "second") if global_cov else ("first",):
        component = configparser.ConfigParser()
        component.optionxform = str
        model_options = {"z evol LYA": "bias_vs_z_std"}
        if marginalize:
            model_options |= {
                "marginalize-below-rtmax": "40",
                "marginalize-prior-sigma": str(PRIOR_SIGMA),
                "fit-marginalized-scales": "True",
            }
        component.read_dict(
            {
                "data": {
                    "name": name,
                    "filename": str(data_path),
                    "tracer1": "LYA",
                    "tracer1-type": "continuous",
                },
                "cuts": {
                    "r-min": "40",
                    "r-max": "220",
                    "rp-min": "0",
                    "rt-min": "0",
                    "rp-max": "200",
                    "rt-max": "120",
                },
                "model": model_options,
            }
        )
        component_path = tmp_path / f"{name}.ini"
        with component_path.open("w") as stream:
            component.write(stream)
        component_paths.append(str(component_path))

    main = configparser.ConfigParser()
    main.optionxform = str
    main.read(find_file("full_configs/main.ini"))
    main["data sets"]["ini files"] = " ".join(component_paths)
    main["output"]["filename"] = str(tmp_path / "result.fits")
    main["control"]["marginalize-in-fit"] = str(marginalize_in_fit)

    if global_cov:
        joint_covariance = np.block(
            [[covariance, 0.1 * covariance], [0.1 * covariance, covariance]]
        )
        global_path = tmp_path / "global.fits"
        fits.BinTableHDU.from_columns(
            [fits.Column(name="COV", format=f"{2 * num_bins}D", array=joint_covariance)]
        ).writeto(global_path)
        main["data sets"]["global-cov-file"] = str(global_path)

    main_path = tmp_path / "main.ini"
    with main_path.open("w") as stream:
        main.write(stream)

    return main_path


def joint_problem(interface, model_cf):
    """Assemble the joint residual, templates and original covariance of an interface.

    Parameters
    ----------
    interface : VegaInterface
        Interface with small-scale marginalization.
    model_cf : dict
        Model correlation functions keyed by correlation name.

    Returns
    -------
    residual : 1D array
        Masked data minus model for all correlations, shape ``(total_data_size,)``.
    templates : 2D array
        Masked templates, block diagonal over correlations, shape
        ``(total_data_size, total_num_coefficients)``.
    original_covariance : 2D array
        Masked covariance before the marginalization update.
    """
    residuals, template_blocks, covariance_blocks = [], [], []
    for name in interface.corr_items:
        data = interface.data[name]
        residuals.append(data.masked_data_vec - model_cf[name][data.model_mask])
        template_blocks.append(data.marginalization.templates[data.model_mask, :].toarray())
        covariance_blocks.append(data.cov_mat_org[np.ix_(data.data_mask, data.data_mask)])

    num_coefficients = sum(block.shape[1] for block in template_blocks)
    total_size = sum(block.shape[0] for block in template_blocks)
    templates = np.zeros((total_size, num_coefficients))
    original_covariance = np.zeros((total_size, total_size))
    row, column = 0, 0
    for template_block, covariance_block in zip(template_blocks, covariance_blocks):
        size, num_columns = template_block.shape
        templates[row : row + size, column : column + num_columns] = template_block
        original_covariance[row : row + size, row : row + size] = covariance_block
        row += size
        column += num_columns

    if interface._use_global_cov:
        # The synthetic global covariance correlates the two identical blocks at 10%
        size = template_blocks[0].shape[0]
        original_covariance[:size, size:] = 0.1 * covariance_blocks[0]
        original_covariance[size:, :size] = 0.1 * covariance_blocks[0]

    return np.concatenate(residuals), templates, original_covariance


@pytest.mark.parametrize("global_cov", [False, True])
def test_chi2_and_coefficients_match_explicit_solution(tmp_path, global_cov):
    interface = VegaInterface(str(write_configs(tmp_path, global_cov=global_cov)))
    model_cf = interface.compute_model(run_init=False)
    residual, templates, original_covariance = joint_problem(interface, model_cf)
    num_coefficients = templates.shape[1]
    assert num_coefficients > 0

    # fit-marginalized-scales keeps the marginalized bins in the fit
    for data in interface.data.values():
        assert data.data_mask[0]
        assert data.effective_data_size == data.data_size - data.num_marg_modes

    # Explicit joint solution for the coefficients with their Gaussian prior
    original_precision = np.linalg.inv(original_covariance)
    prior_precision = np.eye(num_coefficients) / PRIOR_SIGMA**2
    normal_matrix = templates.T @ original_precision @ templates + prior_precision
    expected_coefficients = np.linalg.solve(
        normal_matrix, templates.T @ original_precision @ residual
    )

    # The chi2 with the updated covariance is the chi2 at the best-fit coefficients
    # including their prior penalty
    corrected_residual = residual - templates @ expected_coefficients
    expected_chi2 = (
        corrected_residual @ original_precision @ corrected_residual
        + expected_coefficients @ prior_precision @ expected_coefficients
    )
    prior_chi2 = interface.compute_prior_chi2()
    assert interface.chi2() - prior_chi2 == pytest.approx(expected_chi2, rel=1e-10)

    chi2, coefficients = interface.chi2(return_marg_coeff=True)
    assert chi2 == interface.chi2()
    joint_coefficients = np.concatenate([coefficients[name] for name in interface.corr_items])
    np.testing.assert_allclose(joint_coefficients, expected_coefficients, rtol=1e-8)

    # The log-likelihood is that of the residual under the updated covariance
    updated_covariance = original_covariance + PRIOR_SIGMA**2 * templates @ templates.T
    _, log_determinant = np.linalg.slogdet(updated_covariance)
    expected_log_lik = -0.5 * (
        expected_chi2 + log_determinant + residual.size * np.log(2 * np.pi) + prior_chi2
    )
    for prior in interface.priors.values():
        expected_log_lik += interface._gaussian_lik_prior(prior[1])
    assert interface.log_lik() == pytest.approx(expected_log_lik, rel=1e-10)

    # Passing the coefficients to compute_model adds the template contribution
    corrected_model = interface.compute_model(run_init=False, marg_coeff=coefficients)
    for name in interface.corr_items:
        data = interface.data[name]
        expected_model = model_cf[name] + data.marginalization.templates @ coefficients[name]
        np.testing.assert_allclose(corrected_model[name], expected_model)


def test_coefficients_at_every_evaluation_leave_likelihood_unchanged(tmp_path):
    once = VegaInterface(str(write_configs(tmp_path / "once")))
    every_call = VegaInterface(str(write_configs(tmp_path / "every", marginalize_in_fit=True)))

    assert once.chi2() == every_call.chi2()
    assert once.log_lik() == every_call.log_lik()

    # Coefficients are derived chain parameters only when computed at every evaluation
    assert sum(once.corr_num_marg_modes.values()) == 0
    num_coefficients = every_call.data["first"].marginalization.num_coefficients
    assert every_call.corr_num_marg_modes == {"first": num_coefficients}

    log_lik, coefficients = every_call.log_lik(return_marg_coeff=True)
    assert log_lik == every_call.log_lik()
    assert coefficients.shape == (num_coefficients,)


@pytest.mark.parametrize("global_cov", [False, True])
def test_bestfit_model_includes_templates(tmp_path, global_cov):
    interface = VegaInterface(str(write_configs(tmp_path, global_cov=global_cov)))
    interface.minimize()

    base_model = interface.compute_model(interface.minimizer.values, run_init=False)
    coefficients = interface.compute_marg_coeff(base_model)
    for name in interface.corr_items:
        data = interface.data[name]
        expected_model = base_model[name] + data.marginalization.templates @ coefficients[name]
        np.testing.assert_allclose(interface.bestfit_model[name], expected_model)
        np.testing.assert_allclose(
            interface.bestfit_corr_stats[name]["bestfit_marg_coeff"], coefficients[name]
        )


def test_no_marginalization_is_a_no_op(tmp_path):
    interface = VegaInterface(str(write_configs(tmp_path, marginalize=False)))
    data = interface.data["first"]

    assert data.marginalization.families == []
    assert data.marginalization.cov_update is None
    assert data.num_marg_modes == 0
    np.testing.assert_array_equal(data.cov_mat, data.cov_mat_org)

    chi2, coefficients = interface.chi2(return_marg_coeff=True)
    assert coefficients == {}
    assert chi2 == interface.chi2()
