"""Tests of the [analytic marginalization] parameters and of the metal x metal amplitudes.

The configurations are copies of ``tests/full_configs`` with three correlations (Lya x Lya,
Lya x Lyb and Lya x QSO). The autos get the DESI instrumental systematics and
``metal-auto-amplitudes``, so the marginalized parameters cover every kind of template:
fixed and shared by two correlations (``desi_inst_sys_amp``, the metal x metal amplitudes with
fast metals), rebuilt at every evaluation (``qso_rad_strength``, the metal biases, and the
amplitudes without fast metals), and shared by all three correlations (the metal biases).

The reference solutions are independent of the module: the templates are finite differences
of the model (exact, as the model is linear in these parameters), and the marginal likelihood
is computed with dense covariance matrices.
"""

import configparser

import numpy as np
import pytest
from astropy.io import fits

from vega.metals import Metals
from vega.output import Output
from vega.utils import bias_beta

from .test_metals import TESTS_DIR, apply_edits, build_interface, read_config, write_config_copy

INI_FILES = ["lyalya_lyalya.ini", "lyalya_lyalyb.ini", "lyalya_qso.ini"]
AUTO_CORRELATIONS = ["lyalya_lyalya.ini", "lyalya_lyalyb.ini"]

SILICON_LINES = ["SiII(1190)", "SiII(1193)", "SiII(1260)", "SiIII(1207)"]
METAL_PAIRS = [(m1, m2) for i, m1 in enumerate(SILICON_LINES) for m2 in SILICON_LINES[i:]] + [
    ("CIV(eff)", "CIV(eff)")
]
AMPLITUDES = [Metals.amplitude_name(m1, m2) for m1, m2 in METAL_PAIRS]
METAL_BIASES = [f"bias_eta_{metal}" for metal in SILICON_LINES]
LINEAR_NAMES = ["desi_inst_sys_amp", "qso_rad_strength"] + METAL_BIASES + AMPLITUDES

# Values for the model at fixed parameters: the metal biases are those of full_configs
AMPLITUDE_VALUES = {name: 1e-5 * (i + 1) for i, name in enumerate(AMPLITUDES)}
FIXED_VALUES = {"desi_inst_sys_amp": 3e-4, **AMPLITUDE_VALUES}

# Gaussian priors on part of the parameters; the others are flat. The metal matrix file of
# the test data has no matrices and almost equal coordinates for all Si lines, so the
# templates of the Si biases are degenerate and need a prior.
GAUSSIAN_PRIORS = {
    "desi_inst_sys_amp": (3e-4, 1e-4),
    **{name: (0.0, 1e-2) for name in METAL_BIASES},
    **{name: (0.0, 1e-3) for name in AMPLITUDES},
}


def write_linear_configs(
    directory,
    fast_metals=True,
    linear_names=LINEAR_NAMES,
    priors=GAUSSIAN_PRIORS,
    main_edits=None,
    auto_edits=None,
    ini_files=INI_FILES,
):
    """Write a copy of the full configurations with analytically marginalized parameters.

    Parameters
    ----------
    directory : pathlib.Path
        Output directory.
    fast_metals : bool, optional
        Value of ``fast_metals`` in all correlations, by default True.
    linear_names : list of str, optional
        Parameters listed in ``[analytic marginalization]``, by default ``LINEAR_NAMES``.
    priors : dict, optional
        Gaussian priors ``{name: (mean, sigma)}``, written to ``[priors]`` for the parameters
        of ``linear_names``.
    main_edits : dict, optional
        Further edits of the main INI, see ``apply_edits``.
    auto_edits : dict, optional
        Further edits of the auto-correlation INIs, see ``apply_edits``.
    ini_files : list of str, optional
        Correlation INIs to keep, by default ``INI_FILES``.

    Returns
    -------
    pathlib.Path
        Main INI of the copy.
    """
    edits = {
        "analytic marginalization": dict.fromkeys(linear_names, "True"),
        "parameters": {name: str(value) for name, value in FIXED_VALUES.items()},
        "priors": {
            name: f"gaussian {mean} {sigma}"
            for name, (mean, sigma) in priors.items()
            if name in linear_names
        },
    }
    for section, options in (main_edits or {}).items():
        edits.setdefault(section, {}).update(options)

    main_ini = write_config_copy(
        TESTS_DIR / "full_configs" / "main.ini",
        directory,
        main_edits=edits,
        corr_edits={"model": {"fast_metals": str(fast_metals)}},
        ini_files=ini_files,
    )

    auto_model_edits = {
        "model": {"desi-instrumental-systematics": "True", "metal-auto-amplitudes": "True"}
    }
    for section, options in (auto_edits or {}).items():
        auto_model_edits.setdefault(section, {}).update(options)
    for ini_file in AUTO_CORRELATIONS:
        path = directory.resolve() / ini_file
        if not path.exists():
            continue
        config = read_config(path)
        apply_edits(config, auto_model_edits)
        with open(path, "w") as file:
            config.write(file)

    return main_ini


def finite_difference_templates(vega, names, params=None):
    """Templates of linear parameters as differences of models, on the full output grids.

    Parameters
    ----------
    vega : vega.VegaInterface
        Initialized interface.
    names : list of str
        Linear parameters.
    params : dict, optional
        Values of the other parameters, by default those of the interface.

    Returns
    -------
    base_model : dict
        Model with all marginalized coefficients set to zero, keyed by correlation.
    templates : dict
        Arrays with shape ``(model_size, len(names))`` keyed by correlation.
    """
    base_params = vega.marginalization.get_base_model_params(params)
    base_model = vega.compute_model(base_params, run_init=False)

    templates = {name: np.zeros((model.size, len(names))) for name, model in base_model.items()}
    for j, parameter in enumerate(names):
        model = vega.compute_model(base_params | {parameter: 1.0}, run_init=False)
        for name in model:
            templates[name][:, j] = model[name] - base_model[name]

    # The last evaluation must be the base model for the templates captured by the models
    base_model = vega.compute_model(base_params, run_init=False)
    return base_model, templates


def explicit_marginalization(vega, base_model, templates, names, priors):
    """Profile chi2, marginal log-likelihood and best-fit coefficients from dense matrices.

    The Gaussian-prior coefficients are marginalized through the covariance
    ``C + T Sigma T.T`` and the flat ones through the normal equations, an order of
    operations different from the module.

    Parameters
    ----------
    vega : vega.VegaInterface
        Interface with block covariances.
    base_model, templates : dict
        As returned by ``finite_difference_templates``.
    names : list of str
        Linear parameters, in the column order of ``templates``.
    priors : dict
        Gaussian priors ``{name: (mean, sigma)}``; parameters without one are flat.

    Returns
    -------
    chi2 : float
        Chi2 at the best-fit coefficients, including the prior penalty.
    minus_two_log_lik : float
        Minus twice the marginal log-likelihood.
    coefficients : 1D array
        Best-fit coefficients.
    """
    corr_names = list(vega.corr_items)
    data_vector = np.concatenate([vega.data[name].masked_data_vec for name in corr_names])
    model = np.concatenate([base_model[name][vega.data[name].model_mask] for name in corr_names])
    template_matrix = np.vstack(
        [templates[name][vega.data[name].model_mask] for name in corr_names]
    )
    covariance_blocks = [
        vega.data[name].cov_mat[np.ix_(vega.data[name].data_mask, vega.data[name].data_mask)]
        for name in corr_names
    ]
    covariance = np.zeros((data_vector.size, data_vector.size))
    start = 0
    for block in covariance_blocks:
        stop = start + block.shape[0]
        covariance[start:stop, start:stop] = block
        start = stop

    gaussian = np.array([name in priors for name in names])
    means = np.array([priors[name][0] if name in priors else 0.0 for name in names])
    sigmas = np.array([priors[name][1] if name in priors else np.inf for name in names])

    # Gaussian coefficients: covariance update around the prior means
    residual = data_vector - model - template_matrix[:, gaussian].dot(means[gaussian])
    gaussian_templates = template_matrix[:, gaussian]
    updated_covariance = covariance + (gaussian_templates * sigmas[gaussian] ** 2).dot(
        gaussian_templates.T
    )
    precision = np.linalg.inv(updated_covariance)

    # Flat coefficients: normal equations with the updated precision
    flat_templates = template_matrix[:, ~gaussian]
    flat_normal_matrix = flat_templates.T.dot(precision).dot(flat_templates)
    flat_projection = flat_templates.T.dot(precision).dot(residual)
    flat_coefficients = np.linalg.solve(flat_normal_matrix, flat_projection)
    chi2 = residual.dot(precision).dot(residual) - flat_projection.dot(flat_coefficients)

    minus_two_log_lik = (
        chi2
        + data_vector.size * np.log(2 * np.pi)
        + np.linalg.slogdet(updated_covariance)[1]
        + np.linalg.slogdet(flat_normal_matrix)[1]
        - np.sum(~gaussian) * np.log(2 * np.pi)
    )

    # Joint best fit of all coefficients with the original covariance
    original_precision = np.linalg.inv(covariance)
    prior_precision = np.where(gaussian, 1 / sigmas**2, 0.0)
    normal_matrix = template_matrix.T.dot(original_precision).dot(template_matrix)
    normal_matrix += np.diag(prior_precision)
    projection = template_matrix.T.dot(original_precision).dot(data_vector - model)
    projection += prior_precision * means
    coefficients = np.linalg.solve(normal_matrix, projection)

    return chi2, minus_two_log_lik, coefficients


@pytest.fixture(scope="module", params=[True, False], ids=["fast-metals", "slow-metals"])
def linear_interface(request, tmp_path_factory):
    """Interface with all parameters of ``LINEAR_NAMES`` marginalized."""
    directory = tmp_path_factory.mktemp(f"linear_{request.param}")
    vega, _ = build_interface(write_linear_configs(directory, fast_metals=request.param))
    return vega


# ----------------------------------------------------------------------------------------------
# Metal x metal amplitudes
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fast_metals", [True, False])
def test_metal_amplitudes_reproduce_bias_products(tmp_path, fast_metals):
    """With amp_<m1>_<m2> = b_m1 b_m2 the model is that of the bias products."""
    vega, _ = build_interface(
        write_linear_configs(tmp_path / "amplitudes", fast_metals, linear_names=[])
    )
    reference, _ = build_interface(
        write_linear_configs(
            tmp_path / "products",
            fast_metals,
            linear_names=[],
            auto_edits={"model": {"metal-auto-amplitudes": "False"}},
        )
    )

    # Bias products as computed by the metal model (fast metals use the fiducial growth rate)
    metals = vega.models["lyalya_lyalya"].metals
    local_params = dict(vega.params)
    if metals.fast_metals:
        local_params["growth_rate"] = metals.growth_rate
    amplitudes = {}
    for m1, m2 in METAL_PAIRS:
        bias1, _, bias2, _ = bias_beta(local_params, m1, m2)
        amplitudes[Metals.amplitude_name(m1, m2)] = bias1 * bias2

    model = vega.compute_model(amplitudes, run_init=False)
    reference_model = reference.compute_model(run_init=False)
    for name in reference_model:
        scale = np.max(np.abs(reference_model[name]))
        np.testing.assert_allclose(model[name], reference_model[name], rtol=0, atol=1e-14 * scale)


def test_removed_separate_metal_auto_biases_option_raises(tmp_path):
    main_ini = write_linear_configs(
        tmp_path, linear_names=[], auto_edits={"model": {"separate-metal-auto-biases": "True"}}
    )
    with pytest.raises(ValueError, match="metal-auto-amplitudes"):
        build_interface(main_ini)


def test_metal_amplitudes_need_fast_metal_bias(tmp_path):
    main_ini = write_linear_configs(
        tmp_path,
        fast_metals=False,
        linear_names=[],
        auto_edits={"model": {"fast_metal_bias": "False"}},
    )
    with pytest.raises(ValueError, match="fast_metal_bias"):
        build_interface(main_ini)


# ----------------------------------------------------------------------------------------------
# Templates and likelihood
# ----------------------------------------------------------------------------------------------


def test_fixed_and_changing_templates(linear_interface):
    """Fixed templates: instrumental systematics, and the amplitudes with fast metals."""
    linear = linear_interface.marginalization.linear
    fast = linear_interface.models["lyalya_lyalya"].metals.fast_metals
    expected_fixed = ["desi_inst_sys_amp"] + (AMPLITUDES if fast else [])

    assert [name for name, fixed in zip(linear.names, linear.is_fixed) if fixed] == expected_fixed
    assert linear.component_parameters["lyalya_qso"] == ["qso_rad_strength"] + METAL_BIASES
    assert set(linear.component_parameters["lyalya_lyalyb"]) == set(LINEAR_NAMES) - {
        "qso_rad_strength"
    }


def test_templates_reproduce_the_model(linear_interface):
    """Base model plus templates times values equals the model at those values."""
    vega = linear_interface
    values = {name: vega.params[name] for name in METAL_BIASES}
    values |= FIXED_VALUES | {"qso_rad_strength": 0.74}
    coefficients = np.array([values[name] for name in vega.marginalization.linear.names])

    full_model = vega.compute_model(values, run_init=False)
    base_params = vega.marginalization.get_base_model_params(values)
    model = vega.compute_model(base_params, run_init=False)
    vega.marginalization.linear.add_templates(model, coefficients)

    for name in full_model:
        scale = np.max(np.abs(full_model[name]))
        np.testing.assert_allclose(model[name], full_model[name], rtol=0, atol=1e-12 * scale)


def test_chi2_and_log_lik_match_dense_marginalization(linear_interface):
    """Profile chi2, marginal likelihood and coefficients against dense linear algebra."""
    vega = linear_interface
    names = list(vega.marginalization.linear.names)
    params = {"bias_eta_LYA": -0.19, "beta_LYA": 1.6}

    base_model, templates = finite_difference_templates(vega, names, params)
    chi2, minus_two_log_lik, coefficients = explicit_marginalization(
        vega, base_model, templates, names, GAUSSIAN_PRIORS
    )

    assert vega.chi2(params) == pytest.approx(chi2, rel=1e-9)
    assert -2 * vega.log_lik(params) == pytest.approx(minus_two_log_lik, rel=1e-9)

    # Differences in units of the coefficient errors (some coefficients are close to zero)
    _, marg_coeff = vega.chi2(params, return_marg_coeff=True)
    errors = vega.marginalization.linear.last_solution.coefficient_errors
    assert np.all(np.abs(marg_coeff["linear"] - coefficients) < 1e-6 * errors)

    # The priors of the marginalized parameters do not enter the prior chi2
    assert vega.compute_prior_chi2(params) == 0


def test_global_covariance_matches_block_covariance(tmp_path):
    """A global covariance equal to the block covariance gives the same likelihood."""
    ini_files = ["lyalya_qso.ini"]
    linear_names = ["qso_rad_strength"] + METAL_BIASES
    block, _ = build_interface(
        write_linear_configs(tmp_path / "block", linear_names=linear_names, ini_files=ini_files)
    )

    covariance = block.data["lyalya_qso"].cov_mat
    global_path = tmp_path / "global.fits"
    fits.BinTableHDU.from_columns(
        [fits.Column(name="COV", format=f"{covariance.shape[0]}D", array=covariance)]
    ).writeto(global_path)
    main_ini = write_linear_configs(
        tmp_path / "global",
        linear_names=linear_names,
        ini_files=ini_files,
        main_edits={"data sets": {"global-cov-file": str(global_path)}},
    )
    joint, _ = build_interface(main_ini)
    assert joint._use_global_cov

    params = {"bias_eta_LYA": -0.19}
    assert joint.chi2(params) == pytest.approx(block.chi2(params), rel=1e-9)
    assert joint.log_lik(params) == pytest.approx(block.log_lik(params), rel=1e-9)


def test_profile_chi2_with_marginalized_broadbands(tmp_path):
    """Per-correlation and shared coefficients with flat priors: profile over all of them."""
    linear_names = ["desi_inst_sys_amp", "qso_rad_strength"]
    main_ini = write_linear_configs(
        tmp_path,
        linear_names=linear_names,
        priors={},
        auto_edits={"model": {"marginalize-broadbands": "True"}},
        ini_files=["lyalya_lyalya.ini", "lyalya_qso.ini"],
    )
    vega, _ = build_interface(main_ini)
    base_model, templates = finite_difference_templates(vega, linear_names)

    # Add the broadband templates of the auto to the template matrix
    broadband = vega.data["lyalya_lyalya"].marginalization.templates.toarray()
    num_broadband = broadband.shape[1]
    all_templates = {
        "lyalya_lyalya": np.hstack([templates["lyalya_lyalya"], broadband]),
        "lyalya_qso": np.hstack(
            [templates["lyalya_qso"], np.zeros((base_model["lyalya_qso"].size, num_broadband))]
        ),
    }
    names = linear_names + [f"broadband {i}" for i in range(num_broadband)]

    # Flat priors leave the covariance unchanged; the module projects the broadbands out of
    # the precision and marginalizes the shared parameters afterwards.
    chi2, _, coefficients = explicit_marginalization(vega, base_model, all_templates, names, {})

    vega_chi2, marg_coeff = vega.chi2(return_marg_coeff=True)
    assert vega_chi2 == pytest.approx(chi2, rel=1e-9)
    vega_coefficients = np.concatenate([marg_coeff["linear"], marg_coeff["lyalya_lyalya"]])
    scale = np.abs(coefficients)
    assert np.all(np.abs(vega_coefficients - coefficients) < 1e-6 * scale + 1e-12)


def test_missing_parameter_values_are_zero(tmp_path):
    """Marginalized parameters can be left out of [parameters]."""
    main_ini = write_linear_configs(tmp_path, fast_metals=True)
    config = read_config(main_ini)
    for name in ["qso_rad_strength", "desi_inst_sys_amp"] + AMPLITUDES:
        config.remove_option("parameters", name)
    with open(main_ini, "w") as file:
        config.write(file)

    vega, _ = build_interface(main_ini)
    reference, _ = build_interface(write_linear_configs(tmp_path / "reference"))
    assert vega.params["qso_rad_strength"] == 0
    assert vega.chi2() == reference.chi2()


def test_degrees_of_freedom_and_output_table(tmp_path):
    linear_names = ["qso_rad_strength"] + METAL_BIASES
    main_ini = write_linear_configs(
        tmp_path, linear_names=linear_names, ini_files=["lyalya_qso.ini"]
    )
    vega, _ = build_interface(main_ini)
    assert vega.num_model_params == len(vega.sample_params["limits"]) + len(linear_names)

    vega.minimize()
    assert vega.dof_counts["num_other_marginalized"] == len(linear_names)
    assert set(vega.bestfit_marginalized_values) == set(linear_names)

    # The best-fit model includes the marginalized terms at their best-fit values
    params = vega.params | vega.minimizer.values | vega.bestfit_marginalized_values
    full_model = vega.compute_model(params, run_init=False)["lyalya_qso"]
    scale = np.max(np.abs(full_model))
    np.testing.assert_allclose(
        vega.bestfit_model["lyalya_qso"], full_model, rtol=0, atol=1e-12 * scale
    )

    hdu = Output._marg_coeff_hdu(vega.bestfit_corr_stats, vega.minimizer.linear_coefficient_table)
    assert list(hdu.data["names"]) == linear_names
    assert set(hdu.data["correlation"]) == {"lyalya_qso"}


def test_monte_carlo_forecast_recovers_input_values(tmp_path):
    """A noiseless mock of the full model is fitted exactly by the marginalized parameters."""
    linear_names = ["qso_rad_strength", METAL_BIASES[0]]
    main_ini = write_linear_configs(
        tmp_path, linear_names=linear_names, priors={}, ini_files=["lyalya_qso.ini"]
    )
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(main_ini)
    config["monte carlo"] = dict(config["sample"])
    config["mc parameters"] = {}
    config["sample"] = {}
    config["control"]["forecast"] = "True"
    with open(main_ini, "w") as file:
        config.write(file)

    vega, _ = build_interface(main_ini)
    vega.initialize_monte_carlo()

    # The other Si lines are not marginalized: the mock contains them at their values

    chi2, marg_coeff = vega.chi2(return_marg_coeff=True)
    assert chi2 == pytest.approx(0, abs=1e-8)
    expected = [vega.params[name] for name in linear_names]
    np.testing.assert_allclose(marg_coeff["linear"], expected, rtol=1e-6)


# ----------------------------------------------------------------------------------------------
# Refused configurations
# ----------------------------------------------------------------------------------------------

REFUSED_CASES = [
    pytest.param(
        {"main_edits": {"sample": {"qso_rad_strength": "0 2 0.74 0.1"}}},
        "both varied and analytically marginalized",
        id="also-sampled",
    ),
    pytest.param(
        {"linear_names": ["bias_hcd"]},
        "bias_hcd has no linear template",
        id="no-template",
    ),
    pytest.param(
        {
            "linear_names": AMPLITUDES[:1],
            "auto_edits": {"model": {"metal-auto-amplitudes": "False"}},
        },
        "metal-auto-amplitudes is not set",
        id="amplitude-without-option",
    ),
    pytest.param(
        {"linear_names": METAL_BIASES, "auto_edits": {"model": {"metal-auto-amplitudes": "False"}}},
        "quadratic in its bias",
        id="bias-with-metal-products",
    ),
    pytest.param(
        {"linear_names": METAL_BIASES[:1], "auto_edits": {"model": {"no-metal-decomp": "False"}}},
        "no-metal-decomp = False",
        id="no-metal-decomp",
    ),
    pytest.param(
        {
            "linear_names": METAL_BIASES[:1],
            "main_edits": {"parameters": {"bias_SiII(1190)": "-0.01"}},
        },
        "would replace bias_eta_SiII(1190)",
        id="bias-and-bias-eta",
    ),
]


@pytest.mark.parametrize("config_options, message", REFUSED_CASES)
def test_refused_configurations(tmp_path, config_options, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build_interface(write_linear_configs(tmp_path, fast_metals=False, **config_options))


def test_direct_model_refuses_metal_parameters(linear_interface):
    with pytest.raises(ValueError, match="no metals"):
        linear_interface.chi2(direct_pk=linear_interface.fiducial["pk_full"])


def test_degenerate_flat_templates_are_refused(tmp_path):
    """The Si-bias templates of the test data are degenerate: flat priors are refused."""
    main_ini = write_linear_configs(tmp_path, linear_names=METAL_BIASES, priors={})
    with pytest.raises(ValueError, match="degenerate"):
        build_interface(main_ini)


def test_coefficients_at_every_evaluation(tmp_path):
    """With marginalize-in-fit the coefficients are derived parameters named after them."""
    linear_names = ["qso_rad_strength"] + METAL_BIASES
    main_ini = write_linear_configs(
        tmp_path,
        linear_names=linear_names,
        ini_files=["lyalya_qso.ini"],
        main_edits={"control": {"marginalize-in-fit": "True"}},
    )
    vega, _ = build_interface(main_ini)
    assert vega.corr_num_marg_modes["linear"] == linear_names

    log_lik, coefficients = vega.log_lik(return_marg_coeff=True)
    assert log_lik == vega.log_lik()
    assert coefficients.shape == (len(linear_names),)
