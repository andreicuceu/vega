"""Small synthetic tests of elimination and the actual model response path."""

import configparser
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.sparse import csr_array
from scipy.stats import multivariate_normal

from vega.broadband_poly import BroadbandPolynomials
from vega.linear_nuisance import LinearNuisance, solve_linear
from vega.model import Model
from vega.utils import find_file
from vega.vega_interface import VegaInterface


def test_gaussian_integration_matches_covariance_update():
    rng = np.random.default_rng(25)
    factor = rng.normal(size=(9, 9))
    cov = factor @ factor.T + np.eye(9)
    templates = rng.normal(size=(9, 4))
    residual = rng.normal(size=9)
    means = np.array([0.2, -0.7, 1.0, 0.4])
    sigmas = np.array([0.2, 0.4, 1.3, 0.9])
    precision = np.linalg.inv(cov)
    result = solve_linear(
        residual, templates, precision, means=means, sigmas=sigmas, mode="integrate"
    )
    updated = cov + (templates * sigmas**2) @ templates.T
    actual = -0.5 * (result.objective + np.linalg.slogdet(cov)[1] + 9 * np.log(2 * np.pi))
    expected = multivariate_normal.logpdf(residual, mean=templates @ means, cov=updated)
    assert actual == pytest.approx(expected)
    a = templates.T @ precision @ templates + np.diag(sigmas**-2)
    expected_coeff = np.linalg.solve(a, templates.T @ precision @ residual + means / sigmas**2)
    np.testing.assert_allclose(result.coefficients, expected_coeff)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(a))


def test_profile_matches_explicit_joint_minimization():
    templates = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.9], [1.0, -0.2]])
    cov = np.array(
        [[2.0, 0.5, 0.1, 0.3], [0.5, 1.0, 0.2, 0.1], [0.1, 0.2, 3.0, 0.8], [0.3, 0.1, 0.8, 1.5]]
    )
    precision = np.linalg.inv(cov)
    residual = np.array([0.3, 2.0, -0.5, 1.0])
    means, sigmas = np.array([0.2, -0.1]), np.array([0.7, np.inf])
    result = solve_linear(residual, templates, precision, means=means, sigmas=sigmas)

    def objective(coeff):
        diff = residual - templates @ coeff
        return diff @ precision @ diff + np.sum(((coeff - means) / sigmas) ** 2)

    numerical = minimize(objective, np.zeros(2))
    assert result.profile == pytest.approx(numerical.fun)
    np.testing.assert_allclose(result.coefficients, numerical.x, atol=1e-6)


def test_flat_integral_and_parameter_dependent_volume():
    residual = np.array([1.0, 2.0, -0.7])
    template = np.array([[1.0], [0.2], [0.6]])
    result = solve_linear(residual, template, np.eye(3), mode="integrate")
    integral = quad(
        lambda a: np.exp(-0.5 * np.sum((residual - template[:, 0] * a) ** 2)), -np.inf, np.inf
    )[0]
    assert result.objective == pytest.approx(-2 * np.log(integral))
    scaled = solve_linear(residual, 3 * template, np.eye(3), mode="integrate")
    assert scaled.profile == pytest.approx(result.profile)
    assert scaled.objective - result.objective == pytest.approx(2 * np.log(3))


def test_degenerate_modes_require_priors():
    templates = np.ones((4, 2))
    with pytest.raises(ValueError, match="Unidentifiable"):
        solve_linear(np.ones(4), templates, np.eye(4))
    result = solve_linear(np.ones(4), templates, np.eye(4), sigmas=[1.0, 2.0], mode="integrate")
    assert np.isfinite(result.objective)
    # A masked/null mode with a normalized proper prior integrates to one.
    null = solve_linear(
        np.ones(4), np.zeros((4, 1)), np.eye(4), means=[9.0], sigmas=[3.0], mode="integrate"
    )
    assert null.objective == pytest.approx(4.0)
    np.testing.assert_allclose(null.coefficients, [9.0])
    empty = solve_linear(np.ones(4), np.empty((4, 0)), np.eye(4))
    assert empty.objective == 4.0


@pytest.mark.parametrize("sigmas", [[0.0], [-1.0], [np.nan]])
def test_invalid_priors(sigmas):
    with pytest.raises(ValueError, match="sigmas"):
        solve_linear(np.ones(3), np.ones((3, 1)), np.eye(3), sigmas=sigmas)


def make_adapter(mode="profile", shared=False, rectangular=False):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_dict({"linear nuisance": {"mode": mode}})
    coordinates = SimpleNamespace(
        r_grid=np.array([30.0, 60.0, 100.0]), mu_grid=np.array([0.2, 0.4, 0.8])
    )
    model_coordinates = (
        SimpleNamespace(r_grid=np.linspace(10.0, 130.0, 6), mu_grid=np.linspace(0.1, 0.9, 6))
        if rectangular
        else coordinates
    )
    models, data, items = {}, {}, {}
    params = {"bao_amp": 1.7}
    for name in ("a", "b"):
        bb = BroadbandPolynomials(
            {
                "1": "add pre r,mu 0:0:1 0:0:1",
                "2": "add post r,mu 1:1:1 0:0:1",
                "3": "mul post r,mu 0:0:1 0:0:1",
            },
            name,
            model_coordinates,
            coordinates,
        )
        model = Model.__new__(Model)
        model.broadband = bb
        model._has_distortion_mat = True
        model._is_multipoles = False
        model._model_pk = False
        model._corr_item = SimpleNamespace(has_metals=False)
        model._instrumental_systematics_flag = False
        model.save_components = False
        model.Pk_core = SimpleNamespace(compute=lambda pk, pars: pk)
        model.PktoXi = SimpleNamespace(cache_pars=None)
        model.Xi_core = SimpleNamespace(compute=lambda pk, pk_input, transform, pars: pk.copy())
        dmat = np.array([[1.0, 0.2, -0.1], [0.1, 0.8, 0.3], [0.3, -0.2, 0.9]])
        if rectangular:
            dmat = np.concatenate([dmat, 0.2 * dmat], axis=1)
        datum = SimpleNamespace(
            model_mask=np.array([True, False, True]),
            distortion_mat=csr_array(dmat),
            is_direct_multipoles=False,
            use_multipoles=False,
            masked_data_vec=np.array([0.4, -0.3]),
            inv_masked_cov=np.eye(2),
        )
        model._data = datum
        models[name], data[name] = model, datum
        items[name] = SimpleNamespace(marginalize_small_scales={})
        params[f"BB-{name}-0 add pre r,mu (0,0)"] = 0.5
        params[f"BB-{name}-1 add post r,mu (1,0)"] = -0.2
        params[f"BB-{name}-2 mul post r,mu (0,0)"] = 0.3
    if shared:
        config.read_dict(
            {
                "linear nuisance ids": {
                    "BB-a-0 add pre r,mu (0,0)": "shared-constant",
                    "BB-b-0 add pre r,mu (0,0)": "shared-constant",
                }
            }
        )
    interface = SimpleNamespace(
        main_config=config,
        _has_data=True,
        model_pk=False,
        models=models,
        data=data,
        corr_items=items,
        sample_params={"limits": {}},
        mc_config=None,
        priors={},
        monte_carlo=False,
        _use_global_cov=False,
    )
    return LinearNuisance(interface), params


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("rectangular", [False, True])
def test_templates_match_actual_model_including_distortion_and_bao(direct, rectangular):
    adapter, params = make_adapter(rectangular=rectangular)
    templates = adapter.templates(params, direct)
    pk = np.linspace(0.2, 0.4, 6 if rectangular else 3)
    smooth = pk * 0.8
    for name, model in adapter.interface.models.items():
        base_params = adapter.base_params(params)

        def compute(pars, current_model=model):
            if direct:
                return current_model.compute_direct(dict(pars), pk)
            return current_model.compute(dict(pars), pk, smooth)

        base = compute(base_params)
        for key, _ in adapter.bb_names[name]:
            unit = dict(base_params)
            unit[key] = 1.0
            np.testing.assert_allclose(
                templates[name][:, adapter.ids.index(key)], compute(unit) - base, atol=1e-14
            )


def test_shared_ids_global_covariance_and_masks():
    adapter, params = make_adapter(shared=True)
    interface = adapter.interface
    interface._use_global_cov = True
    cov = np.array(
        [[2.0, 0.2, 0.3, 0.1], [0.2, 1.0, 0.1, 0.3], [0.3, 0.1, 1.0, 0.2], [0.1, 0.3, 0.2, 2.0]]
    )
    interface.masked_global_invcov = np.linalg.inv(cov)
    base = {"a": np.array([0.1, 99.0, 0.2]), "b": np.array([0.2, -99.0, -0.1])}
    actual = adapter.evaluate(base, params)
    matrix = np.vstack(
        [
            adapter.templates(params)[name][interface.data[name].model_mask]
            for name in interface.corr_items
        ]
    )
    residual = np.concatenate(
        [
            interface.data[name].masked_data_vec - base[name][interface.data[name].model_mask]
            for name in interface.corr_items
        ]
    )
    expected = solve_linear(residual, matrix, interface.masked_global_invcov)
    assert len(actual.ids) == 3
    np.testing.assert_allclose(actual.coefficients, expected.coefficients)
    assert actual.objective == pytest.approx(expected.objective)
    interface.monte_carlo = True
    interface.analysis = SimpleNamespace(current_mc_mock=np.array([1.0, 0.2, 0.3, 2.0]))
    mock = adapter.evaluate(base, params)
    expected_mock = solve_linear(
        interface.analysis.current_mc_mock
        - np.concatenate(
            [base[name][interface.data[name].model_mask] for name in interface.corr_items]
        ),
        matrix,
        interface.masked_global_invcov,
    )
    np.testing.assert_allclose(mock.coefficients, expected_mock.coefficients)


def test_small_scales_use_one_full_model_amplitude_and_post_multiplier():
    adapter, params = make_adapter()
    interface = adapter.interface
    item = interface.corr_items["a"]
    item.marginalize_small_scales = {"rtmax": 10.0}
    item.marginalize_small_scales_prior_sigma = 2.0
    item.get_undist_xi_marg_templates = lambda: csr_array(np.array([[1.0], [0.0], [0.0]]))
    adapter = LinearNuisance(interface)
    key = adapter.ids.index("small-scale-a (0)")
    actual = adapter.templates(params)["a"][:, key]
    np.testing.assert_allclose(actual, np.array([1.0, 0.1, 0.3]) * 1.3)
    np.testing.assert_allclose(adapter.templates(params, direct=True)["a"][:, key], actual)


def test_sampled_coefficients_rejected():
    adapter, _ = make_adapter()
    adapter.interface.sample_params["limits"][adapter.ids[0]] = (0.0, 1.0)
    with pytest.raises(ValueError, match="Eliminated coefficients"):
        LinearNuisance(adapter.interface)


@pytest.mark.parametrize("section", ["sample", "chi2 scan"])
def test_alias_cannot_be_sampled_or_scanned(section):
    adapter, _ = make_adapter(shared=True)
    if section == "sample":
        adapter.interface.sample_params["limits"]["shared-constant"] = (-1.0, 1.0)
    else:
        adapter.interface.main_config["chi2 scan"] = {"shared-constant": "-1 1 5"}
    with pytest.raises(ValueError, match="Eliminated coefficients"):
        LinearNuisance(adapter.interface)


def test_independent_mock_precision_scaling():
    adapter, params = make_adapter(shared=True)
    interface = adapter.interface
    interface.monte_carlo = True
    base = {name: np.zeros(3) for name in interface.corr_items}
    for name, datum in interface.data.items():
        datum.masked_mc_mock = np.array([0.1, 0.2]) if name == "a" else np.array([0.4, 0.6])
        datum.scaled_inv_masked_cov = datum.inv_masked_cov / (2.0 if name == "a" else 4.0)
    result = adapter.evaluate(base, params)
    matrix = np.vstack(
        [
            adapter.last_templates[name][interface.data[name].model_mask]
            for name in interface.corr_items
        ]
    )
    expected = solve_linear(
        np.concatenate([datum.masked_mc_mock for datum in interface.data.values()]),
        matrix,
        np.diag([0.5, 0.5, 0.25, 0.25]),
    )
    np.testing.assert_allclose(result.coefficients, expected.coefficients)


def tiny_config(tmp_path, global_cov=False):
    """Real FITS/config input on a 12-bin grid, no external scientific data."""
    rp = np.repeat([25.0, 75.0, 125.0, 175.0], 3)
    rt = np.tile([20.0, 60.0, 100.0], 4)
    n = rp.size
    covariance = np.eye(n) * 0.01 + 0.001
    distortion = np.eye(n) + 0.02
    columns = [
        fits.Column(name=key, format="D", array=value)
        for key, value in {
            "RP": rp,
            "RT": rt,
            "Z": np.full(n, 2.33),
            "DA": np.linspace(0.0, 0.01, n),
        }.items()
    ]
    columns.extend(
        [
            fits.Column(name="CO", format=f"{n}D", array=covariance),
            fits.Column(name="DM", format=f"{n}D", array=distortion),
        ]
    )
    hdu = fits.BinTableHDU.from_columns(columns)
    for key, value in {"RPMIN": 0.0, "RPMAX": 200.0, "RTMAX": 120.0, "NP": 4, "NT": 3}.items():
        hdu.header[key] = value
    data_path = tmp_path / "data.fits"
    hdu.writeto(data_path)
    component_paths = []
    priors = {}
    for name in ("tiny", "other") if global_cov else ("tiny",):
        component = configparser.ConfigParser()
        component.optionxform = str
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
                "model": {
                    "z evol LYA": "bias_vs_z_std",
                    "marginalize-below-rtmax": "40",
                    "marginalize-prior-sigma": "0.3",
                    "fit-marginalized-scales": "True",
                },
                "broadband": {"1": "add pre r,mu 0:0:1 0:0:1", "2": "add post r,mu 1:1:1 0:0:1"},
            }
        )
        component_path = tmp_path / f"{name}.ini"
        with component_path.open("w") as stream:
            component.write(stream)
        component_paths.append(str(component_path))
        priors[f"BB-{name}-0 add pre r,mu (0,0)"] = "0.01 0.2"
        priors[f"BB-{name}-1 add post r,mu (1,0)"] = "0 0.1"
    main = configparser.ConfigParser()
    main.optionxform = str
    main.read(find_file("full_configs/main.ini"))
    main["data sets"]["ini files"] = " ".join(component_paths)
    main["linear nuisance"] = {"mode": "integrate"}
    main["linear nuisance priors"] = priors
    for key in priors:
        main["parameters"][key] = "0"
    main["output"]["filename"] = str(tmp_path / "result.fits")
    if global_cov:
        joint_cov = np.block([[covariance, 0.1 * covariance], [0.1 * covariance, covariance]])
        global_path = tmp_path / "global.fits"
        fits.BinTableHDU.from_columns(
            [fits.Column(name="COV", format=f"{2 * n}D", array=joint_cov)]
        ).writeto(global_path)
        main["data sets"]["global-cov-file"] = str(global_path)
    path = tmp_path / "main.ini"
    with path.open("w") as stream:
        main.write(stream)
    return path


@pytest.mark.parametrize("joint", [False, True])
def test_full_interface_covariance_equivalence_and_outputs(tmp_path, joint):
    path = tiny_config(tmp_path, joint)
    interface = VegaInterface(str(path))
    adapter = interface.linear_nuisance
    base = interface.compute_model(run_init=False)
    result = interface.compute_linear_nuisance()
    assert interface.corr_num_marg_modes == {"linear_nuisance": len(result.ids)}
    # New path never applies the old covariance inflation or component-wise solve.
    for datum in interface.data.values():
        assert datum.cov_marg_update is None
        assert datum.marg_diff2coeff_matrix is None
        np.testing.assert_allclose(datum.cov_mat, datum.cov_mat_org)
        assert datum.data_mask[0]  # fit-marginalized-scales expands the small-bin mask
    matrix = np.vstack(
        [
            adapter.last_templates[name][interface.data[name].model_mask]
            for name in interface.corr_items
        ]
    )
    residual = np.concatenate(
        [
            interface.data[name].masked_data_vec - base[name][interface.data[name].model_mask]
            for name in interface.corr_items
        ]
    )
    if joint:
        cov = np.linalg.inv(interface.masked_global_invcov)
    else:
        cov = np.linalg.inv(interface.data["tiny"].inv_masked_cov)
    updated = cov + (matrix * adapter.sigmas**2) @ matrix.T
    expected = multivariate_normal.logpdf(residual, mean=matrix @ adapter.means, cov=updated)
    assert interface.log_lik() == pytest.approx(expected)
    value, coefficients = interface.log_lik(return_marg_coeff=True)
    assert value == pytest.approx(expected)
    np.testing.assert_allclose(coefficients, result.coefficients)
    corrected = interface.compute_model(
        run_init=False, marg_coeff=dict(zip(result.ids, result.coefficients))
    )
    for name in corrected:
        np.testing.assert_allclose(
            corrected[name], base[name] + adapter.last_templates[name] @ result.coefficients
        )
    # The ordinary notebook/script fit path must use jointly corrected output.
    interface.minimize()
    assert interface.p_value == -1.0
    assert interface.bestfit_linear_nuisance.ids == result.ids
    interface.output.write_results(
        interface.bestfit_model, interface.params, interface.minimizer, interface.bestfit_corr_stats
    )
    with fits.open(interface.output.outfile) as hdul:
        assert hdul[1].header["linear_nuisance_mode"] == "integrate"
        assert hdul[-1].header["PVALUE"] == -1.0


def test_no_opt_in_preserves_legacy_and_rejects_mixed_mode(tmp_path):
    path = tiny_config(tmp_path)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)
    config["control"]["marginalize-in-fit"] = "True"
    with path.open("w") as stream:
        config.write(stream)
    with pytest.raises(ValueError, match="cannot be combined"):
        VegaInterface(str(path))
    config.remove_section("linear nuisance")
    with path.open("w") as stream:
        config.write(stream)
    legacy = VegaInterface(str(path))
    assert legacy.linear_nuisance is None
    assert legacy.data["tiny"].marg_diff2coeff_matrix is not None
    assert np.isfinite(legacy.chi2())
