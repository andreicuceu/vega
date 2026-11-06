"""Small analytic and integration checks for configuration-space forecasting."""

import configparser
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from vega import VegaInterface
from vega.fisher import compute_fisher
from vega.scripts.run_vega_fisher import run_fisher
from vega.utils import find_file


def toy():
    """Two correlations with deliberately different model and data grids."""
    vega = SimpleNamespace(
        params={"a": 1.0, "b": 2.0},
        sample_params={"limits": {"a": (None, None), "b": (None, None)}},
        priors={},
        model_pk=False,
        _has_data=True,
        monte_carlo=False,
        marginalize_in_fit=False,
        _blind=False,
        _use_global_cov=True,
        low_mem_mode=False,
        _percival_corr=1.0,
        corr_items={"second": None, "first": None},
        data={
            "second": SimpleNamespace(
                model_mask=np.array([False, True, True, False]),
                data_mask=np.array([True, False, True]),
            ),
            "first": SimpleNamespace(
                model_mask=np.array([True, False]),
                data_mask=np.array([False, True]),
            ),
        },
        full_data_mask=np.array([True, False, True, False, True]),
    )
    matrix = np.array(
        [
            [2.0, 0.0, 0.4, 0.0, 0.3],
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 3.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 7.0, 0.0],
            [0.3, 0.0, 0.2, 0.0, 4.0],
        ]
    )
    vega.global_cov = matrix

    def compute_model(params, run_init=False):
        assert not run_init
        vega.last_params = params.copy()
        a, b = params["a"], params["b"]
        return {
            "first": np.array([3 * a - b, 999.0]),
            "second": np.array([999.0, a + 2 * b, 2 * a + b, 999.0]),
        }

    vega.compute_model = compute_model
    return vega


def test_joint_covariance_masks_order_priors_and_save(tmp_path):
    vega = toy()
    original = copy.deepcopy(vega.params)
    vega.priors = {"b": (9.0, 0.5)}
    result = compute_fisher(vega, ["b", "a"], steps={"b": 0.01, "a": 0.001})
    jacobian = np.array([[2.0, 1.0], [1.0, 2.0], [-1.0, 3.0]])
    selected = vega.global_cov[np.ix_(vega.full_data_mask, vega.full_data_mask)]
    expected = jacobian.T @ np.linalg.solve(selected, jacobian)
    np.testing.assert_allclose(result.jacobian, jacobian, atol=1e-10)
    np.testing.assert_allclose(result.fisher_data, expected)
    np.testing.assert_allclose(result.fisher_prior, np.diag([4.0, 0.0]))
    np.testing.assert_allclose(result.covariance, np.linalg.inv(expected + np.diag([4.0, 0.0])))
    np.testing.assert_allclose(result.step_changes, 0.0, atol=1e-10)
    assert result.rank == 2
    assert result.null_directions.shape == (2, 0)
    assert result.metadata["components"] == ["second", "first"]
    assert vega.params == original == vega.last_params
    path = tmp_path / "forecast.npz"
    result.save(path)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["fisher"], result.fisher)
        assert json.loads(str(saved["metadata"]))["selected_bin_counts"] == [2, 1]
    with pytest.raises(FileExistsError):
        result.save(path)


def test_explicit_covariance_and_low_memory_precision_agree():
    vega = toy()
    steps = {"a": 0.01, "b": 0.01}
    expected = compute_fisher(vega, steps=steps)
    selected = vega.global_cov[np.ix_(vega.full_data_mask, vega.full_data_mask)]
    explicit = compute_fisher(vega, steps=steps, covariance=selected)
    vega.low_mem_mode = True
    vega.masked_global_invcov = np.linalg.inv(selected)
    del vega.global_cov
    low_mem = compute_fisher(vega, steps=steps)
    np.testing.assert_allclose(explicit.fisher, expected.fisher)
    np.testing.assert_allclose(low_mem.fisher, expected.fisher)


def test_rank_deficiency_null_directions_and_prior_regularization():
    vega = toy()
    vega.compute_model = lambda p, run_init=False: {
        "second": np.array([0.0, p["a"] + p["b"], 2 * (p["a"] + p["b"]), 0.0]),
        "first": np.array([3 * (p["a"] + p["b"]), 0.0]),
    }
    with pytest.warns(RuntimeWarning, match="rank 1/2"):
        result = compute_fisher(vega, steps={"a": 0.01, "b": 0.01})
    assert result.rank == 1
    assert result.covariance is None and result.sigma is None
    np.testing.assert_allclose(result.jacobian @ result.null_directions, 0.0, atol=1e-10)
    fixed = compute_fisher(vega, steps={"a": 0.01, "b": 0.01}, priors={"a": (1.0, 1.0)})
    assert fixed.rank == 2
    assert np.all(np.isfinite(fixed.sigma))


def test_step_check_detects_truncation_error():
    vega = toy()
    vega.compute_model = lambda p, run_init=False: {
        "second": np.array([0.0, p["a"] ** 3, p["a"] ** 3, 0.0]),
        "first": np.array([p["a"] ** 3, 0.0]),
    }
    with pytest.warns(RuntimeWarning, match="changes by"):
        result = compute_fisher(vega, ["a"], steps={"a": 1.0})
    assert result.step_changes[0] == pytest.approx(0.75 / 3.25)
    np.testing.assert_allclose(result.jacobian, 4.0)


@pytest.mark.parametrize(
    "covariance,match",
    [
        (np.eye(2), "finite"),
        (np.full((3, 3), np.nan), "finite"),
        (np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.zeros((3, 3)), "positive definite"),
    ],
)
def test_invalid_covariance(covariance, match):
    with pytest.raises(ValueError, match=match):
        compute_fisher(toy(), steps={"a": 0.01, "b": 0.01}, covariance=covariance)


@pytest.mark.parametrize(
    "steps",
    [
        {"a": 0, "b": 0.01},
        {"a": -1, "b": 0.01},
        {"a": np.nan, "b": 0.01},
        {"a": 1e-100, "b": 0.01},
        {"a": 0.01},
    ],
)
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        compute_fisher(toy(), steps=steps)


def test_bounds_blinding_and_unsupported_modes():
    vega = toy()
    vega.sample_params["limits"]["a"] = (1.0, 2.0)
    with pytest.raises(ValueError, match="crosses"):
        compute_fisher(vega, ["a"], steps={"a": 0.01})
    del vega.sample_params["limits"]["a"]
    vega._blind = True
    with pytest.raises(ValueError, match="validated"):
        compute_fisher(vega, ["a"], steps={"a": 0.01})
    for attribute in ["model_pk", "monte_carlo", "marginalize_in_fit"]:
        vega = toy()
        setattr(vega, attribute, True)
        with pytest.raises(ValueError):
            compute_fisher(vega, steps={"a": 0.01, "b": 0.01})


def test_forecast_preserves_actual_parameter_blinding():
    vega = toy()
    vega._blind = True
    vega._rnsps = {"a": 0.2}

    def compute_model(params, run_init=False):
        local = VegaInterface._get_lcl_prms(vega, params)
        return {
            "second": np.array([0.0, local["a"] ** 2, local["a"] ** 2, 0.0]),
            "first": np.array([local["a"] ** 2, 0.0]),
        }

    vega.compute_model = compute_model
    result = compute_fisher(vega, ["a"], steps={"a": 0.001})
    effective = vega.params["a"] + np.pi - np.exp(0.2**2)
    np.testing.assert_allclose(result.jacobian, 2 * effective)
    np.testing.assert_allclose(result.fiducial, [1.0])
    assert result.metadata["blind"]
    assert "_rnsps" not in result.metadata


def test_nonfinite_model_fails_and_restores_fiducial():
    vega = toy()
    compute_model = vega.compute_model

    def broken(params, run_init=False):
        models = compute_model(params, run_init)
        if params["a"] != 1:
            models["first"][0] = np.nan
        return models

    vega.compute_model = broken
    with pytest.raises(ValueError, match="nonfinite"):
        compute_fisher(vega, steps={"a": 0.01, "b": 0.01})
    assert vega.last_params == vega.params


def test_independent_covariances_and_missing_covariance(tmp_path):
    vega = toy()
    vega._use_global_cov = False
    for name, data in vega.data.items():
        # This test supplies an actual FITS CO column, including an identity C.
        count = data.data_mask.size
        path = tmp_path / f"{name}.fits"
        hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="CO", format=f"{count}D", array=np.eye(count)),
            ]
        )
        hdu.writeto(path)
        config = configparser.ConfigParser()
        config["data"] = {"filename": str(path)}
        data.corr_item = SimpleNamespace(config=config)
        data.is_direct_multipoles = False
        data.cov_mat = 3 * np.eye(count)  # includes a configured rescaling
    result = compute_fisher(vega, steps={"a": 0.01, "b": 0.01})
    np.testing.assert_allclose(result.fisher_data, result.jacobian.T @ result.jacobian / 3)
    vega.data["first"].is_direct_multipoles = True
    with pytest.raises(ValueError, match="identity fallback"):
        compute_fisher(vega, steps={"a": 0.01, "b": 0.01})


def test_direct_multipoles_global_full_grid_mask_and_cross_block(tmp_path):
    """Exercise Vega's real NELL/uncut-grid selection before forecasting."""
    vega = toy()
    vega.apply_global_hartlap = False
    vega.corr_items = {
        "second": SimpleNamespace(marginalize_small_scales=False),
        "first": SimpleNamespace(marginalize_small_scales=False),
    }
    data = vega.data["second"]
    data.is_direct_multipoles = False
    data.use_multipoles = False
    mp = vega.data["first"]
    mp.is_direct_multipoles = True
    mp.use_multipoles = True
    mp.nells = 1
    mp.ells_to_model = [2]
    mp._mp_n_ells_file = 2
    mp._mp_n_s_full = 3
    mp._mp_full_s_mask = np.array([False, True, False])
    mp._mp_ell_file_indices = [1]
    mp.data_mask = np.array([True])
    # 3 forest pixels + 2 multipoles x 3 uncut s bins, with cross correlations.
    cov = np.eye(9) + 0.1 * np.ones((9, 9))
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="COV", format="9D", array=cov),
        ]
    )
    hdu.header["NELL"] = 2
    path = tmp_path / "global.fits"
    hdu.writeto(path)
    VegaInterface.read_global_cov(vega, str(path), scale=2.0)
    selected = [0, 2, 7]
    np.testing.assert_array_equal(np.flatnonzero(vega.full_data_mask), selected)
    result = compute_fisher(vega, steps={"a": 0.01, "b": 0.01})
    expected = result.jacobian.T @ np.linalg.solve(
        2 * cov[np.ix_(selected, selected)], result.jacobian
    )
    np.testing.assert_allclose(result.fisher_data, expected)


def qso_config(tmp_path):
    component = configparser.ConfigParser()
    component.optionxform = str
    component.read(find_file("configs/qsoxqso.ini"))
    for option in ("weights-tracer1", "weights-tracer2"):
        component["data"][option] = str(find_file(component["data"][option]))
    component_path = tmp_path / "qso.ini"
    with component_path.open("w") as stream:
        component.write(stream)
    main = configparser.ConfigParser()
    main.optionxform = str
    main.read(find_file("full_configs/main.ini"))
    main["data sets"]["ini files"] = str(component_path)
    main["parameters"]["sigma_velo_disp_gauss_QSO"] = "2.0"
    main["sample"] = {"bias_eta_QSO": "None None 1.0 .01", "beta_QSO": "None None .25 .01"}
    main["fisher"] = {
        "steps": ".001 .001",
        "output": str(tmp_path / "real-fisher.npz"),
    }
    main_path = tmp_path / "main.ini"
    with main_path.open("w") as stream:
        main.write(stream)
    return main_path


def test_real_vega_model_matches_likelihood_curvature(tmp_path):
    main_path = qso_config(tmp_path)
    vega = VegaInterface(str(main_path))
    params_before = copy.deepcopy(vega.params)
    result = compute_fisher(vega, steps={"bias_eta_QSO": 1e-3, "beta_QSO": 1e-3})
    assert result.rank == 2
    # The expected Fisher equals half the Hessian of chi2 for noiseless data
    # at the fiducial, with all actual model transformations left active.
    models = vega.compute_model(run_init=False)
    for name, data in vega.data.items():
        data._masked_data_vec = models[name][data.model_mask].copy()
    step = result.steps[0]
    plus = {"bias_eta_QSO": vega.params["bias_eta_QSO"] + step}
    minus = {"bias_eta_QSO": vega.params["bias_eta_QSO"] - step}
    curvature = (vega.chi2(plus) + vega.chi2(minus) - 2 * vega.chi2()) / (2 * step**2)
    assert curvature == pytest.approx(result.fisher[0, 0], rel=1e-5)
    assert vega.params == params_before
    from_ini = run_fisher(str(main_path))
    np.testing.assert_allclose(from_ini.fisher, result.fisher)


def test_cli_adapter_calls_same_engine(monkeypatch, tmp_path):
    vega = toy()
    config = configparser.ConfigParser()
    config["fisher"] = {
        "parameters": "b a",
        "steps": ".01 .001",
        "fiducial": "2.5 1.5",
        "output": str(tmp_path / "cli.npz"),
        "include-priors": "False",
    }
    vega.main_config = config
    monkeypatch.setattr("vega.scripts.run_vega_fisher.VegaInterface", lambda path: vega)
    result = run_fisher("unused.ini")
    expected = compute_fisher(
        vega, ["b", "a"], steps={"b": 0.01, "a": 0.001}, fiducial={"b": 2.5, "a": 1.5}, priors={}
    )
    np.testing.assert_allclose(result.fisher, expected.fisher)
    with np.load(tmp_path / "cli.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["fiducial"], [2.5, 1.5])
    config["fisher"]["stpes"] = ".1 .1"
    with pytest.raises(ValueError, match="Unknown"):
        run_fisher("unused.ini")


def test_module_cli_smoke(tmp_path):
    main_path = qso_config(tmp_path)
    process = subprocess.run(
        [sys.executable, "-m", "vega.scripts.run_vega_fisher", str(main_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Fisher rank 2/2" in process.stdout
    with np.load(tmp_path / "real-fisher.npz", allow_pickle=False) as saved:
        assert saved["fisher"].shape == (2, 2)
