"""Check DR3 blinding metadata, parameter selection and blind-data restrictions."""

import configparser
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.io import fits

from vega import utils
from vega.correlation_item import CorrelationItem
from vega.data import Data
from vega.vega_interface import VegaInterface


@pytest.fixture(autouse=True)
def _redirect_blinding_dir(monkeypatch, tmp_path):
    """Point parameter-blinding file selection to an empty temporary directory.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing the module-level blinding directory.
    tmp_path : pathlib.Path
        Empty temporary directory supplied by pytest.
    """
    monkeypatch.setattr(utils, "PARAMETER_BLINDING_DIR", tmp_path / "blinding")


def _write_correlation(data_path, strategy, data_column="DA_BLIND"):
    """Write a four-bin synthetic correlation with identity matrices.

    Parameters
    ----------
    data_path : pathlib.Path
        Temporary output FITS path.
    strategy : str or None
        Header blinding strategy; ``None`` omits the header entry.
    data_column : str, optional
        Correlation column name, by default ``DA_BLIND``.

    Returns
    -------
    numpy.ndarray
        Dimensionless correlation values, shape (4,).
    """
    correlation = np.array([-0.03, 0.02, 0.012, -0.004])
    columns = [
        fits.Column(name=data_column, format="D", array=correlation),
        fits.Column(name="RP", format="D", array=[20.0, 60.0, 20.0, 60.0]),
        fits.Column(name="RT", format="D", array=[20.0, 20.0, 60.0, 60.0]),
        fits.Column(name="Z", format="D", array=np.full(4, 2.2)),
        fits.Column(name="CO", format="4D", array=np.eye(4)),
        fits.Column(name="DM_BLIND", format="4D", array=np.eye(4)),
    ]
    correlation_hdu = fits.BinTableHDU.from_columns(columns)
    correlation_hdu.header.update(RPMIN=0.0, RPMAX=80.0, RTMAX=80.0, NP=2, NT=2)
    if strategy is not None:
        correlation_hdu.header["BLINDING"] = strategy
    fits.HDUList([fits.PrimaryHDU(), correlation_hdu]).writeto(data_path)
    return correlation


def _read_correlation(data_path):
    """Read synthetic correlation data without constructing a fitting interface.

    Parameters
    ----------
    data_path : pathlib.Path
        Temporary input FITS path.

    Returns
    -------
    Data
        Data instance initialized only by the correlation reader.
    """
    cuts = configparser.ConfigParser()
    cuts.add_section("cuts")
    data = Data.__new__(Data)
    data._apply_hartlap = False
    data.use_multipoles = False
    data._read_data(str(data_path), cuts["cuts"])
    return data


def _blinding_interface(
    parameters, strategy="desi_dr3_civ", tracer_names=("CIV", "QSO"), tracer_types=(None, None)
):
    """Construct only the state required for parameter-blinding initialization.

    Parameters
    ----------
    parameters : tuple of str
        Names of sampled parameters.
    strategy : str, optional
        Active data strategy, by default ``desi_dr3_civ``.
    tracer_names : tuple of str, optional
        Primary measurement tracers, by default ``("CIV", "QSO")``.
    tracer_types : tuple of str or None, optional
        Tracer types (``continuous`` or ``discrete``), by default unset.

    Returns
    -------
    VegaInterface
        Lightweight interface without models, minimizers or parameter offsets.
    """
    correlation = CorrelationItem.__new__(CorrelationItem)
    correlation.tracer1 = {"name": tracer_names[0], "type": tracer_types[0]}
    correlation.tracer2 = {"name": tracer_names[1], "type": tracer_types[1]}

    interface = VegaInterface.__new__(VegaInterface)
    interface._blind = False
    interface._rnsps = None
    interface.data = {"correlation": SimpleNamespace(blind=True, blinding_strat=strategy)}
    interface.corr_items = {"correlation": correlation}
    interface.sample_params = {"limits": dict.fromkeys(parameters, (0.5, 1.5))}
    return interface


@pytest.mark.parametrize("strategy", ["desi_dr3_civ", "desi_dr3"])
def test_active_strategy_preserves_correlation(tmp_path, strategy):
    """Require an active flag while leaving synthetic correlation values unchanged.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    strategy : str
        Active DR3 blinding strategy.
    """
    data_path = tmp_path / "correlation.fits"
    correlation = _write_correlation(data_path, strategy)
    data = _read_correlation(data_path)

    assert data.blind is True
    assert data.blinding_strat == strategy
    np.testing.assert_array_equal(data.data_vec, correlation)
    np.testing.assert_array_equal(data.cov_mat, np.eye(4))
    np.testing.assert_array_equal(data.distortion_mat.toarray(), np.eye(4))


@pytest.mark.parametrize("strategy", ["desi_dr3_civ", "desi_dr3"])
def test_active_strategy_requires_blind_column(tmp_path, strategy):
    """Reject active DR3 data lacking the expected correlation column.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    strategy : str
        Active DR3 blinding strategy.
    """
    data_path = tmp_path / "correlation.fits"
    _write_correlation(data_path, strategy, data_column="DA")
    with pytest.raises(AssertionError, match="Blinding failed, do not run!!!"):
        _read_correlation(data_path)


@pytest.mark.parametrize("strategy", [None, "none", "desi_m2", "desi_y1", "desi_y3"])
def test_legacy_data_strategies_are_unchanged(tmp_path, strategy):
    """Retain unblinded data reading for legacy and absent strategies.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.
    strategy : str or None
        Legacy, unblinded or absent strategy.
    """
    data_path = tmp_path / "correlation.fits"
    correlation = _write_correlation(data_path, strategy, data_column="DA")
    data = _read_correlation(data_path)

    assert data.blind is False
    assert data.blinding_strat == (None if strategy in (None, "none") else strategy)
    np.testing.assert_array_equal(data.data_vec, correlation)


@pytest.mark.parametrize("parameters", [("ap", "at"), ("alpha",)])
@pytest.mark.parametrize(
    "tracer_names", [("CIV", "QSO"), ("civ", "QSO"), ("CIV", "CIV"), ("QSO", "CIV")]
)
def test_civ_bao_parameter_selection(monkeypatch, parameters, tracer_names):
    """Route sampled CIV BAO parameters and the exact strategy to file selection.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing file selection with a recorder.
    parameters : tuple of str
        Sampled anisotropic or isotropic BAO parameters.
    tracer_names : tuple of str
        Primary correlation tracer names.
    """
    get_blinding = Mock(return_value=None)
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(parameters, tracer_names=tracer_names)
    interface._init_blinding()

    assert interface._blind is True
    get_blinding.assert_called_once_with(list(parameters), "desi_dr3_civ")


@pytest.mark.parametrize("parameters", [("ap", "at"), ("alpha",)])
def test_civ_bao_missing_file_error_precedes_file_access(monkeypatch, parameters):
    """Raise the file-not-found error for CIV BAO offsets before opening any file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing NumPy file loading with a forbidden operation.
    parameters : tuple of str
        Sampled anisotropic or isotropic BAO parameters.
    """
    load_parameters = Mock(side_effect=AssertionError("Parameter resources must not be opened."))
    monkeypatch.setattr(utils.np, "load", load_parameters)
    interface = _blinding_interface(parameters)

    with pytest.raises(
        ValueError, match=r"^Blinding file not found: .*dr3_bao_CIV_blinding_12_10_2026\.npz\."
    ):
        interface._init_blinding()
    load_parameters.assert_not_called()


@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_dr3_civ"])
def test_dr3_full_shape_requires_configured_file(monkeypatch, strategy):
    """Reject DR3 full-shape fits while no full-shape offsets are configured.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing NumPy file loading with a forbidden operation.
    strategy : str
        Active DR3 blinding strategy.
    """
    load_parameters = Mock(side_effect=AssertionError("Parameter resources must not be opened."))
    monkeypatch.setattr(utils.np, "load", load_parameters)
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", None)
    interface = _blinding_interface(("phi_smooth", "growth_rate"), strategy=strategy)

    with pytest.raises(
        ValueError, match=rf"^No parameter-blinding file is configured for {strategy} full-shape\.$"
    ):
        interface._init_blinding()
    load_parameters.assert_not_called()


@pytest.mark.parametrize("parameters", [("ap", "at", "growth_rate"), ("alpha", "phi_smooth")])
def test_mixed_bao_and_full_shape_parameters_are_rejected(monkeypatch, parameters):
    """Reject requests that would read full-shape offsets from the BAO file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing NumPy file loading with a forbidden operation.
    parameters : tuple of str
        Sampled CIV BAO and full-shape parameters.
    """
    load_parameters = Mock(side_effect=AssertionError("Parameter resources must not be opened."))
    monkeypatch.setattr(utils.np, "load", load_parameters)
    interface = _blinding_interface(parameters)

    with pytest.raises(ValueError, match=r"^Cannot blind BAO and full-shape parameters together"):
        interface._init_blinding()
    load_parameters.assert_not_called()


@pytest.mark.parametrize("parameter", ["phi", "aiso", "epsilon", "aap"])
def test_civ_rejects_unblinded_bao_parametrisations(monkeypatch, parameter):
    """Require BAO parameters without CIV offsets to be fixed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    parameter : str
        Sampled BAO parameter of the phi_alpha, aiso_epsilon or aiso_aap
        parametrisations.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface((parameter, "alpha"))

    with pytest.raises(ValueError, match=rf"desi_dr3_civ blind data, parameter {parameter} must"):
        interface._init_blinding()
    get_blinding.assert_not_called()


@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_y1", "desi_y3"])
def test_other_strategies_allow_alternative_bao_parametrisations(monkeypatch, strategy):
    """Leave phi, aiso, epsilon and aap unrestricted outside desi_dr3_civ.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    strategy : str
        Strategy other than ``desi_dr3_civ``.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(("phi", "aiso", "epsilon", "aap"), strategy=strategy)
    interface._init_blinding()
    get_blinding.assert_not_called()
    assert interface._rnsps is None


@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_y1", "desi_y3"])
def test_other_strategies_do_not_select_civ_bao(monkeypatch, strategy):
    """Leave CIV BAO parameters unselected under other blinding strategies.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    strategy : str
        Strategy other than ``desi_dr3_civ``.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(("ap", "at", "alpha"), strategy=strategy)
    interface._init_blinding()
    get_blinding.assert_not_called()
    assert interface._rnsps is None


@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_dr3_civ"])
def test_full_shape_parameter_selection_is_unchanged(monkeypatch, strategy):
    """Retain the sampled full-shape parameter selection under both DR3 schemes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    strategy : str
        Active DR3 blinding strategy.
    """
    get_blinding = Mock(return_value=None)
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(("phi_smooth", "growth_rate"), strategy=strategy)
    interface._init_blinding()
    get_blinding.assert_called_once_with(["phi_smooth", "growth_rate"], strategy)


@pytest.mark.parametrize("parameters", [("bias_CIV",), ()])
def test_no_protected_parameters_requires_no_file(monkeypatch, parameters):
    """Keep initialization valid when no sampled parameter requires blinding.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    parameters : tuple of str
        Unprotected sampled parameters, or an empty tuple.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(parameters)
    interface._init_blinding()
    get_blinding.assert_not_called()
    assert interface._blind is True
    assert interface._rnsps is None


def test_non_civ_tracers_do_not_select_bao(monkeypatch):
    """Preserve primary-tracer selection even if CIV metadata are supplied.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(("ap", "at"), tracer_names=("LYA", "QSO"))
    interface._init_blinding()
    get_blinding.assert_not_called()


@pytest.mark.parametrize("parameters", [("ap", "at"), ("alpha",), ("phi_smooth",)])
@pytest.mark.parametrize("strategy", ["desi_y1", "desi_y3"])
def test_legacy_unset_files_remain_unblinded(monkeypatch, parameters, strategy):
    """Retain the legacy return value for unset parameter files.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording NumPy file-loading calls.
    parameters : tuple of str
        Parameters accepted by the legacy file selector.
    strategy : str
        Legacy strategy with no active parameter file.
    """
    load_parameters = Mock()
    monkeypatch.setattr(utils.np, "load", load_parameters)
    assert utils.get_blinding(list(parameters), strategy) is None
    load_parameters.assert_not_called()


def test_standard_dr3_bao_file_selection_returns_none(monkeypatch):
    """Return no offsets for BAO parameters under the standard DR3 strategy.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording NumPy file-loading calls.
    """
    load_parameters = Mock()
    monkeypatch.setattr(utils.np, "load", load_parameters)
    assert utils.get_blinding(["ap", "at"], "desi_dr3") is None
    load_parameters.assert_not_called()


@pytest.mark.parametrize("parameters", [("ap_full",), ("bias_QSO", "beta_QSO")])
def test_existing_blind_parameter_restrictions(parameters):
    """Retain restrictions on fixed parameters and on sampling the QSO RSD parameter.

    Parameters
    ----------
    parameters : tuple of str
        Sampled parameters violating an existing blind-data restriction.
    """
    interface = _blinding_interface(parameters)
    with pytest.raises(ValueError, match="Running on blind data"):
        interface._init_blinding()


def test_mixed_active_strategies_are_rejected():
    """Reject combining standard DR3 and CIV DR3 strategies in one fit."""
    interface = _blinding_interface(("ap", "at"))
    interface.data["other"] = SimpleNamespace(blind=True, blinding_strat="desi_dr3")
    with pytest.raises(ValueError, match="Different blinding strategies found in the data sets"):
        interface._init_blinding()


def _write_full_shape_offsets(offsets):
    """Write synthetic DR3 full-shape offsets in the obfuscated format of blind_fs.py.

    Parameters
    ----------
    offsets : dict of str to float
        Additive offsets keyed by parameter name.

    Returns
    -------
    pathlib.Path
        Path of the written file inside the redirected blinding directory.
    """
    file_name = "dr3_fs_blinding_test.npz"
    output_dir = utils.PARAMETER_BLINDING_DIR / "full-shape-blinding"
    output_dir.mkdir(parents=True)

    # blind_fs.py stores sqrt(log(pi - offset)), inverted by apply_blinding
    obfuscated = {par: np.sqrt(np.log(np.pi - offset)) for par, offset in offsets.items()}
    np.savez(output_dir / file_name, seed=12345, **obfuscated)
    return output_dir / file_name


def test_dr3_full_shape_file_selection(monkeypatch):
    """Read and apply DR3 full-shape offsets from the configured file name.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the DR3 full-shape file name.
    """
    offsets = {"phi_smooth": 0.031, "growth_rate": -0.27, "alpha_smooth": 0.02}
    offsets_path = _write_full_shape_offsets(offsets)
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", offsets_path.name)

    blinding = utils.get_blinding(["phi_smooth", "growth_rate"], "desi_dr3")
    assert set(blinding) == {"phi_smooth", "growth_rate"}

    params = {"phi_smooth": 1.0, "growth_rate": 0.97, "bias_LYA": -0.1}
    shifted_params = utils.apply_blinding(dict(params), blinding)
    assert shifted_params["phi_smooth"] == pytest.approx(1.0 + offsets["phi_smooth"], abs=1e-12)
    assert shifted_params["growth_rate"] == pytest.approx(0.97 + offsets["growth_rate"], abs=1e-12)
    assert shifted_params["bias_LYA"] == params["bias_LYA"]


def test_dr3_full_shape_missing_file_is_rejected(monkeypatch):
    """Abort when the configured DR3 full-shape file does not exist.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the DR3 full-shape file name.
    """
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", "dr3_fs_blinding_missing.npz")
    with pytest.raises(ValueError, match=r"^Blinding file not found: .*dr3_fs_blinding_missing"):
        utils.get_blinding(["phi_smooth", "growth_rate"], "desi_dr3")


def test_civ_full_shape_ignores_dr3_file(monkeypatch):
    """Keep desi_dr3_civ full-shape fits blocked when the DR3 file is configured.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the DR3 full-shape file name.
    """
    offsets_path = _write_full_shape_offsets({"phi_smooth": 0.01, "growth_rate": 0.1})
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", offsets_path.name)
    with pytest.raises(
        ValueError, match=r"^No parameter-blinding file is configured for desi_dr3_"
    ):
        utils.get_blinding(["phi_smooth", "growth_rate"], "desi_dr3_civ")


@pytest.mark.parametrize(
    "parameters",
    [
        ("bias_eta_QSO",),
        ("bias_QSO", "bias_eta_QSO"),
        ("beta_QSO", "bias_eta_QSO"),
        ("bias_QSO", "bias_eta_QSO", "growth_rate"),
        ("bias_DLA", "beta_DLA"),
        ("bias_SBLA", "bias_eta_SBLA"),
        ("beta_QSO",),
        ("beta_DLA", "growth_rate"),
    ],
)
@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_dr3_civ"])
def test_discrete_tracer_rsd_bypass_is_rejected(monkeypatch, parameters, strategy):
    """Reject discrete-tracer RSD parametrisations that measure f without the offset.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    parameters : tuple of str
        Sampled bias, beta and velocity-bias parameters of a discrete tracer.
    strategy : str
        Active DR3 blinding strategy.
    """
    get_blinding = Mock()
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(parameters, strategy=strategy, tracer_names=("LYA", "QSO"))

    with pytest.raises(ValueError, match="Running on blind data and sampling (bias_eta|beta)_"):
        interface._init_blinding()
    get_blinding.assert_not_called()


def test_discrete_tracer_identified_by_type(monkeypatch):
    """Apply the RSD restriction to discrete tracers outside the default name list.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    """
    monkeypatch.setattr(utils, "get_blinding", Mock())
    interface = _blinding_interface(
        ("bias_LRG", "beta_LRG"),
        strategy="desi_dr3",
        tracer_names=("LYA", "LRG"),
        tracer_types=("continuous", "discrete"),
    )
    with pytest.raises(ValueError, match="sampling beta_LRG"):
        interface._init_blinding()


@pytest.mark.parametrize(
    "parameters",
    [
        ("bias_QSO",),
        ("bias_QSO", "growth_rate"),
        ("bias_LYA", "bias_eta_LYA"),
        ("bias_LYA", "beta_LYA"),
    ],
)
def test_allowed_rsd_parametrisations(monkeypatch, parameters):
    """Allow a sampled QSO bias and any Lya bias parametrisation on blind data.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    parameters : tuple of str
        Sampled bias parameters that keep the growth rate blinded.
    """
    monkeypatch.setattr(utils, "get_blinding", Mock(return_value=None))
    interface = _blinding_interface(
        parameters,
        strategy="desi_dr3",
        tracer_names=("LYA", "QSO"),
        tracer_types=("continuous", "discrete"),
    )
    interface._init_blinding()
    assert interface._blind is True


@pytest.mark.parametrize("prior_parameter", ["growth_rate", "phi_smooth"])
def test_priors_on_blinded_parameters_are_rejected(prior_parameter):
    """Reject Gaussian priors on parameters that receive blinding offsets.

    Parameters
    ----------
    prior_parameter : str
        Blinded parameter with a prior.
    """
    interface = _blinding_interface(("phi_smooth", "growth_rate"), strategy="desi_dr3")
    interface._rnsps = {"phi_smooth": 1.0, "growth_rate": 1.0}
    interface.priors = {prior_parameter: np.array([1.0, 0.1]), "bias_LYA": np.array([-0.1, 0.01])}

    with pytest.raises(ValueError, match=rf"priors on blinded parameters \['{prior_parameter}'\]"):
        interface._check_blinded_priors()


@pytest.mark.parametrize("offsets", [None, {"growth_rate": 1.0}])
def test_priors_on_unblinded_parameters_are_allowed(offsets):
    """Allow priors when no offsets are active or on parameters without offsets.

    Parameters
    ----------
    offsets : dict or None
        Active parameter offsets, or None for unblinded fits.
    """
    interface = _blinding_interface(("phi_smooth", "bias_LYA"), strategy="desi_dr3")
    interface._rnsps = offsets
    interface.priors = {"phi_smooth": np.array([1.0, 0.1]), "bias_LYA": np.array([-0.1, 0.01])}

    # Neither prior parameter carries an offset, so the check passes
    interface._check_blinded_priors()


@pytest.mark.parametrize(
    "parameters",
    [
        ("phi_smooth", "alpha_smooth", "growth_rate"),
        ("alpha_smooth",),
        ("phi_smooth", "alpha_smooth_LYAxLYA", "alpha_smooth_LYAxQSO"),
    ],
)
def test_alpha_smooth_parameter_selection(monkeypatch, parameters):
    """Select sampled isotropic smooth dilations for full-shape blinding under desi_dr3.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    parameters : tuple of str
        Sampled full-shape parameters, including per-correlation dilations.
    """
    get_blinding = Mock(return_value=None)
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(parameters, strategy="desi_dr3")
    interface._init_blinding()
    get_blinding.assert_called_once_with(list(parameters), "desi_dr3")


def test_fixed_alpha_smooth_is_not_blinded(monkeypatch):
    """Leave a fixed alpha_smooth without offset so the model is unchanged.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture recording calls to parameter file selection.
    """
    get_blinding = Mock(return_value=None)
    monkeypatch.setattr(utils, "get_blinding", get_blinding)
    interface = _blinding_interface(("phi_smooth", "growth_rate"), strategy="desi_dr3")
    interface._init_blinding()
    get_blinding.assert_called_once_with(["phi_smooth", "growth_rate"], "desi_dr3")


def test_alpha_smooth_offsets_applied(monkeypatch):
    """Apply the stored alpha_smooth offset to alpha_smooth and to every per-correlation dilation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the DR3 full-shape file name.
    """
    offsets = {"phi_smooth": -0.02, "growth_rate": 0.15, "alpha_smooth": 0.043}
    offsets_path = _write_full_shape_offsets(offsets)
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", offsets_path.name)

    blind_pars = ["alpha_smooth", "alpha_smooth_LYAxLYA", "alpha_smooth_LYAxQSO", "phi_smooth"]
    blinding = utils.get_blinding(blind_pars, "desi_dr3")
    assert set(blinding) == set(blind_pars)

    params = dict.fromkeys(blind_pars, 1.0)
    shifted_params = utils.apply_blinding(params, blinding)
    for par in ["alpha_smooth", "alpha_smooth_LYAxLYA", "alpha_smooth_LYAxQSO"]:
        assert shifted_params[par] == pytest.approx(1.0 + offsets["alpha_smooth"], abs=1e-12)
    assert shifted_params["phi_smooth"] == pytest.approx(1.0 + offsets["phi_smooth"], abs=1e-12)


def test_missing_alpha_smooth_offset_is_rejected(monkeypatch):
    """Abort when the configured file lacks an offset for a sampled parameter.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the DR3 full-shape file name.
    """
    offsets_path = _write_full_shape_offsets({"phi_smooth": 0.01, "growth_rate": 0.1})
    monkeypatch.setattr(utils, "DR3_FS_BLINDING_FILE", offsets_path.name)
    with pytest.raises(ValueError, match="has no offset for alpha_smooth"):
        utils.get_blinding(["phi_smooth", "alpha_smooth"], "desi_dr3")


@pytest.mark.parametrize(
    "parameters", [("ap", "at", "alpha_smooth"), ("alpha", "alpha_smooth_CIVxQSO")]
)
def test_civ_bao_and_alpha_smooth_are_rejected(monkeypatch, parameters):
    """Reject requests that combine CIV BAO offsets and the alpha_smooth offset.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing NumPy file loading with a forbidden operation.
    parameters : tuple of str
        Sampled CIV BAO parameters and isotropic smooth dilations.
    """
    load_parameters = Mock(side_effect=AssertionError("Parameter resources must not be opened."))
    monkeypatch.setattr(utils.np, "load", load_parameters)
    interface = _blinding_interface(parameters)

    with pytest.raises(ValueError, match=r"^Cannot blind BAO and full-shape parameters together"):
        interface._init_blinding()
    load_parameters.assert_not_called()


@pytest.mark.parametrize("strategy", ["desi_dr3", "desi_dr3_civ"])
def test_alpha_full_must_be_fixed(strategy):
    """Require alpha_full to be fixed, since it bypasses the alpha_smooth offset.

    Parameters
    ----------
    strategy : str
        Active DR3 blinding strategy.
    """
    interface = _blinding_interface(("alpha_full", "phi_smooth"), strategy=strategy)
    with pytest.raises(ValueError, match="parameter alpha_full must be fixed"):
        interface._init_blinding()


def test_priors_on_per_correlation_alpha_smooth_are_rejected():
    """Reject priors on per-correlation dilations that receive the alpha_smooth offset."""
    interface = _blinding_interface(("alpha_smooth_LYAxLYA",), strategy="desi_dr3")
    interface._rnsps = {"alpha_smooth_LYAxLYA": 1.0}
    interface.priors = {"alpha_smooth_LYAxLYA": np.array([1.0, 0.05])}
    with pytest.raises(ValueError, match="priors on blinded parameters"):
        interface._check_blinded_priors()
