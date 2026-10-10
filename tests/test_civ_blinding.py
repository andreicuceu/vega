"""Check CIV blinding metadata and parameter selection without evaluating a model."""

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


def _blinding_interface(parameters, strategy="desi_dr3_civ", tracer_names=("CIV", "QSO")):
    """Construct only the state required for parameter-blinding initialization.

    Parameters
    ----------
    parameters : tuple of str
        Names of sampled parameters.
    strategy : str, optional
        Active data strategy, by default ``desi_dr3_civ``.
    tracer_names : tuple of str, optional
        Primary measurement tracers, by default ``("CIV", "QSO")``.

    Returns
    -------
    VegaInterface
        Lightweight interface without models, minimizers or parameter offsets.
    """
    correlation = CorrelationItem.__new__(CorrelationItem)
    correlation.tracer1 = {"name": tracer_names[0]}
    correlation.tracer2 = {"name": tracer_names[1]}

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


@pytest.mark.parametrize("parameters", [("ap", "at"), ("alpha",), ("phi_smooth", "growth_rate")])
def test_civ_missing_file_error_precedes_file_access(monkeypatch, parameters):
    """Raise the explicit unset-file error before opening parameter resources.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing NumPy file loading with a forbidden operation.
    parameters : tuple of str
        Sampled BAO or full-shape parameters.
    """
    load_parameters = Mock(side_effect=AssertionError("Parameter resources must not be opened."))
    monkeypatch.setattr(utils.np, "load", load_parameters)
    interface = _blinding_interface(parameters)

    with pytest.raises(
        ValueError, match=r"^No parameter-blinding file is configured for desi_dr3_civ\.$"
    ):
        interface._init_blinding()
    load_parameters.assert_not_called()


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


def test_standard_dr3_file_selection_is_unchanged():
    """Preserve the existing error for the unconfigured standard DR3 strategy."""
    with pytest.raises(ValueError, match=r"^Unknown blinding version: desi_dr3\.$"):
        utils.get_blinding(["phi_smooth"], "desi_dr3")


@pytest.mark.parametrize("parameters", [("ap_full",), ("bias_QSO", "beta_QSO")])
def test_existing_blind_parameter_restrictions(parameters):
    """Retain restrictions on fixed parameters and simultaneous QSO bias sampling.

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
