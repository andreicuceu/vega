"""Tests of the per-correlation QSO radiation amplitude (radiation-strength-name, -scale) and of
QSO sub-sample tracers (discrete tracers named QSO*) in Lya x QSO."""

import configparser

import numpy as np

from vega import VegaInterface
from vega.utils import find_file

STRENGTH = 0.74


def _write_configs(tmp_path, tag, model_options=None, tracer_name="QSO", marginalize=None):
    """One-dataset copy of the full Lya x QSO test configs, without metals.

    Parameters
    ----------
    tmp_path : pathlib.Path
    tag : str
        Distinguishes the files of different variants.
    model_options : dict, optional
        Extra [model] options of the correlation config.
    tracer_name : str, optional
        Name of the discrete tracer; parameters named *_QSO are copied to *_<tracer_name>.
    marginalize : str, optional
        Parameter listed in [analytic marginalization] of the main config.

    Returns
    -------
    pathlib.Path
        Path of the main config
    """
    corr_config = configparser.ConfigParser()
    corr_config.optionxform = str
    corr_config.read(find_file("full_configs/lyalya_qso.ini"))
    corr_config.remove_section("metals")
    corr_config["data"]["tracer2"] = tracer_name
    for key in list(corr_config["model"]):
        if key == "z evol QSO":
            corr_config["model"][f"z evol {tracer_name}"] = corr_config["model"].pop(key)
    for key, value in (model_options or {}).items():
        corr_config["model"][key] = value
    corr_path = tmp_path / f"lyalya_qso_{tag}.ini"
    with open(corr_path, "w") as corr_file:
        corr_config.write(corr_file)

    main_config = configparser.ConfigParser()
    main_config.optionxform = str
    main_config.read(find_file("full_configs/main.ini"))
    main_config["data sets"]["ini files"] = str(corr_path)
    main_config["output"]["filename"] = str(tmp_path / f"output_{tag}")
    if main_config.has_section("Polychord"):
        main_config.remove_section("Polychord")
    parameters = main_config["parameters"]
    for key in [key for key in parameters if key.endswith("_QSO")]:
        parameters[key[: -len("QSO")] + tracer_name] = parameters[key]
    strength_name = (model_options or {}).get("radiation-strength-name", "qso_rad_strength")
    parameters[strength_name] = str(STRENGTH)
    if marginalize is not None:
        main_config["analytic marginalization"] = {marginalize: "True"}
    main_path = tmp_path / f"main_{tag}.ini"
    with open(main_path, "w") as main_file:
        main_config.write(main_file)
    return main_path


def _model(main_path, params=None):
    """Model of the single correlation at the config parameters updated with params."""
    vega = VegaInterface(str(main_path))
    return vega.compute_model({**dict(vega.params), **(params or {})})["lyalya_qso"]


def test_renamed_strength_reproduces_the_default(tmp_path):
    default = _model(_write_configs(tmp_path, "default"))
    renamed_path = _write_configs(
        tmp_path, "renamed", {"radiation-strength-name": "qso_rad_strength_X"}
    )
    assert np.allclose(_model(renamed_path), default, rtol=1e-10, atol=0)

    # The renamed parameter controls the term; the default name no longer does
    assert not np.allclose(_model(renamed_path, {"qso_rad_strength_X": 0.0}), default)
    assert np.allclose(_model(renamed_path, {"qso_rad_strength": 0.0}), default, rtol=1e-10, atol=0)


def test_strength_scale_multiplies_the_amplitude(tmp_path):
    scale = 0.3
    scaled = _model(_write_configs(tmp_path, "scaled", {"radiation-strength-scale": str(scale)}))
    reference_path = _write_configs(tmp_path, "reference")
    reference = _model(reference_path, {"qso_rad_strength": scale * STRENGTH})
    assert np.allclose(scaled, reference, rtol=1e-10, atol=0)


def test_qso_subsample_tracer_name(tmp_path):
    """A discrete tracer named QSOF with copies of the QSO parameters gives the same model."""
    default = _model(_write_configs(tmp_path, "qso"))
    subsample = _model(_write_configs(tmp_path, "qsof", tracer_name="QSOF"))
    assert np.allclose(subsample, default, rtol=1e-10, atol=0)


def test_analytic_marginalization_of_the_renamed_strength(tmp_path):
    default = VegaInterface(str(_write_configs(tmp_path, "marg", marginalize="qso_rad_strength")))
    renamed = VegaInterface(
        str(
            _write_configs(
                tmp_path,
                "marg_renamed",
                {"radiation-strength-name": "qso_rad_strength_X"},
                marginalize="qso_rad_strength_X",
            )
        )
    )
    assert np.isclose(renamed.chi2(renamed.params), default.chi2(default.params), rtol=1e-10)
    assert renamed.marginalization.linear.names == ("qso_rad_strength_X",)
