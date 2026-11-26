"""Selection of the velocity-dispersion kernels of the discrete tracer in PowerSpectrum.compute."""

import configparser

import numpy as np
from astropy.io import fits

from vega import utils
from vega.power_spectrum import PowerSpectrum

PARAMS = {
    "bias_LYA": -0.12,
    "beta_LYA": 1.6,
    "bias_QSO": 3.7,
    "beta_QSO": 0.25,
    "sigma_velo_disp_gauss_QSO": 4.0,
    "sigma_velo_disp_lorentz_QSO": 6.0,
    "peak": False,
}


def cross_power_spectrum(velocity_dispersion):
    """Lyα×QSO PowerSpectrum and the smooth linear P(k) of the PlanckDR16 template.

    Parameters
    ----------
    velocity_dispersion : str or None
        Value of 'velocity dispersion' in [model]; None leaves it out (no kernel).

    Returns
    -------
    tuple
        (PowerSpectrum, smooth linear P(k) array)
    """
    template = utils.find_file("PlanckDR16/PlanckDR16.fits")
    with fits.open(template) as hdul:
        fiducial = {
            "z_eff": 2.25,
            "k": hdul[1].data["K"],
            "pk_full": hdul[1].data["PK"],
            "pk_smooth": hdul[1].data["PKSB"],
            "z_fiducial": hdul[1].header["ZREF"],
        }
    config = configparser.ConfigParser()
    config["model"] = {"bin_size_rp": "4", "bin_size_rt": "4", "model binning": "False"}
    if velocity_dispersion is not None:
        config["model"]["velocity dispersion"] = velocity_dispersion
    tracer1 = {"name": "LYA", "type": "continuous"}
    tracer2 = {"name": "QSO", "type": "discrete"}
    pk = PowerSpectrum(config["model"], fiducial, tracer1, tracer2, "lyaxqso")
    return pk, fiducial["pk_smooth"]


def test_lorentz_gauss_applies_both_kernels():
    """'lorentz_gauss' multiplies the undamped P(k) by the Lorentzian and the Gaussian kernels."""
    pk_none, pk_lin = cross_power_spectrum(None)
    undamped = pk_none.compute(pk_lin, PARAMS)

    pk_both, _ = cross_power_spectrum("lorentz_gauss")
    expected = (
        undamped
        * pk_both.compute_velocity_dispersion_lorentz(PARAMS)
        * pk_both.compute_velocity_dispersion_gauss(PARAMS)
    )
    assert np.allclose(pk_both.compute(pk_lin, PARAMS), expected, rtol=1e-12, atol=0)


def test_single_kernels_unchanged():
    """'lorentz' and 'gauss' still apply only their own kernel."""
    pk_none, pk_lin = cross_power_spectrum(None)
    undamped = pk_none.compute(pk_lin, PARAMS)
    for name, kernel in (
        ("lorentz", PowerSpectrum.compute_velocity_dispersion_lorentz),
        ("gauss", PowerSpectrum.compute_velocity_dispersion_gauss),
    ):
        pk, _ = cross_power_spectrum(name)
        expected = undamped * kernel(pk, PARAMS)
        assert np.allclose(pk.compute(pk_lin, PARAMS), expected, rtol=1e-12, atol=0)
