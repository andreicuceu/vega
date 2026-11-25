"""Tests of the asymmetric redshift-error kernels of the cross-correlation model."""

import configparser

import numpy as np
import pytest
from astropy.io import fits
from scipy import interpolate

from vega import VegaInterface
from vega.correlation_func import CorrelationFunction
from vega.pktoxi import PktoXi
from vega.utils import find_file

SIGMA = 3.0
R_TRANS = 30.0
R_PAR = np.arange(-100.0, 100.1, 8.0)


@pytest.fixture(scope="module")
def cross_setup():
    """PktoXi and a Lya x QSO-like anisotropic power spectrum with the symmetric Lorentzian kernel.

    Returns
    -------
    dict
        'pktoxi', 'k_par' (on the (muk, k) grid), 'pk_true' (no redshift errors) and 'pk_sym'
        (times the symmetric kernel)
    """
    with fits.open(find_file("Planck18/DESI-2024_z_2.33.fits")) as hdul:
        k_grid = np.array(hdul[1].data["K"])
        pk_lin = np.array(hdul[1].data["PK"])
    config = configparser.ConfigParser()
    config.read_dict({"model": {}})
    muk_grid = ((np.arange(400) + 0.5) / 400)[:, None]
    pktoxi = PktoXi(k_grid, muk_grid, config["model"])
    k_par = k_grid * muk_grid
    kaiser = -0.17 * (1 + 1.2 * muk_grid**2) * 3.6 * (1 + 0.27 * muk_grid**2)
    pk_true = kaiser * pk_lin
    pk_sym = pk_true / np.sqrt(1 + (k_par * SIGMA) ** 2)
    return {"pktoxi": pktoxi, "k_par": k_par, "pk_true": pk_true, "pk_sym": pk_sym}


def _xi(setup, pk, r_par, odd=False):
    """Even- or odd-path correlation function at (r_par, R_TRANS)."""
    r_grid = np.hypot(r_par, R_TRANS)
    if odd:
        return setup["pktoxi"].compute_odd(r_grid, r_par / r_grid, pk)
    return setup["pktoxi"].compute(r_grid, r_par / r_grid, pk)


def _kernel_ratios(form, params, k_par):
    """zerr_kernel_ratios of a bare CorrelationFunction with the given form."""
    correlation = CorrelationFunction.__new__(CorrelationFunction)
    correlation.zerr_asymmetry = form
    correlation._zerr_tracer = "QSO"
    return correlation.zerr_kernel_ratios(params, k_par)


def test_pure_phase_is_a_shift(cross_setup):
    """q = exp(i k_par a) must give xi(r_par + a): sign and normalisation of compute_odd."""
    shift = 4.0
    k_par, pk_sym = cross_setup["k_par"], cross_setup["pk_sym"]
    model = _xi(cross_setup, pk_sym * np.cos(k_par * shift), R_PAR)
    model += _xi(cross_setup, pk_sym * np.sin(k_par * shift), R_PAR, odd=True)
    direct = _xi(cross_setup, pk_sym, R_PAR + shift)
    assert np.max(np.abs(model - direct)) < 1e-3 * np.max(np.abs(direct))


def test_skewness_is_the_third_derivative(cross_setup):
    kappa3 = 50.0
    k_par, pk_sym = cross_setup["k_par"], cross_setup["pk_sym"]
    even_ratio, odd_ratio = _kernel_ratios("skewness", {"zerr_kappa3_QSO": kappa3}, k_par)
    assert even_ratio is None
    model = _xi(cross_setup, pk_sym * odd_ratio, R_PAR, odd=True)

    # (kappa3 / 6) d^3 xi / d r_par^3 by central finite differences
    step = 0.5
    values = [_xi(cross_setup, pk_sym, R_PAR + offset * step) for offset in (-2, -1, 1, 2)]
    third_derivative = (values[3] - 2 * values[2] + 2 * values[1] - values[0]) / (2 * step**3)
    direct = kappa3 / 6 * third_derivative
    assert np.max(np.abs(model - direct)) < 0.02 * np.max(np.abs(direct))


@pytest.mark.parametrize(
    "form, parameter, value, third_cumulant",
    [
        # l+^3 - l-^3 = sigma^3 [(1 + eps)^3 - (1 - eps)^3] (half the asymmetric-Laplace value),
        # and 2 tau^3
        ("asymmetric-lorentz", "zerr_asymmetry_QSO", 0.5, SIGMA**3 * (1.5**3 - 0.5**3)),
        ("exponential-tail", "zerr_tail_QSO", 4.0, 2 * 4.0**3),
        ("exponential-tail", "zerr_tail_QSO", -4.0, -2 * 4.0**3),
    ],
)
def test_kernels_match_direct_convolution(cross_setup, form, parameter, value, third_cumulant):
    """The model equals xi_true convolved with the PDF of Delta (from an inverse FFT of q)."""
    params = {"sigma_velo_disp_lorentz_QSO": SIGMA, parameter: value}
    k_par, pk_sym = cross_setup["k_par"], cross_setup["pk_sym"]
    even_ratio, odd_ratio = _kernel_ratios(form, params, k_par)
    model = _xi(cross_setup, pk_sym * even_ratio, R_PAR)
    model += _xi(cross_setup, pk_sym * odd_ratio, R_PAR, odd=True)

    # PDF p(Delta) = (1 / 2 pi) int dk q(k) exp(-i k Delta) on a fine symmetric k grid
    num_k, delta_k = 2**19, 0.004
    k_fine = (np.arange(num_k) - num_k // 2) * delta_k
    even_fine, odd_fine = _kernel_ratios(form, params, np.abs(k_fine))
    q_fine = (even_fine + 1j * np.sign(k_fine) * odd_fine) / np.sqrt(1 + (k_fine * SIGMA) ** 2)
    pdf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(q_fine))).real * delta_k / (2 * np.pi)
    delta_grid = (np.arange(num_k) - num_k // 2) * (2 * np.pi / (num_k * delta_k))
    keep = np.abs(delta_grid) < 80
    delta_grid, pdf = delta_grid[keep], pdf[keep]
    step = delta_grid[1] - delta_grid[0]

    # Normalised, mean-free, and with the analytic third cumulant
    assert abs(np.sum(pdf) * step - 1) < 1e-3
    assert abs(np.sum(delta_grid * pdf) * step) < 1e-2
    assert np.isclose(np.sum(delta_grid**3 * pdf) * step, third_cumulant, rtol=0.02)

    r_par_fine = np.arange(-200.0, 200.0, 0.05)
    xi_true = interpolate.interp1d(
        r_par_fine, _xi(cross_setup, cross_setup["pk_true"], r_par_fine), kind="cubic"
    )
    direct = np.array([np.sum(pdf * xi_true(r_par + delta_grid)) * step for r_par in R_PAR])
    odd_part = _xi(cross_setup, pk_sym * odd_ratio, R_PAR, odd=True)
    assert np.max(np.abs(model - direct)) < 0.05 * np.max(np.abs(odd_part))


def test_zero_asymmetry_reduces_to_symmetric_kernel(cross_setup):
    k_par = cross_setup["k_par"]
    for form, parameter in (
        ("asymmetric-lorentz", "zerr_asymmetry_QSO"),
        ("exponential-tail", "zerr_tail_QSO"),
    ):
        params = {"sigma_velo_disp_lorentz_QSO": SIGMA, parameter: 0.0}
        even_ratio, odd_ratio = _kernel_ratios(form, params, k_par)
        assert np.allclose(even_ratio, 1.0) and np.allclose(odd_ratio, 0.0)


def _write_cross_configs(tmp_path, form, keep_metals=False, form_in_metals=False):
    """One-dataset copy of the full Lya x QSO test configs, by default without metals.

    Parameters
    ----------
    tmp_path : pathlib.Path
    form : str or None
        Value of the 'redshift-error asymmetry' option in [model] (None: option absent)
    keep_metals : bool, optional
        Keep the [metals] section (metal x QSO terms), by default False
    form_in_metals : bool, optional
        Also write the option to [metals], which the metal x QSO correlations read, by default False

    Returns
    -------
    pathlib.Path
        Path of the main config
    """
    corr_config = configparser.ConfigParser()
    corr_config.optionxform = str
    corr_config.read(find_file("full_configs/lyalya_qso.ini"))
    if not keep_metals:
        corr_config.remove_section("metals")
    if form is not None:
        corr_config["model"]["redshift-error asymmetry"] = form
        if form_in_metals:
            corr_config["metals"]["redshift-error asymmetry"] = form
    tag = f"{form}_metals{int(keep_metals)}{int(form_in_metals)}"
    corr_path = tmp_path / f"lyalya_qso_{tag}.ini"
    with open(corr_path, "w") as corr_file:
        corr_config.write(corr_file)

    main_config = configparser.ConfigParser()
    main_config.optionxform = str
    main_config.read(find_file("full_configs/main.ini"))
    main_config["data sets"]["ini files"] = str(corr_path)
    main_config["output"]["filename"] = str(tmp_path / "output")
    if main_config.has_section("Polychord"):
        main_config.remove_section("Polychord")
    main_path = tmp_path / f"main_{tag}.ini"
    with open(main_path, "w") as main_file:
        main_config.write(main_file)
    return main_path


@pytest.mark.parametrize(
    "form, parameter, value",
    [
        ("skewness", "zerr_kappa3_QSO", 30.0),
        ("asymmetric-lorentz", "zerr_asymmetry_QSO", 0.3),
        ("exponential-tail", "zerr_tail_QSO", 2.0),
    ],
)
def test_zerr_asymmetry_end_to_end(tmp_path, form, parameter, value):
    vega_without = VegaInterface(str(_write_cross_configs(tmp_path, None)))
    model_without = vega_without.compute_model(vega_without.params)["lyalya_qso"]

    vega_with = VegaInterface(str(_write_cross_configs(tmp_path, form)))
    params = dict(vega_with.params)
    model_zero = vega_with.compute_model({**params, parameter: 0.0})["lyalya_qso"]
    model_on = vega_with.compute_model({**params, parameter: value}, run_init=False)["lyalya_qso"]

    # Zero asymmetry reproduces the model without the option; a non-zero one changes it
    assert np.allclose(model_zero, model_without, rtol=1e-6, atol=1e-12)
    assert np.max(np.abs(model_on - model_zero)) > 1e-3 * np.max(np.abs(model_zero))


def test_zerr_asymmetry_reaches_metal_correlations(tmp_path):
    """With the option in [metals], the metal x QSO terms get the asymmetric kernel too."""
    parameter, value = "zerr_tail_QSO", 2.0
    form = "exponential-tail"
    model_only = VegaInterface(str(_write_cross_configs(tmp_path, form, keep_metals=True)))
    with_metals = VegaInterface(
        str(_write_cross_configs(tmp_path, form, keep_metals=True, form_in_metals=True))
    )
    params = {**dict(with_metals.params), parameter: value}
    xi_model_only = model_only.compute_model(params)["lyalya_qso"]
    xi_with_metals = with_metals.compute_model(params)["lyalya_qso"]

    # The two differ only through the metal x QSO terms, which are a small part of the model
    difference = np.max(np.abs(xi_with_metals - xi_model_only))
    assert difference > 1e-6 * np.max(np.abs(xi_model_only))
    assert difference < 0.2 * np.max(np.abs(xi_model_only))

    # At zero asymmetry the metal terms reduce to the symmetric kernel
    params_zero = {**params, parameter: 0.0}
    assert np.allclose(
        model_only.compute_model(params_zero)["lyalya_qso"],
        with_metals.compute_model(params_zero)["lyalya_qso"],
        rtol=1e-6,
        atol=1e-12,
    )
