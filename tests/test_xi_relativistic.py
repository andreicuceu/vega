"""Tests of the relativistic (Doppler) and asymmetry terms of the cross-correlation model."""

import configparser

import numpy as np
import pytest
from astropy.io import fits
from scipy.special import legendre, spherical_jn

from vega import VegaInterface
from vega.pktoxi import PktoXi
from vega.utils import find_file

R_GRID = np.array([10.0, 20.0, 50.0, 100.0, 150.0, 190.0])


@pytest.fixture(scope="module")
def damped_template():
    """PlanckDR16 linear P(k), with a Gaussian damping so that the direct integrals converge.

    Returns
    -------
    tuple of 1D Array
        (k [h/Mpc], P(k) exp(-k^2))
    """
    with fits.open(find_file("PlanckDR16/PlanckDR16.fits")) as hdul:
        k_grid = np.array(hdul[1].data["K"])
        pk = np.array(hdul[1].data["PK"]) * np.exp(-(k_grid**2))
    return k_grid, pk


@pytest.fixture(scope="module")
def pktoxi(damped_template):
    """PktoXi with the default model options on the template k grid."""
    config = configparser.ConfigParser()
    config.read_dict({"model": {}})
    k_grid, _ = damped_template
    return PktoXi(k_grid, np.linspace(0, 1, 10)[:, None], config["model"])


def direct_transform(k_grid, pk, r_grid, ell, n):
    """Brute-force 1/(2 pi^2) int dk k^(2 + n) P(k) j_ell(kr) on a fine log-k grid.

    Parameters
    ----------
    k_grid, pk : 1D Array
        Power spectrum
    r_grid : 1D Array
        Separations [Mpc/h]
    ell, n : int
        Order of the spherical Bessel function and extra power of k

    Returns
    -------
    1D Array
    """
    log_k_fine = np.linspace(np.log(1e-4), np.log(6.0), 200000)
    k_fine = np.exp(log_k_fine)
    pk_fine = np.interp(log_k_fine, np.log(k_grid), pk)
    bessel = spherical_jn(ell, k_fine[None, :] * r_grid[:, None])
    integrand = k_fine[None, :] ** (3 + n) * pk_fine[None, :] * bessel
    return np.trapezoid(integrand, log_k_fine, axis=1) / (2 * np.pi**2)


@pytest.mark.parametrize("ell, n", [(1, -1), (3, -1), (0, 0), (2, 0)])
def test_isotropic_transform_matches_direct_integral(pktoxi, damped_template, ell, n):
    k_grid, pk = damped_template
    transformed = pktoxi.transform_isotropic(R_GRID, pk, ell=ell, n=n)
    direct = direct_transform(k_grid, pk, R_GRID, ell, n)
    assert np.max(np.abs(transformed - direct)) < 1e-3 * np.max(np.abs(direct))


def test_relativistic_term_is_the_doppler_dipole_and_octupole(pktoxi, damped_template):
    k_grid, pk = damped_template
    mu_grid = np.linspace(-0.9, 0.9, len(R_GRID))
    nu_1 = direct_transform(k_grid, pk, R_GRID, 1, -1)
    nu_3 = direct_transform(k_grid, pk, R_GRID, 3, -1)

    dipole = pktoxi.pk_to_xi_relativistic(R_GRID, mu_grid, pk, {"Arel1": 1.0, "Arel3": 0.0})
    octupole = pktoxi.pk_to_xi_relativistic(R_GRID, mu_grid, pk, {"Arel1": 0.0, "Arel3": 1.0})
    assert np.allclose(dipole, nu_1 * mu_grid, rtol=1e-3, atol=1e-3 * np.max(np.abs(nu_1)))
    octupole_expected = nu_3 * legendre(3)(mu_grid)
    assert np.allclose(octupole, octupole_expected, rtol=1e-3, atol=1e-3 * np.max(np.abs(nu_3)))

    # Linear in the amplitudes, zero without them, odd in mu
    combined = pktoxi.pk_to_xi_relativistic(R_GRID, mu_grid, pk, {"Arel1": -2.0, "Arel3": 0.5})
    assert np.allclose(combined, -2.0 * dipole + 0.5 * octupole)
    zero = pktoxi.pk_to_xi_relativistic(R_GRID, mu_grid, pk, {"Arel1": 0.0, "Arel3": 0.0})
    assert np.all(zero == 0)
    mirrored = pktoxi.pk_to_xi_relativistic(R_GRID, -mu_grid, pk, {"Arel1": -2.0, "Arel3": 0.5})
    assert np.allclose(mirrored, -combined)


def test_asymmetry_keeps_previous_normalisation(pktoxi, damped_template):
    """The mcfit asymmetry term agrees with the previous Pk2Mp version, up to its 2-3% error."""
    _, pk = damped_template
    r_grid = np.array([10.0, 20.0, 50.0])
    mu_grid = np.array([0.3, -0.5, 0.8])
    params = {"Aasy0": 1.0, "Aasy2": 0.7, "Aasy3": -0.4}
    new = pktoxi.pk_to_xi_asymmetry(r_grid, mu_grid, pk, params)

    old_kernels = PktoXi.Pk2Mp(r_grid, pktoxi.k_grid, pk, [0, 2], None, None, tform="asy")
    old = (params["Aasy0"] * old_kernels[0] - params["Aasy2"] * old_kernels[1]) * r_grid * mu_grid
    old += params["Aasy3"] * old_kernels[1] * r_grid * legendre(3)(mu_grid)
    assert np.allclose(new, old, rtol=0.04)


def _write_cross_configs(tmp_path, relativistic):
    """One-dataset copy of the full Lya x QSO test configs, without metals.

    Parameters
    ----------
    tmp_path : pathlib.Path
    relativistic : bool
        Value of the 'relativistic correction' model option

    Returns
    -------
    pathlib.Path
        Path of the main config
    """
    corr_config = configparser.ConfigParser()
    corr_config.optionxform = str
    corr_config.read(find_file("full_configs/lyalya_qso.ini"))
    corr_config.remove_section("metals")
    corr_config["model"]["relativistic correction"] = str(relativistic)
    corr_path = tmp_path / f"lyalya_qso_rel{relativistic}.ini"
    with open(corr_path, "w") as corr_file:
        corr_config.write(corr_file)

    main_config = configparser.ConfigParser()
    main_config.optionxform = str
    main_config.read(find_file("full_configs/main.ini"))
    main_config["data sets"]["ini files"] = str(corr_path)
    main_config["output"]["filename"] = str(tmp_path / "output")
    for section in ("Polychord",):
        if main_config.has_section(section):
            main_config.remove_section(section)
    main_path = tmp_path / f"main_rel{relativistic}.ini"
    with open(main_path, "w") as main_file:
        main_config.write(main_file)
    return main_path


def test_relativistic_term_end_to_end(tmp_path):
    vega_without = VegaInterface(str(_write_cross_configs(tmp_path, False)))
    model_without = vega_without.compute_model(vega_without.params)["lyalya_qso"]

    vega_with = VegaInterface(str(_write_cross_configs(tmp_path, True)))
    params = dict(vega_with.params)
    model_zero = vega_with.compute_model({**params, "Arel1": 0.0, "Arel3": 0.0})["lyalya_qso"]
    model_one = vega_with.compute_model({**params, "Arel1": 1e-3, "Arel3": 0.0}, run_init=False)
    model_one = model_one["lyalya_qso"]
    model_two = vega_with.compute_model({**params, "Arel1": 2e-3, "Arel3": 0.0}, run_init=False)
    model_two = model_two["lyalya_qso"]

    # Zero amplitudes reproduce the model without the term; the term is linear and non-zero
    assert np.allclose(model_zero, model_without, rtol=1e-10, atol=0)
    difference = model_one - model_zero
    assert np.max(np.abs(difference)) > 0
    assert np.allclose(model_two - model_zero, 2 * difference, rtol=1e-8, atol=1e-14)
