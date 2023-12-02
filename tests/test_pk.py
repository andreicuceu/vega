import pytest
import numpy as np
import configparser
from astropy.io import fits

from vega.power_spectrum import PowerSpectrum
from vega import utils


def kaiser():
    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'LYA')
    assert bias1 == bias2
    assert beta1 == beta2
    assert bias1 == -0.12
    assert beta1 == 1.6

    params = {'bias_eta_LYA': -0.2, 'beta_LYA': 1.6, 'growth_rate': 0.97}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'LYA')
    assert bias1 == pytest.approx(-0.2 * 0.97 / 1.6)
    assert beta1 == 1.6

    params = {'bias_eta_LYA': -0.2, 'bias_LYA': -0.12, 'growth_rate': 0.97}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'LYA')
    assert bias1 == -0.12
    assert beta1 == pytest.approx(-0.2 * 0.97 / -0.12)

    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6,
              'bias_QSO': 3.7, 'beta_QSO': 0.25}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'QSO')
    assert bias1 == -0.12
    assert beta1 == 1.6
    assert bias2 == 3.7
    assert beta2 == 0.25

    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6,
              'bias_eta_QSO': 1, 'beta_QSO': 0.25,
              'growth_rate': 0.97}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'QSO')
    assert bias1 == -0.12
    assert beta1 == 1.6
    assert bias2 == pytest.approx(0.97 / 0.25)
    assert beta2 == 0.25

    params = {'bias_eta_LYA': -0.2, 'beta_LYA': 1.6,
              'bias_eta_QSO': 1, 'bias_QSO': 3.7,
              'growth_rate': 0.97}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'QSO')
    assert bias1 == pytest.approx(-0.2 * 0.97 / 1.6)
    assert beta1 == 1.6
    assert bias2 == 3.7
    assert beta2 == pytest.approx(0.97 / 3.7)


def compute_bias_beta_uv(model_config, fiducial, tracer1, tracer2, dataset_name):
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    params = {'bias_gamma': 0.1125, 'bias_prim': -0.66, 'lambda_uv': 300}
    bias_uv, beat_uv = pk.compute_bias_beta_uv(-0.12, 1.6, params)
    assert np.sum(bias_uv) == pytest.approx(-35.268497)
    assert np.sum(beat_uv) == pytest.approx(1138.77689)


def compute_bias_beta_hcd(model_config, fiducial, tracer1, tracer2, dataset_name):
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    assert pk.hcd_model == 'Rogers'
    params = {'bias_hcd': -0.05, 'beta_hcd': 0.5, 'L0_hcd': 10}
    bias_eff, beta_eff = pk.compute_bias_beta_hcd(-0.12, 1.6, params)
    assert np.sum(bias_eff) == pytest.approx(-116031.686)
    assert np.sum(beta_eff) == pytest.approx(1179867.64849)

    F_hcd = pk._hcd_Rogers2018(10, pk.k_par_grid)
    assert np.allclose(F_hcd, pk._F_hcd)
    assert pk._L0_hcd_cache == 10

    model_config['model-hcd'] = 'fvoigt'
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    assert pk.hcd_model == 'fvoigt'
    pk._F_hcd = None
    bias_eff, beta_eff = pk.compute_bias_beta_hcd(-0.12, 1.6, params)
    assert np.sum(bias_eff) == pytest.approx(-121782.768388)
    assert np.sum(beta_eff) == pytest.approx(1142662.6535)

    F_hcd = pk._hcd_fvoigt(1)
    assert np.allclose(F_hcd, pk._F_hcd)

    model_config['model-hcd'] = 'sinc'
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    assert pk.hcd_model == 'sinc'
    pk._F_hcd = None
    params['L0_sinc'] = 10
    bias_eff, beta_eff = pk.compute_bias_beta_hcd(-0.12, 1.6, params)
    assert np.sum(bias_eff) == pytest.approx(-118530.3944)
    assert np.sum(beta_eff) == pytest.approx(1166657.39777)

    F_hcd = pk._hcd_sinc(10)
    assert np.allclose(F_hcd, pk._F_hcd)
    pk.hcd_model = 'Rogers'
    pk._F_hcd = None


def compute_peak_nl(model_config, fiducial, tracer1, tracer2, dataset_name):
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    pk._peak_nl_cache = None
    params = {'sigmaNL_par': 6.36984, 'sigmaNL_per': 3.24}
    peak_nl = pk.compute_peak_nl(params)
    assert np.sum(peak_nl) == pytest.approx(390698.51738)
    assert np.allclose(peak_nl, pk._peak_nl_cache)

    pk._peak_nl_cache = None
    params = {'sigmaNL_par': 6.36984, 'growth_rate': 0.97}
    peak_nl = pk.compute_peak_nl(params)
    assert np.sum(peak_nl) == pytest.approx(390747.02382)

    pk._peak_nl_cache = None
    params = {'sigmaNL_per': 3.24, 'growth_rate': 0.97}
    peak_nl = pk.compute_peak_nl(params)
    assert np.sum(peak_nl) == pytest.approx(390645.39796)


def compute_dnl(model_config, fiducial, tracer1, tracer2, dataset_name):
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    pk._arinyo_pars = None
    params = {'dnl_arinyo_q1': 0.8558, 'dnl_arinyo_kv': 1.11454, 'dnl_arinyo_av': 0.5378,
              'dnl_arinyo_bv': 1.607, 'dnl_arinyo_kp': 19.47}
    dnl = pk.compute_dnl_arinyo(params)
    assert np.sum(dnl) == pytest.approx(680327.61617)
    assert np.allclose(dnl, pk._arinyo_dnl_cache)

    dnl = pk.compute_dnl_mcdonald()
    assert np.sum(dnl) == pytest.approx(632262.53194)


def compute_fullshape_smoothing(model_config, fiducial, tracer1, tracer2, dataset_name):
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    params = {'par_sigma_smooth': 2, 'per_sigma_smooth': 2.5}
    fs_smoothing = pk.compute_fullshape_gauss_smoothing(params)
    assert np.sum(fs_smoothing) == pytest.approx(404166.27948)

    params = {'par_sigma_smooth': 2, 'per_sigma_smooth': 2.5,
              'par_exp_smooth': 2, 'per_exp_smooth': 2.5}
    fs_smoothing = pk.compute_fullshape_exp_smoothing(params)
    assert np.sum(fs_smoothing) == pytest.approx(333204.95791)


def compute_velocity_dispersion(pk):
    params = {'sigma_velo_disp_gauss_QSO': 6.8, 'sigma_velo_disp_lorentz_QSO': 7.2}
    pk_velo_disp = pk.compute_velocity_dispersion_gauss(params)
    assert np.sum(pk_velo_disp) == pytest.approx(435379.6457)

    pk_velo_disp = pk.compute_velocity_dispersion_lorentz(params)
    assert np.sum(pk_velo_disp) == pytest.approx(446899.3964)


def auto_pk(fiducial):
    tracer1 = {'name': 'LYA', 'type': 'continuous'}
    tracer2 = {'name': 'LYA', 'type': 'continuous'}
    dataset_name = 'lyaxlya'

    config = configparser.ConfigParser()
    config['model'] = {}
    model_config = config['model']
    model_config['bin_size_rp'] = '4'
    model_config['bin_size_rt'] = '4'
    model_config['model binning'] = 'False'

    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    assert not pk.use_Gk
    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6, 'peak': False}
    bias1, beta1, bias2, beta2 = utils.bias_beta(params, 'LYA', 'LYA')

    pk_kaiser = pk.compute_kaiser(bias1, beta1, bias2, beta2)
    assert np.shape(pk_kaiser) == (1000, 1)
    assert np.sum(pk_kaiser) == pytest.approx(37.13279)
    pk_computed = pk.compute(fiducial['pk_smooth'], params)
    assert np.shape(pk_computed) == (1000, 814)
    assert pk.use_Gk is False
    assert np.allclose(pk_computed, fiducial['pk_smooth'] * pk_kaiser)

    Gk = pk.compute_Gk({'par binsize lyaxlya': 2, 'per binsize lyaxlya': 3})
    assert np.sum(Gk) == pytest.approx(470301.136422)
    Gk = pk.compute_Gk(params)
    assert np.sum(Gk) == pytest.approx(450783.949889)

    pk.pk_Gk = Gk
    pk.use_Gk = True
    pk_computed = pk.compute(fiducial['pk_smooth'], params)
    assert np.allclose(pk_computed, fiducial['pk_smooth'] * pk_kaiser * Gk)

    model_config['num_bins_muk'] = '500'
    model_config['model binning'] = 'True'
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    assert np.shape(pk.muk_grid) == (500, 1)
    assert pk.use_Gk

    model_config['num_bins_muk'] = '1000'
    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    pk_metals = pk.compute(fiducial['pk_smooth'], params, fast_metals=True)
    assert np.mean(pk_metals) == pytest.approx(1228.9847366)

    model_config['model-hcd'] = 'Rogers'
    model_config['add uv'] = 'True'
    model_config['fvoigt_model'] = 'exp'
    model_config['small scale nl'] = 'arinyo'  # mcdonald
    model_config['fullshape smoothing'] = 'gauss'  # exp

    compute_bias_beta_uv(model_config, fiducial, tracer1, tracer2, dataset_name)
    compute_bias_beta_hcd(model_config, fiducial, tracer1, tracer2, dataset_name)
    compute_peak_nl(model_config, fiducial, tracer1, tracer2, dataset_name)
    compute_dnl(model_config, fiducial, tracer1, tracer2, dataset_name)
    compute_fullshape_smoothing(model_config, fiducial, tracer1, tracer2, dataset_name)

    model_config['model-hcd'] = 'Rogers'
    model_config['add uv'] = 'True'
    model_config['fvoigt_model'] = 'exp'
    model_config['small scale nl'] = 'arinyo'  # mcdonald
    model_config['fullshape smoothing'] = 'gauss'  # exp

    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6, 'bias_gamma': 0.1125, 'bias_prim': -0.66,
              'lambda_uv': 300, 'bias_hcd': -0.05, 'beta_hcd': 0.5, 'L0_hcd': 10,
              'sigmaNL_par': 6.36984, 'sigmaNL_per': 3.24, 'par_sigma_smooth': 2,
              'per_sigma_smooth': 2.5, 'dnl_arinyo_q1': 0.8558, 'dnl_arinyo_kv': 1.11454,
              'dnl_arinyo_av': 0.5378, 'dnl_arinyo_bv': 1.607, 'dnl_arinyo_kp': 19.47}

    params['peak'] = True
    pk_model = pk.compute(fiducial['pk_full']-fiducial['pk_smooth'], params)
    assert np.mean(pk_model) == pytest.approx(2.8794436016)

    params['peak'] = False
    pk_model = pk.compute(fiducial['pk_smooth'], params)
    assert np.mean(pk_model) == pytest.approx(19.67878957)


def cross_pk(fiducial):
    tracer1 = {'name': 'LYA', 'type': 'continuous'}
    tracer2 = {'name': 'QSO', 'type': 'discrete'}
    dataset_name = 'lyaxqso'

    config = configparser.ConfigParser()
    config['model'] = {}
    model_config = config['model']
    model_config['bin_size_rp'] = '4'
    model_config['bin_size_rt'] = '4'
    model_config['num_bins_muk'] = '1000'
    model_config['model-hcd'] = 'Rogers'
    model_config['add uv'] = 'True'
    model_config['fvoigt_model'] = 'exp'
    model_config['fullshape smoothing'] = 'gauss'  # exp
    model_config['velocity dispersion'] = 'lorentz'  # gauss

    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    compute_velocity_dispersion(pk)

    pk = PowerSpectrum(model_config, fiducial, tracer1, tracer2, dataset_name)
    params = {'bias_LYA': -0.12, 'beta_LYA': 1.6, 'bias_QSO': 3.7, 'beta_QSO': 0.26,
              'bias_gamma': 0.1125, 'bias_prim': -0.66, 'lambda_uv': 300, 'bias_hcd': -0.05,
              'beta_hcd': 0.5, 'L0_hcd': 10, 'sigmaNL_par': 6.36984, 'sigmaNL_per': 3.24,
              'par_sigma_smooth': 2, 'per_sigma_smooth': 2.5, 'sigma_velo_disp_lorentz_QSO': 7.2}

    params['peak'] = True
    pk_model = pk.compute(fiducial['pk_full']-fiducial['pk_smooth'], params)
    assert np.mean(pk_model) == pytest.approx(-2.9406788865)

    params['peak'] = False
    pk_model = pk.compute(fiducial['pk_smooth'], params)
    assert np.mean(pk_model) == pytest.approx(-401.0937936)


def test_pk():
    fiducial = {}
    fiducial['z_eff'] = 2.25
    template = utils.find_file('PlanckDR16/PlanckDR16.fits')
    with fits.open(template) as hdul:
        fiducial['k'] = hdul[1].data['K']
        fiducial['pk_full'] = hdul[1].data['PK']
        fiducial['pk_smooth'] = hdul[1].data['PKSB']
        fiducial['z_fiducial'] = hdul[1].header['ZREF']

    kaiser()
    auto_pk(fiducial)
    cross_pk(fiducial)
