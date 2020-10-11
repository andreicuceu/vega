import pytest
import numpy as np

from vega.power_spectrum import PowerSpectrum
from vega import VegaInterface


def test_pk():
    vega = VegaInterface('configs/main.ini')

    # Test the Auto
    auto_corr = vega.corr_items['lyaxlya']
    auto_config = auto_corr.config['model']
    auto_config['num_bins_muk'] = str(2)

    temp_fiducial = vega.fiducial.copy()
    temp_fiducial['k'] = np.array([1e-2, 1e2])
    pk_auto = PowerSpectrum(auto_corr.config['model'], temp_fiducial,
                            auto_corr.tracer1, auto_corr.tracer2, 'lyaxlya')

    temp_params = vega.params.copy()

    kaiser_result = pk_auto.compute_kaiser(-0.1, 1., -0.1, 1.)
    kaiser_expected = np.array([[0.01128906], [0.02441406]])
    assert np.allclose(kaiser_result, kaiser_expected)

    bias_res, beta_res = pk_auto.compute_bias_beta_uv(-0.1, 1., temp_params)
    bias_exp = np.array([-0.03541289, -0.09999411])
    beta_exp = np.array([2.82383085, 1.00005891])
    assert np.allclose(bias_res, bias_exp)
    assert np.allclose(beta_res, beta_exp)

    # auto_config['num_bins_muk'] = str(1000)
    # pk_auto = PowerSpectrum(auto_corr.config['model'], vega.fiducial,
                            # auto_corr.tracer1, auto_corr.tracer2, 'lyaxlya')

    # temp_params['peak'] = False
    # print(pk_auto.compute(vega.fiducial['pk_full'], temp_params))

# if __name__ == '__main__':
    # test_pk()
