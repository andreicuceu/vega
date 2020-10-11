import pytest
import numpy as np
from astropy.io import fits

from vega import VegaInterface
from vega.utils import find_file


def test_vega():
    hdul = fits.open(find_file('data/picca_bench_data.fits'))
    names = ['test_' + str(i) for i in range(8)]

    vega_auto = VegaInterface(
        'examples/picca_benchmarks/configs/vega/main.ini')

    vega_auto.fiducial['Omega_de'] = None
    xi_vega_auto = vega_auto.compute_model(run_init=True)

    vega_cross = VegaInterface(
        'examples/picca_benchmarks/configs/vega/main_cross.ini')
    vega_cross.fiducial['Omega_de'] = None
    xi_vega_cross = vega_cross.compute_model(run_init=True)

    for name in names:
        xi_picca_auto = np.array(hdul[1].data['auto_' + name])
        xi_picca_cross = np.array(hdul[2].data['cross_' + name])

        assert np.allclose(xi_vega_auto[name], xi_picca_auto)
        assert np.allclose(xi_vega_cross[name], xi_picca_cross)

    hdul.close()
