import pytest
import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix

from lyafit.model import model
import lyafit.parser as parser

@pytest.mark.skip
def test_model():
    filename = "D:\\work\\run\\DR16\\chi2.ini"
    dic_init = parser.parse_chi2(filename)

    pars = dic_init['data sets']['data'][0].pars_init
    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pars['SB'] = False
    # pksb_lin = dic_init['fiducial']['pksb']

    path_dr16_cf = "D:\\work\\data\\DR16\\cf_z_0_10-exp.fits"
    with fits.open(path_dr16_cf) as h:
        rp = h[1].data['RP'][:]
        rt = h[1].data['RT'][:]
        z = h[1].data['Z'][:]
        dm = csr_matrix(h[1].data['DM'][:])

    r = np.sqrt(rp**2 + rt**2)
    mu = rp / r

    cf_model = model(dic_init['data sets']['data'][0].dic_init, r, mu, z, dm)
    xi_model = cf_model.xi_model(k, pk_lin, pars)

    xi_model_data = dic_init['data sets']['data'][0].xi_model(k, pk_lin, pars)
    # print(xi_model)

    assert np.allclose(xi_model, xi_model_data), "The model class \
        produces different results from the old data class"


# if __name__ == "__main__":
#     model_test()
