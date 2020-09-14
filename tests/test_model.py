import pytest
import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix

# from lyafit.new_model import Model
# import lyafit.parser as parser

@pytest.mark.skip
def test_model():
    filename = "D:\\work\\run\\DR16\\chi2.ini"
    dic_init = parser.parse_chi2(filename)

    # names = dic_init['data sets']['data'][0].par_names
    pars = dic_init['data sets']['data'][0].pars_init
    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pksb_lin = dic_init['fiducial']['pksb']
    data = dic_init['data sets']['data'][0]

    path_dr16_cf = "D:\\work\\data\\DR16\\cf_z_0_10-exp.fits"
    with fits.open(path_dr16_cf) as h:
        rp = h[1].data['RP'][:]
        rt = h[1].data['RT'][:]
        z = h[1].data['Z'][:]
        dm = csr_matrix(h[1].data['DM'][:])

    # rp_scaled_bins = np.linspace(0, 200, 51)
    # rt_scaled_bins = np.linspace(0, 200, 51)
    # rp_centers = (rp_scaled_bins[1:] + rp_scaled_bins[:-1]) / 2
    # rt_centers = (rt_scaled_bins[1:] + rt_scaled_bins[:-1]) / 2

    # rt_mesh, rp_mesh = np.meshgrid(rt_centers, rp_centers)
    # rt = rt_mesh.flatten()
    # rp = rp_mesh.flatten()
    r = np.sqrt(rp**2 + rt**2)
    mu = rp / r
    z = data.dic_init['model']['zeff']
    cf_model = Model(data.dic_init, r, mu, z, dm)
    xi_model = cf_model.compute(pars, pk_lin, pksb_lin)

    pars['SB'] = False
    xi_peak = data.xi_model(k, pk_lin-pksb_lin, pars)
    pars['SB'] = True
    pars['sigmaNL_par'] = 0.
    pars['sigmaNL_per'] = 0.
    xi_sb = data.xi_model(k, pksb_lin, pars)
    xi_model_data = pars['bao_amp']*xi_peak + xi_sb
    # print(xi_model)

    print('New:', np.sum(xi_model))
    print('Old:', np.sum(xi_model_data))
    # assert np.allclose(xi_model, xi_model_data), "The model class \
        # produces different results from the old data class"


# if __name__ == "__main__":
    # test_model()
