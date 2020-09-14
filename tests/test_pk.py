import pytest
import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix

# from lyafit.model import model
# import lyafit.parser as parser
# import lyafit.pk
# from lyafit.new_pk import PowerSpectrum

@pytest.mark.skip
def test_model():
    filename = "D:\\work\\run\\DR16\\chi2.ini"
    dic_init = parser.parse_chi2(filename)

    # names = dic_init['data sets']['data'][0].par_names
    pars = dic_init['data sets']['data'][0].pars_init
    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pars['SB'] = False
    pksb_lin = dic_init['fiducial']['pksb']

    path_dr16_cf = "D:\\work\\data\\DR16\\cf_z_0_10-exp.fits"
    with fits.open(path_dr16_cf) as h:
        rp = h[1].data['RP'][:]
        rt = h[1].data['RT'][:]
        z = h[1].data['Z'][:]
        dm = csr_matrix(h[1].data['DM'][:])

    r = np.sqrt(rp**2 + rt**2)
    mu = rp / r

    cf_model = model(dic_init['data sets']['data'][0].dic_init, r, mu, z, dm)

    pars['name'] = cf_model.name
    # print(pars)
    pk_full = cf_model.pk(k, pk_lin, tracer1=cf_model.tracer1, tracer2=cf_model.tracer2, **pars)

    # print(np.shape(pk_full))
    # print('PK Median: ', np.median(pk_full))
    # print('PK Mean: ', np.mean(pk_full))

    config = dic_init['data sets']['data'][0].dic_init['model']
    pk_fid = dic_init['data sets']['data'][0].dic_init['model']['pk']
    pk_fid_new = pk_fid * ((1+config['zref'])/(1.+config['zeff']))**2

    new_pk = PowerSpectrum(config, cf_model.tracer1, cf_model.tracer2, cf_model.name, pk_fid_new)
    pars['peak'] = True
    pk_full_new = new_pk.compute(k, pk_lin, pars)
    # print(np.shape(pk_full_new))
    print('Diff:', np.sum(pk_full - pk_full_new))
    print('Sum old:', np.sum(pk_full))
    print('Sum new:', np.sum(pk_full_new))
    # raise ValueError('here')
    # print(pk_full_new)
    # xi_model = cf_model.xi_model(k, pk_lin, pars)
    # xi_model_data = dic_init['data sets']['data'][0].xi_model(k, pk_lin, pars)
    # print(xi_model)

    # assert np.allclose(xi_model, xi_model_data), "The model class \
        # produces different results from the old data class"


# if __name__ == "__main__":
    # test_model()
