import pytest
import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix

from lyafit.model import model
import lyafit.parser as parser
# import lyafit.pk
from lyafit.new_pk import PowerSpectrum
from lyafit.new_xi import CorrelationFunction

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
    data = dic_init['data sets']['data'][0]

    path_dr16_cf = "D:\\work\\data\\DR16\\cf_z_0_10-exp.fits"
    with fits.open(path_dr16_cf) as h:
        rp = h[1].data['RP'][:]
        rt = h[1].data['RT'][:]
        z = h[1].data['Z'][:]
        dm = csr_matrix(h[1].data['DM'][:])

    r = np.sqrt(rp**2 + rt**2)
    mu = rp / r

    cf_model = model(data.dic_init, r, mu, z, dm)
    print(data.dic_init['model'].keys())
    pars['name'] = cf_model.name

    config = dic_init['data sets']['data'][0].dic_init['model']
    pk_fid = dic_init['data sets']['data'][0].dic_init['model']['pk']
    pk_fid_new = pk_fid * ((1+config['zref'])/(1.+config['zeff']))**2

    pk_obj = PowerSpectrum(config, cf_model.tracer1, cf_model.tracer2, cf_model.name, pk_fid_new)
    pars['peak'] = False
    pars['SB'] = True
    pars['sigmaNL_per'] = 0.
    pars['sigmaNL_par'] = 0.
    pk = pk_obj.compute(k, pksb_lin, pars)
    print('pk:', np.sum(pk))
    xi_old = cf_model.xi_model(k, pksb_lin, pars)

    config['ell_max'] = cf_model.ell_max
    config['zfid'] = config['zref']
    config['Omega_m'] = data.dic_init['model']['Om']
    config['Omega_de'] = data.dic_init['model']['OL']
    xi_obj = CorrelationFunction(config, r, mu, z, cf_model.tracer1, cf_model.tracer2)
    xi_new = xi_obj.compute(pk_obj._k, pk_obj.muk, pk_obj.dmuk, pk, pars)
    print('Base:', np.sum(xi_obj.xi_base))
    print('Evol:', np.sum(xi_obj.xi_evol))
    print('Growth:', np.sum(xi_obj.xi_growth))
    print(np.sum(xi_old))
    print(np.sum(xi_new))

# if __name__ == "__main__":
    # test_model()
