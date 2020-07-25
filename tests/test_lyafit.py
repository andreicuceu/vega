import pytest
import numpy as np

import lyafit.old_fitter.parser as parser

from lyafit.lyafit import LyaFit


@pytest.mark.skip
def test_lyafit():
    # New
    main_config = "D:\\work\\run\\DR16_new\\main.ini"
    lyafit = LyaFit(main_config)
    pk_full = lyafit.fiducial['pk_full']
    pk_smooth = lyafit.fiducial['pk_smooth']
    auto_model = lyafit.models['lyaxlya']
    new_xi = auto_model.compute(lyafit.params, pk_full, pk_smooth)

    # Old
    filename = "D:\\work\\run\\DR16\\chi2.ini"
    dic_init = parser.parse_chi2(filename)
    pars = dic_init['data sets']['data'][0].pars_init
    k = lyafit.fiducial['k']
    full_shape = False
    pars['SB'] = False
    old_data = dic_init['data sets']['data'][0]
    chi2 = old_data.chi2(k, pk_full, pk_smooth, full_shape, pars)

    # print('new peak xi:', np.sum(auto_model.xi['peak']))
    # print('new smooth xi:', np.sum(auto_model.xi['smooth']))
    # for name1, name2, in auto_model.corr_item.metal_correlations:
    #     print(name1 + '_' + name2)
        # print('new peak r:', np.sum(auto_model.Xi_metal[(name1, name2)]._r))
        # print('new peak mu:', np.sum(auto_model.Xi_metal[(name1, name2)]._mu))
        # print('new peak xi:', np.sum(auto_model.xi_metal['peak'][(name1, name2)]))
        # print('new smooth xi:', np.sum(auto_model.xi_metal['smooth'][(name1, name2)]))

    print('old xi full', np.sum(old_data.xi_full))
    print('new xi full', np.sum(new_xi))
    print('old xi full', np.sum(np.log(np.abs(old_data.xi_full))))
    print('new xi full', np.sum(np.log(np.abs(new_xi))))


# if __name__ == "__main__":
    # test_lyafit()
