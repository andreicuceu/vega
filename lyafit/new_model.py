from functools import partial
import numpy as np

import scipy
from scipy import linalg
from scipy.sparse import csr_matrix

# from . import new_pk, new_xi
from .new_pk import PowerSpectrum
from .new_xi import CorrelationFunction


class Model:
    """
    Class for computing Lyman-alpha forest correlation function models
    """

    def __init__(self, dic_init, r=None, mu=None, z=None, dm=None):

        # ! For now we need to import r, mu, z and dm,
        # ! until I add defaults
        assert r is not None
        assert mu is not None
        assert z is not None
        self._r = r
        self._mu = mu
        self._z = z
        self._dm = dm
        self._config = dic_init['model']

        self._name = dic_init['data']['name']
        self._tracer1 = {}
        self._tracer2 = {}
        self._tracer1['name'] = dic_init['data']['tracer1']
        self._tracer1['type'] = dic_init['data']['tracer1-type']
        if 'tracer2' in dic_init['data']:
            self._tracer2['name'] = dic_init['data']['tracer2']
            self._tracer2['type'] = dic_init['data']['tracer2-type']
        else:
            self._tracer2['name'] = self._tracer1['name']
            self._tracer2['type'] = self._tracer1['type']

        # TODO Move these to parser and update them in the config files
        self._config['ell_max'] = dic_init['data']['ell-max']
        self._config['zfid'] = self._config['zref']
        self._config['Omega_m'] = self._config['Om']
        self._config['Omega_de'] = self._config['OL']
        self._config['smooth_scaling'] = False

        # Initialize Power Spectrum object
        self.Pk_base = PowerSpectrum(self._config, self._tracer1,
                                     self._tracer2, self._name)

        # Initialize Correlation function object
        self.Xi_base = CorrelationFunction(self._config, self._r,
                                           self._mu, self._z,
                                           self._tracer1, self._tracer2)

        self.par_names = dic_init['parameters']['values'].keys()
        self.pars_init = dic_init['parameters']['values']
        self.par_error = dic_init['parameters']['errors']
        self.par_limit = dic_init['parameters']['limits']
        self.par_fixed = dic_init['parameters']['fix']

    def _compute_model(self, pars, pk_lin, component='smooth'):
        k, muk, pk_model = self.Pk_base.compute(pk_lin, pars)
        self.pk_model[component] = pk_model.copy()

        xi_model = self.Xi_base.compute(k, muk, self.Pk_base.dmuk,
                                        pk_model, pars)
        self.xi_model[component] = xi_model.copy()

        if self._dm is not None:
            xi_model = self._dm.dot(xi_model)
            self.xi_model_distorted[component] = xi_model.copy()

        return xi_model

    def compute(self, pars, pk_full, pk_smooth):
        pars['smooth_scaling'] = self._config['smooth_scaling']
        self.pk_model = {}
        self.xi_model = {}
        self.xi_model_distorted = {}

        pars['peak'] = True
        xi_peak = self._compute_model(pars, pk_full - pk_smooth, 'peak')

        pars['peak'] = False
        xi_smooth = self._compute_model(pars, pk_smooth, 'smooth')

        xi_full = pars['bao_amp'] * xi_peak + xi_smooth
        return xi_full
