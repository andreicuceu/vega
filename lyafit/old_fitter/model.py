from functools import partial
import numpy as np

import scipy
from scipy import linalg
from scipy.sparse import csr_matrix

from . import pk, xi


class model:
    """
    Class for computing Lyman-alpha forest correlation function models
    """

    def __init__(self, dic_init, r=None, mu=None, z=None, dm=None):

        # ! For now we need to import r, mu, z and dm,
        # ! until I figure out how to compute defaults
        assert r is not None
        assert mu is not None
        assert z is not None
        self.r = r
        self.mu = mu
        self.z = z
        self.dm = dm

        self.name = dic_init['data']['name']
        self.tracer1 = {}
        self.tracer2 = {}
        self.tracer1['name'] = dic_init['data']['tracer1']
        self.tracer1['type'] = dic_init['data']['tracer1-type']
        if 'tracer2' in dic_init['data']:
            self.tracer2['name'] = dic_init['data']['tracer2']
            self.tracer2['type'] = dic_init['data']['tracer2-type']
        else:
            self.tracer2['name'] = self.tracer1['name']
            self.tracer2['type'] = self.tracer1['type']

        self.ell_max = dic_init['data']['ell-max']
        zeff = dic_init['model']['zeff']
        zref = dic_init['model']['zref']
        Om = dic_init['model']['Om']
        OL = dic_init['model']['OL']

        if 'hcd_model' in dic_init:
            self.pk = pk.pk(getattr(pk, dic_init['model']['model-pk']), dic_init['hcd_model']['name_hcd_model'])
        else:
            self.pk = pk.pk(getattr(pk, dic_init['model']['model-pk']))

        self.pk *= partial(getattr(pk, 'G2'), dataset_name=self.name)

        if 'pk-gauss-smoothing' in dic_init['model']:
            self.pk *= partial(getattr(pk, dic_init['model']['pk-gauss-smoothing']))
        if 'small scale nl' in dic_init['model']:
            self.pk *= partial(getattr(pk, dic_init['model']['small scale nl']), pk_fid=dic_init['model']['pk']*((1+zref)/(1.+zeff))**2)

        if 'velocity dispersion' in dic_init['model']:
            self.pk *= getattr(pk, dic_init['model']['velocity dispersion'])

        # add non linear large scales
        self.pk *= pk.pk_NL

        self.xi = partial(getattr(xi, dic_init['model']['model-xi']), name=self.name)

        self.z_evol = {}
        self.z_evol[self.tracer1['name']] = partial(getattr(xi, dic_init['model']['z evol {}'.format(self.tracer1['name'])]),zref=zeff)
        self.z_evol[self.tracer2['name']] = partial(getattr(xi, dic_init['model']['z evol {}'.format(self.tracer2['name'])]),zref=zeff)
        if dic_init['model']['growth function'] in ['growth_factor_de']:
            self.growth_function = partial(getattr(xi, dic_init['model']['growth function']), zref=zref, Om=Om, OL=OL)
        else:
            self.growth_function = partial(getattr(xi, dic_init['model']['growth function']), zref=zref)

        self.xi_rad_model = None
        if 'radiation effects' in dic_init['model']:
            self.xi_rad_model = partial(getattr(xi, dic_init['model']['radiation effects']), name=self.name)

        self.xi_rel_model = None
        if 'relativistic correction' in dic_init['model']:
            self.xi_rel_model = partial(getattr(xi, dic_init['model']['relativistic correction']), name=self.name)

        self.xi_asy_model = None
        if 'standard asymmetry' in dic_init['model']:
            self.xi_asy_model = partial(getattr(xi, dic_init['model']['standard asymmetry']), name=self.name)

        self.par_names = dic_init['parameters']['values'].keys()
        self.pars_init = dic_init['parameters']['values']
        self.par_error = dic_init['parameters']['errors']
        self.par_limit = dic_init['parameters']['limits']
        self.par_fixed = dic_init['parameters']['fix']

    def xi_model(self, k, pk_lin, pars):
        xi = self.xi(self.r, self.mu, k, pk_lin, self.pk,
                    tracer1 = self.tracer1, tracer2 = self.tracer2, ell_max = self.ell_max, **pars)
        print('Base:',np.sum(xi))
        evol = self.z_evol[self.tracer1['name']](self.z, self.tracer1, **pars)*self.z_evol[self.tracer2['name']](self.z, self.tracer2, **pars)
        xi *= evol
        print('Evol:', np.sum(xi))
        growth = self.growth_function(self.z, **pars)**2
        xi *= growth
        print('Growth:', np.sum(growth))

        if self.xi_rad_model is not None and pars['SB'] == True:
            xi += self.xi_rad_model(self.r, self.mu, self.tracer1, self.tracer2, **pars)

        if self.xi_rel_model is not None:
            xi += self.xi_rel_model(self.r, self.mu, k, pk_lin, self.tracer1, self.tracer2, **pars)

        if self.xi_asy_model is not None:
            xi += self.xi_asy_model(self.r, self.mu, k, pk_lin, self.tracer1, self.tracer2, **pars)

        xi = self.dm.dot(xi)

        return xi
