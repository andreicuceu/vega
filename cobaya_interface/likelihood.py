from cobaya.likelihood import Likelihood
import numpy as np
import pandas as pd
import os
import fnmatch

import matplotlib.pyplot as plt
from vega.vega_interface_mod import VegaInterface

import scipy
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from typing import Optional


class Likelihood(Likelihood):

    def initialize(self, **params_values):

        #print(params_values)
        self.vega = VegaInterface('simple_example/complex_main.ini')
        self.effective_redshift = 2.3
        self.k_grid = np.logspace(-4,2,700)
        self.vega.fiducial['z_fiducial'] = self.effective_redshift   # fix z_eff z_fiducial


    def get_requirements(self):
        
        return {'bias_LYA': None, 'beta_LYA': None,'D_M_fid': None, 'D_H_fid': None, 'or_photon':None, 'or_neutrino':None, \
                'H0': None, 'ombh2': None , 'omch2': None, 'omnuh2': None, 'omk': None,'As': None, 'ns': None, \
                'Pk_grid': {'z': [self.effective_redshift], 'k_max':100, 'nonlinear':[False]},
                'angular_diameter_distance':{'z': [self.effective_redshift]}, 'Hubble':{'z': [self.effective_redshift]},
                }

    def logp(self, **params_values):

        scale_factor = 1/(1+self.effective_redshift)
        h = params_values['H0']/100
        D_M = (self.provider.get_angular_diameter_distance(self.effective_redshift)/scale_factor)*h
        D_H = ((scipy.constants.c/1000)/self.provider.get_Hubble(self.effective_redshift))*h
        params_values['ap_full'] = D_H[0]/params_values['D_H_fid']
        params_values['at_full'] = D_M[0] / params_values['D_M_fid']
        
        k_Mpc, z, pk_Mpc = self.provider.get_Pk_grid(nonlinear = False)
        cs = interp1d(np.array(k_Mpc/h), np.array(pk_Mpc*(h**3)), kind='cubic')
        k_hMpc = self.k_grid
        pk_hMpc = cs(k_hMpc)
        
        if 'pk_full' not in self.vega.fiducial.keys():
            #print("self.vega.fiducial is empty")
            self.vega.fiducial['k'] = k_hMpc
            self.vega.fiducial['pk_full'] = pk_hMpc
        self.pk_full = pk_hMpc
            
        omega_m =(params_values['ombh2'] + params_values['omch2'] + params_values['omnuh2'])/(h**2)
        self.vega.fiducial['Omega_m'] = omega_m
        self.vega.fiducial['Omega_de'] = (1 - params_values['omk'] - params_values['or_photon'] - params_values['or_neutrino'] - omega_m) 
        
        chi2 = self.vega.chi2(params_values, direct_pk = self.pk_full)
        return -chi2 / 2