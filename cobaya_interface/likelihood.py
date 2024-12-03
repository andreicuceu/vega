from cobaya.likelihood import Likelihood
import numpy as np
import pandas as pd
import os
import fnmatch

import matplotlib.pyplot as plt
from vega.vega_interface import VegaInterface

import scipy
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from typing import Optional


class Likelihood(Likelihood): # class that inherits from cobaya.likelihood

    def initialize(self, **params_values):
        '''
        Set up initial parameters
        '''
        self.vega = VegaInterface('cobaya_interface/configs/complex_main.ini') # Creates an instance of VegaInterface with a configuration file containing cosmological or model parameters
        self.effective_redshift = 2.33
        self.k_grid = np.logspace(-4,3.061640934061686,814) #np.logspace(-4,2,700) # grid of scales for power spectrum
        self.vega.fiducial['z_fiducial'] = self.effective_redshift   # fix z_eff z_fiducial


    def get_requirements(self):
        '''
        Specifies what cosmological parameters are required by cobaya
        '''
        return {'bias_LYA': None, 'beta_LYA': None,'D_M_fid': None, 'D_H_fid': None, 'or_photon':None, 'or_neutrino':None, \
                'H0': None, 'ombh2': None , 'omch2': None, 'omnuh2': None, 'omk': None,'As': None, 'ns': None, \
                'Pk_grid': {'z': [self.effective_redshift], 'k_max':10**3.061640934061686, 'nonlinear':[False]},
                'angular_diameter_distance':{'z': [self.effective_redshift]}, 'Hubble':{'z': [self.effective_redshift]},
                }

    def logp(self, **params_values):
        '''
        Method to calculate the log-likelihood based on the parameters provided
        '''
        scale_factor = 1/(1+self.effective_redshift) # scale factor at effective redshift
        h = params_values['H0']/100 # defines the reduced Hubble constant from provided H0
        D_M = (self.provider.get_angular_diameter_distance(self.effective_redshift)/scale_factor)*h # computes angular diameter distance to effective redshift 
        D_H = ((scipy.constants.c/1000)/self.provider.get_Hubble(self.effective_redshift))*h # computes Hubble distance
        params_values['ap_full'] = D_H[0]/params_values['D_H_fid'] # computes alpha_parallel 
        params_values['at_full'] = D_M[0]/params_values['D_M_fid'] # computes alpha_transverse
        
        k_Mpc, z, pk_Mpc = self.provider.get_Pk_grid(nonlinear = False) # retrieves power spectrum
        cs = interp1d(np.array(k_Mpc/h), np.array(pk_Mpc*(h**3)), kind='cubic')
        k_hMpc = self.k_grid # initial grid is in units of Mpc/h
        pk_hMpc = cs(k_hMpc) # interpolates retrieved power spectrum onto this grid
        
        if 'pk_full' not in self.vega.fiducial.keys(): # if vega fiducial pk_full is empty, store it here
            #print("self.vega.fiducial is empty")
            self.vega.fiducial['k'] = k_hMpc
            self.vega.fiducial['pk_full'] = pk_hMpc
        self.pk_full = pk_hMpc
            
        omega_m = (params_values['ombh2'] + params_values['omch2'] + params_values['omnuh2'])/(h**2) # computes Omega_m from baryons, cold dark matter, and non-relativistic neutrinos 
        self.vega.fiducial['Omega_m'] = omega_m
        self.vega.fiducial['Omega_de'] = (1 - params_values['omk'] - params_values['or_photon'] - params_values['or_neutrino'] - omega_m) # computes Omega_de from Omega_m, flatness, photons, and relativistic neutrinos
        
        chi2 = self.vega.chi2(params_values, direct_pk = self.pk_full) # uses vega to compute chi2 from power spectrum and parameter values 
        print('param values:', params_values)
        print('chi2:', chi2)
        return -chi2 / 2 # returns log likelihood
