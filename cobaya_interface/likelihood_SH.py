from cobaya.likelihood import Likelihood
import numpy as np
import pandas as pd
import os
import fnmatch

import matplotlib.pyplot as plt
from vega.vega_interface import VegaInterface, Minimizer

import scipy
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from typing import Optional


class Likelihood(Likelihood): # class that inherits from cobaya.likelihood

    correlation_type: Optional[str]
    vega_ini: Optional[str]

    def initialize(self, **params_values):
        '''
        Set up initial parameters
        '''
        self.vega = VegaInterface(self.vega_ini) # Creates an instance of VegaInterface with a configuration file containing cosmological or model parameters
        _ = self.vega.compute_model(run_init=False)
        
        # Check if we need to run over a Monte Carlo mock
        if 'control' in self.vega.main_config:
            run_montecarlo = self.vega.main_config['control'].getboolean('run_montecarlo', False)
            if run_montecarlo and self.vega.mc_config is not None:
                _ = self.vega.initialize_monte_carlo()
                # Get the MC seed and forecast flag
                #seed = self.vega.main_config['control'].getint('mc_seed', 0)
                #forecast = self.vega.main_config['control'].getboolean('forecast', False)

                # Create the mocks
                #self.vega.monte_carlo_sim(self.vega.mc_config['params'], seed=seed, forecast=forecast)

                # Set to sample the MC params
                #sampling_params = self.vega.mc_config['sample']
                #self.vega.minimizer = Minimizer(self.vega.chi2, sampling_params)
            elif run_montecarlo:
                raise ValueError('You asked to run over a Monte Carlo simulation,'
                                 ' but no "[monte carlo]" section provided.')
        
        self.effective_redshift = 2.33
        self.k_grid = np.logspace(-3,1,260) # grid of scales for power spectrum
        self.vega.fiducial['z_fiducial'] = self.effective_redshift   # fix z_eff z_fiducial


    def get_requirements(self):
        '''
        Specifies what cosmological parameters are required by cobaya
        '''
        reqs = {'bias_LYA': None, 'beta_LYA': None,'D_M_fid': None, 'D_H_fid': None, 'or_photon':None, 'or_neutrino':None, \
                'H0': None, 'ombh2': None , 'omch2': None, 'omnuh2': None, 'omk': None,'As': None, 'ns': None, \
                'Pk_grid': {'z': [self.effective_redshift], 'k_max': 10, 'nonlinear':[False]},
                'angular_diameter_distance':{'z': [self.effective_redshift]}, 'Hubble':{'z': [self.effective_redshift]},
                }
        
        if self.correlation_type=='AUTO+CROSS':
            reqs['sigma_velo_disp_lorentz_QSO']=None
            reqs['bias_QSO']=None
            reqs['sigma8_z'] = {'z': [self.effective_redshift]} # needed?
            reqs['fsigma8'] = {'z': [self.effective_redshift]}

        return reqs

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

        if self.correlation_type=='AUTO+CROSS':
            growth_rate = (self.provider.get_fsigma8(self.effective_redshift))/(self.provider.get_sigma8_z(self.effective_redshift))
            #print('growth rate from CAMB:', growth_rate)
            params_values["growth_rate"] = growth_rate[0]
            params_values["_derived"]["growth_rate"] = growth_rate[0]
            params_values["_derived"]["f_sigma8"] = self.provider.get_fsigma8(self.effective_redshift)[0]
        
        k_Mpc, z, pk_Mpc = self.provider.get_Pk_grid(nonlinear = False) # retrieves power spectrum
        #print('retrieved Pk grid:', k_Mpc[0], k_Mpc[-1], len(k_Mpc))
        cs = interp1d(np.array(k_Mpc/h), np.array(pk_Mpc*(h**3)), kind='cubic')
        k_hMpc = self.k_grid # initial grid is in units of Mpc/h
        #print('self k grid:', k_hMpc[0], k_hMpc[-1], len(k_hMpc))
        pk_hMpc = cs(k_hMpc) # interpolates retrieved power spectrum onto this grid
        #print('interpolated Pk onto self k grid')
        
        if 'pk_full' not in self.vega.fiducial.keys(): # if vega fiducial pk_full is empty, store it here
            print("self.vega.fiducial is empty")
            self.vega.fiducial['k'] = k_hMpc
            self.vega.fiducial['pk_full'] = pk_hMpc
        self.pk_full = pk_hMpc
            
        omega_m = (params_values['ombh2'] + params_values['omch2'] + params_values['omnuh2'])/(h**2) # computes Omega_m from baryons, cold dark matter, and non-relativistic neutrinos 
        self.vega.fiducial['Omega_m'] = omega_m
        self.vega.fiducial['Omega_de'] = (1 - params_values['omk'] - params_values['or_photon'] - params_values['or_neutrino'] - omega_m) # computes Omega_de from Omega_m, flatness, photons, and relativistic neutrinos
        
        chi2 = self.vega.chi2(params_values, direct_pk = self.pk_full) # uses vega to compute chi2 from power spectrum and parameter values 
        print('param values:', params_values)
        print('chi2:', chi2)
        print('Using SH correction')
        return (-24358/2)*np.log(1+chi2/(24358-1)) # returns log likelihood using Sellentin & Heavens correction
        #return -chi2 / 2 # returns log likelihood
