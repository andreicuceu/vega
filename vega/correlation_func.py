import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, interp1d
from astropy.table import Table
from scipy.stats import gaussian_kde
import picca.constants
from astropy.io import fits
from numba import njit
import matplotlib.pyplot as plt
import fitsio

from . import utils
from .utils import gen_gamma, jitted_interp, get_gamma_QQ

class CorrelationFunction:
    """Correlation function computation and handling.

    # ! Slow operations should be kept in init as that is only called once

    # ! Compute is called many times and should be fast

    Extensions should have their separate method of the form
    'compute_extension' that can be called from outside
    """
    def __init__(self, config, fiducial, coordinates, scale_params,
                 tracer1, tracer2, xcf_obj, metal_corr=False):
        """

        Parameters
        ----------
        config : ConfigParser
            model section of config file
        fiducial : dict
            fiducial config
        coordinates : Coordinates
            Vega coordinates object
        scale_params : ScaleParameters
            ScaleParameters object
        tracer1 : dict
            Config of tracer 1
        tracer2 : dict
            Config of tracer 2
        metal_corr : bool, optional
            Whether this is a metal correlation, by default False
        """
        self._config = config
        self._r = coordinates.r_grid
        self._mu = coordinates.mu_grid
        self._z = coordinates.z_grid
        self._multipole = config.getint('single_multipole', -1)
        self._tracer1 = tracer1
        self._tracer2 = tracer2
        self._z_eff = fiducial['z_eff']
        self._rel_z_evol = (1. + self._z) / (1 + self._z_eff)
        self._scale_params = scale_params
        self._metal_corr = metal_corr
        self._names = [tracer1['name'], tracer2['name']]

        # Check if we need delta rp (Only for the cross)
        self._delta_rp_name = None
        if tracer1['type'] == 'discrete' and tracer2['type'] != 'discrete':
            self._delta_rp_name = 'drp_' + tracer1['name']
        elif tracer2['type'] == 'discrete' and tracer1['type'] != 'discrete':
            self._delta_rp_name = 'drp_' + tracer2['name']

        # Precompute growth
        self._z_fid = fiducial['z_fiducial']
        self._Omega_m = fiducial.get('Omega_m', None)
        self._Omega_de = fiducial.get('Omega_de', None)
        if not config.getboolean('old_growth_func', False):
            self.xi_growth = self.compute_growth(
                self._z, self._z_fid, self._Omega_m, self._Omega_de)
        else:
            self.xi_growth = self.compute_growth_old(
                self._z, self._z_fid, self._Omega_m, self._Omega_de)

        # Check for QSO radiation modeling and check if it is QSOxLYA
        # Does this work for the QSO auto as well?
        self.radiation_flag = False
        if 'radiation effects' in self._config:
            self.radiation_flag = self._config.getboolean('radiation effects')
            if self.radiation_flag:
                if not ('QSO' in self._names and 'LYA' in self._names):
                    raise ValueError('You asked for QSO radiation effects, but it'
                                     ' can only be applied to the cross (QSOxLya)')

        # Check for relativistic effects and standard asymmetry
        self.relativistic_flag = False
        if 'relativistic correction' in self._config:
            self.relativistic_flag = self._config.getboolean('relativistic correction')

        self.asymmetry_flag = False
        if 'standard asymmetry' in self._config:
            self.asymmetry_flag = self._config.getboolean('standard asymmetry')
        if self.relativistic_flag or self.asymmetry_flag:
            types = [self._tracer1['type'], self._tracer2['type']]
            if ('continuous' not in types) or (types[0] == types[1]):
                raise ValueError('You asked for relativistic effects or standard asymmetry,'
                                 ' but they only work for the cross')

        # Place holder for interpolation function for DESI intrumental systematics
        self.desi_instrumental_systematics_interp = None

        #initialise cont dist models 
        if 'cont_dist_cross' in self._config:
            self.cont_dist_flag = self._config.getboolean('cont_dist_cross')
            if self.cont_dist_flag:
                if not ('QSO' in self._names and 'LYA' in self._names):
                    raise ValueError('The continuum distortion cross model can'
                                    'only be applied to the cross (LyaxQSO)')
                
                if np.any(self._tracer1['weights-path'] is None or self._tracer2['qso-cat-path'] is None):
                    raise ValueError('The continuum distortion cross model requires'
                                    'weights-tracer1 and a qso catalogue in qso-cat')
                
                self._rp_grid = coordinates.rp_regular_grid
                self._rt_grid = coordinates.rt_regular_grid
                
                self._init_zerr_cross(self._tracer2['qso-cat-path'], 
                                      self._tracer1['weights-path'], 
                                      self._config.get('qso-auto-corr'))
                
                self._prep_gamma_model()

        elif 'cont_dist_auto' in self._config:
            self.cont_dist_flag = self._config.getboolean('cont_dist_auto')
            if self.cont_dist_flag:
                if not ('LYA' in self._names[0] and 'LYA' in self._names[1]):
                    raise ValueError('The continuum distortion auto model can'
                                    'only be applied to the Lya auto (LyaxLya)')
                
                if np.any(self._tracer1['weights-path'] is None or self._tracer2['qso-cat-path'] is None):
                    raise ValueError('The continuum distortion auto model requires'
                                    'weights-tracer1 and "qso-cat" ')

                self._rp_grid = coordinates.rp_regular_grid.reshape(
                    coordinates.rp_nbins,coordinates.rt_nbins).T[0]
                self._rt_grid = coordinates.rt_regular_grid.reshape(
                    coordinates.rp_nbins,coordinates.rt_nbins)[0]

                self._init_zerr_auto(
                    self._tracer2['qso-cat-path'], 
                    self._tracer1['weights-path'], 
                    self._config.get('qso-lya-cross'),
                    xcf_obj
                        )
                
                self._prep_delta_gamma_model()

    def P_ZQSO(self,z,zqso,P_zqso):
        return np.interp(z,zqso,P_zqso)

    def _init_zerr_cross(self, qso_cat, delta_attr, qso_auto_corr):

        self.cosmo_class = self._get_cosmo()
        self._nbins_z = 20
        z0, z1 = 2.1,3
        self._zq1 = np.linspace(z0, z1, self._nbins_z)
        self._dzq1 = self._zq1[1]-self._zq1[0] 
        self.lya_wave = 1215.67
        self._gamma_wave = None
        self._gamma = None

        #just for rf limits
        with fitsio.FITS(delta_attr) as hdul:
            #from attributes
            self._min_rf_pix = min(10**hdul[3]['LOGLAM_REST'][:])
            self._max_rf_pix = max(10**hdul[3]['LOGLAM_REST'][:])


        self._nbins_wave = 50
        self._rf_wave = np.linspace(1150, self._max_rf_pix, self._nbins_wave)
        
        #get zqso hist from qso cat
        with fitsio.FITS(qso_cat) as hdul:
            zcat = hdul[1]['Z'][:]
            self._p_qso,bedges,_= plt.hist(zcat,bins=100,density=True,cumulative=False);
            self._zqso=(bedges[:-1]+bedges[1:])/2

        #read qso auto
        qso_auto_hdul = np.loadtxt(qso_auto_corr)
        self._r_qso_auto = qso_auto_hdul[0]
        self._xi_qso_auto = qso_auto_hdul[1]


        self._rpix = np.abs(np.add.outer(self.cosmo_class.get_r_comov(self._zq1), self._rp_grid)).T
        self._zpix = self.cosmo_class.distance_to_redshift(self._rpix)

        d_M_zq1 = self.cosmo_class.get_dist_m(self._zq1)
        d_c_zq1 = self.cosmo_class.get_r_comov(self._zq1)

        #possible redshifts of quasars with correlating pixel, shape (nAxi,nzq1,nwave)
        self._zq2 = (1+self._zpix[:,:,None]) * (self.lya_wave / self._rf_wave) - 1
        #self._dzq2 = (self._zq2[0][:,1] - self._zq2[0][:,0])[:,None]

        D_M_zq2 = self.cosmo_class.get_dist_m(self._zq2)
        D_c_zq2 = self.cosmo_class.get_r_comov(self._zq2)

        self._diff_dc = (D_c_zq2 - d_c_zq1[:,None])
        self._diff_dm = (D_M_zq2 + d_M_zq1[:,None])

        self._obs_wave = (1 + self._zq2) * self._rf_wave

        self._p_vec_q1 = np.interp(self._zq1,self._zqso,self._p_qso)[:,None]

        self._prep_matrix = np.zeros((len(self._rp_grid), self._nbins_z, self._nbins_wave)).astype(float)
        self._gamma_M = np.zeros((len(self._rp_grid), self._nbins_z, self._nbins_wave)).astype(float)
        self._mask_M = np.zeros((len(self._rp_grid), self._nbins_z, self._nbins_wave)).astype(bool)
        rf_wave_M = np.zeros((len(self._rp_grid), self._nbins_z, self._nbins_wave)).astype(float)
        self._rf_wave_M = rf_wave_M + self._rf_wave

        measured_gamma_file = self._config.get('measured-gamma', None)
        if measured_gamma_file is not None:
            self._gamma_wave = np.load(measured_gamma_file)['restwave']
            self._gamma = np.load(measured_gamma_file)['gamma']
        else:
            spec_path = ('/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_all/v5.9.4/mock-0/desi-4.0-4/spectra-16/0/1/truth-16-1.fits')
            self._spec = fitsio.FITS(spec_path)[2]['EMLINES'][:][5]



    def _init_zerr_auto(self, qso_cat, delta_attr, cross_corr, xcf_obj=None):
        
        self._gamma_wave = None
        self._gamma = None
        self.cosmo_class = self._get_cosmo()
        self.lya_wave = 1215.67
        self._nbins_wave = 400
        self._nbins_z = 50

        # data for distance/redshift interpolator that is numba friendly
        self._z_interp_list = np.linspace(1.5, 5, 1000)
        self._dist_interp_list = self.cosmo_class.get_r_comov(self._z_interp_list)

        #define list of redshifts to evaluate model over 
        # It's roughly the range of the quasar sample but it's not that important
        #50 bins is enough for convergence
        self._zlist = np.linspace(2, 3.5, self._nbins_z)
        self._dz = abs((3.5 - 2) / self._nbins_z)
        #get zqso hist from qso cat
        with fitsio.FITS(qso_cat) as hdul:
            kde = gaussian_kde(hdul[1]['Z'][:])
        self._p_qso = kde(self._zlist)

        #get zpix hist from delta atts
        with fitsio.FITS(delta_attr) as hdul:
            num_pixels = hdul[2]['NUM_PIXELS'][:]
            log_wave = 10**hdul[2]['LOGLAM'][:]
            self._z_pixels = (log_wave / self.lya_wave) - 1
            self._pdf_pixels = num_pixels / sum(num_pixels * np.diff(self._z_pixels)[0])

            #from attributes
            self._min_rf_pix = min(10**hdul[3]['LOGLAM_REST'][:])
            self._max_rf_pix = max(10**hdul[3]['LOGLAM_REST'][:])
            #number of bins for _rf_wave is being tested now but will be fixed at convergence point
            self._rf_wave = np.linspace(1150, self._max_rf_pix, self._nbins_wave)

        self._cross_interp = None
        if cross_corr is not None:
            with fitsio.FITS(cross_corr) as hdul:
                data = Table(hdul[1].read())
                rp_x = data['RP'].reshape(100, 50).T[0]
                rt_x = data['RT'].reshape(100, 50)[0]
                try:
                    da_x = data['DA'].reshape(100, 50)
                except:
                    da_x = data['DA_BLIND'].reshape(100, 50)
                self._cross_interp = RectBivariateSpline(rp_x, rt_x, da_x)

        # else:
        #     raise ValueError('The continuum distortion auto model requires either'
        #                         'a measured cross-correlation ["model"]["qso-lya-cross"] or '
        #                        'a cross-correlation model.')

        measured_gamma_file = self._config.get('measured-gamma', None)
        if measured_gamma_file is not None:
            self._gamma_wave = np.load(measured_gamma_file)['restwave']
            self._gamma = np.load(measured_gamma_file)['gamma']
        else:
            spec_path = ('/global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_all/v5.9.4/mock-0/desi-4.0-4/spectra-16/0/1/truth-16-1.fits')
            self._spec = fitsio.FITS(spec_path)[2]['EMLINES'][:][5]




    def _part1(self):
        rf_mask = np.zeros((self._nbins_z, self._nbins_wave), dtype=bool)
        for i, zq2  in enumerate(self._zlist):
            wave_obs2 = (1 + zq2) * self._rf_wave
            w2 = (3600 < wave_obs2) & (wave_obs2 < 5772)

            rf_mask[i, :] = w2

        run_rf = np.array([mask.sum() > 1 for mask in rf_mask])
        rf_mask = rf_mask[run_rf]

        zpix2 = np.outer((1 + self._zlist[run_rf]), (self._rf_wave / self.lya_wave)) - 1
        zpix2[~rf_mask] = 0
        dpix2 = jitted_interp(zpix2, self._z_interp_list, self._dist_interp_list)
        dpix2[~rf_mask] = 0
        dzq2 = jitted_interp(self._zlist[run_rf], self._z_interp_list, self._dist_interp_list)

        return rf_mask, run_rf, dpix2, dzq2

    def _part2(self, zpix1, rp_X, dzpix1, rf_mask, dpix2, dzq2):
        for i, rp_A in enumerate(self._rp_grid):
            dpix1 = dpix2[rf_mask] + rp_A
            zpix1[i][rf_mask] = jitted_interp(dpix1, self._dist_interp_list, self._z_interp_list)
            rp_X[i][rf_mask] = (dzq2[:, None] * rf_mask)[rf_mask] - dpix1
            dzpix1[i] = np.array([abs(zpix[rf_mask[j]][1] - zpix[rf_mask[j]][0]) for j, zpix in enumerate(zpix1[i])])
        
        return zpix1, rp_X, dzpix1

    def _part3(self, rp_X, rt_X, rf_mask, mask_rp, pix1_hist, dzpix1, run_rf):
        cross_full_grid = self._cross_interp(rp_X[None, :], rt_X, grid=False)
        cross_full_grid[:, :, ~rf_mask] = 0
        cross_full_grid[:, ~mask_rp] = 0

        p0 = np.einsum('ijkl,jkl->ijkl', cross_full_grid, pix1_hist)
        p1 = np.einsum('jkl,kl->jkl', dzpix1[:, :, None], self._p_qso[run_rf, None])
        p2 = np.einsum('jkl,kl->jkl', p1, np.full(run_rf.sum(), self._dz)[:, None])
        p3 = np.einsum('ijkl,jkl->ijkl', p0, p2)
        return p3

    def _prep_gamma_model(self):
        for k, (rp_A, rt_A) in enumerate(zip(self._rp_grid, self._rt_grid)):   
            mask = (3600<self._obs_wave[k])&(self._obs_wave[k]<5772)                
            
            #calculate r_Q
            rp_Q_M = self._diff_dc[k]

            #rt_Q = rt_A
            r_Q_M = np.sqrt(rp_Q_M**2 + rt_A**2)        
            
            #GET R, MU, INTERPOLATE
            #mus = (rp_Q_M / r_Q_M)
            
            #return r_Q_M, mus
            xi_Q_M = jitted_interp(r_Q_M, self._r_qso_auto, self._xi_qso_auto)  

            #log interp
            #xi_Q_M = jitted_interp(np.log10(r_Q_M), np.log10(self._r_qso_auto), np.log10(self._xi_qso_auto))  
            #xi_Q_M = 10**xi_Q_M

            p_vec_q2 = jitted_interp(self._zq2[k],self._zqso,self._p_qso)
            
            dzq2 = (self._zq2[k][:,1] - self._zq2[k][:,0])[:,None]

            #add to integral
            self._prep_matrix[k] += (self._p_vec_q1 * p_vec_q2 * abs(self._dzq1) * abs(dzq2) * xi_Q_M)
            
            self._mask_M[k] = mask

    def _prep_delta_gamma_model(self):
        rf_mask, run_rf, dpix2, dzq2 = self._part1()

        zpix1 = np.zeros((len(self._rp_grid), run_rf.sum(), self._nbins_wave), dtype=float)
        rp_X = np.zeros((len(self._rp_grid), run_rf.sum(), self._nbins_wave), dtype=float)
        dzpix1 = np.zeros((len(self._rp_grid), run_rf.sum()), dtype=float)

        zpix1, rp_X, dzpix1 = self._part2(zpix1, rp_X, dzpix1, rf_mask, dpix2, dzq2)

        mask_rp = (-200 < rp_X) & (rp_X < 200)
        pix1_hist = jitted_interp(zpix1, self._z_pixels, self._pdf_pixels)
        pix1_hist[:, ~rf_mask] = 0
        pix1_hist[~mask_rp] = 0

        rt_X = self._rt_grid[:, None, None, None]
        prep_matrix = self._part3(rp_X, rt_X, rf_mask, mask_rp, pix1_hist, dzpix1, run_rf)

        self._prep_matrix = prep_matrix
        self._run_rf = run_rf
        self._rf_mask = rf_mask

    def _compute_gamma_extension(self, g_rf_wave, params):

        L1,L2 = 1205.1, 1213
        p1, p2 = params['Ca1'], params['Ca2']

        #this may be a lot slower
        if self._gamma is None:
            mean_gamma = get_gamma_QQ(g_rf_wave, params['cont_sigma_velo'],self._spec)

            if max(g_rf_wave)<L1:
                mean_gamma += p1 / (g_rf_wave - L1)**2
            mean_gamma += p2 / (g_rf_wave - L2)**2

            return mean_gamma

        else:
            mean_gamma = jitted_interp(g_rf_wave,self._gamma_wave,self._gamma) 
            
            if max(g_rf_wave)<L1:
                mean_gamma += p1 * jitted_interp(L1,self._gamma_wave,self._gamma) / (g_rf_wave - L1)**2
            mean_gamma += p2 * jitted_interp(L2,self._gamma_wave,self._gamma) / (g_rf_wave - L2)**2

            return mean_gamma

    def _compute_gamma_extension_QSO(self, g_rf_wave, params):

        L1,L2 = 1205.1, 1213
        p1, p2 = params['Ca1'], params['Ca2']

        #this may be a lot slower
        if self._gamma is None:
            mean_gamma = get_gamma_QQ(g_rf_wave, params['cont_sigma_velo'],self._spec)

            if max(g_rf_wave)<L1:
                mean_gamma += p1 / (g_rf_wave - L1)**2
            mean_gamma += p2 / (g_rf_wave - L2)**2

            return mean_gamma

        else:
            mean_gamma = jitted_interp(g_rf_wave,self._gamma_wave,self._gamma) 
            
            if max(g_rf_wave)<L1:
                mean_gamma += p1 * jitted_interp(L1,self._gamma_wave,self._gamma) / (g_rf_wave - L1)**2
            mean_gamma += p2 * jitted_interp(L2,self._gamma_wave,self._gamma) / (g_rf_wave - L2)**2

            return mean_gamma

    def _compute_gamma_auto(self, params):
        #need to have two options for using model or measured
        gamma_zs = np.zeros((self._run_rf.sum(), self._nbins_wave), dtype=float)
        for i, _ in enumerate(self._zlist[self._run_rf]):
            gamma_zs[i][self._rf_mask[i]] = self._compute_gamma_extension(self._rf_wave[self._rf_mask[i]], params)
        return  params['cont_dist_amp_auto'] * gamma_zs

    def compute_delta_gamma_model(self, params):
        gamma_mod = self._compute_gamma_auto(params)
        p4 = np.einsum('ijkl,kl->ijkl', self._prep_matrix, gamma_mod)
        auto_model = np.einsum('ijkl->ji', p4)
        return auto_model.flatten()

    def compute_gamma_model(self,params):
        gamma_mod = np.zeros_like(self._rp_grid)
        gamma_M = np.zeros((len(self._rp_grid), self._nbins_z, self._nbins_wave)).astype(float)
        gamma_M += self._compute_gamma_extension_QSO(self._rf_wave, params)

        for k, rp_A in enumerate(self._rp_grid):         
            #gamma_M_k = params['cont_dist_amp_cross'] * self._compute_gamma_extension_QSO(self._rf_wave_M[k][self._mask_M[k]], params)
            #gamma_mod[k] =  np.sum(self._prep_matrix[k][self._mask_M[k]] * gamma_M_k) 
            gamma_mod[k] = np.sum(self._prep_matrix[k][self._mask_M[k]] * gamma_M[k][self._mask_M[k]])
        return params['cont_dist_amp_cross'] * gamma_mod

    def _get_cosmo(self):
        return picca.constants.Cosmo(Om=self._Omega_m)

    def compute(self, pk, pk_lin, PktoXi_obj, params):
        """Compute correlation function for input P(k).

        Parameters
        ----------
        pk : ND Array
            Input power spectrum
        pk_lin : 1D Array
            Linear isotropic power spectrum
        PktoXi_obj : vega.PktoXi
            An instance of the transform object used to turn Pk into Xi
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output correlation function
        """
        # Compute the core
        xi = self.compute_core(pk, PktoXi_obj, params)

        # Add bias evolution
        xi *= self.compute_bias_evol(params)

        # Add growth
        xi *= self.xi_growth

        # Add QSO radiation modeling for cross
        if self.radiation_flag and not params['peak']:
            xi += self.compute_qso_radiation(params)

        # Add relativistic effects
        if self.relativistic_flag:
            xi += self.compute_xi_relativistic(pk_lin, PktoXi_obj, params)

        # Add standard asymmetry
        if self.asymmetry_flag:
            xi += self.compute_xi_asymmetry(pk_lin, PktoXi_obj, params)

        return xi

    def compute_core(self, pk, PktoXi_obj, params):
        """Compute the core of the correlation function.

        This does the Hankel transform of the input P(k),
        sums the necessary multipoles and rescales the coordinates

        Parameters
        ----------
        pk : ND Array
            Input power spectrum
        PktoXi_obj : vega.PktoXi
            An instance of the transform object used to turn Pk into Xi
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output correlation function
        """

        # Check for delta rp
        delta_rp = 0.
        if self._delta_rp_name is not None:
            delta_rp = params.get(self._delta_rp_name, 0.)

        # Get rescaled Xi coordinates
        ap, at = self._scale_params.get_ap_at(params, metal_corr=self._metal_corr)

        rescaled_r, rescaled_mu = self._rescale_coords(self._r, self._mu, ap, at, delta_rp)

        # Compute correlation function
        xi = PktoXi_obj.compute(rescaled_r, rescaled_mu, pk, self._multipole)

        return xi

    @staticmethod
    def _rescale_coords(r, mu, ap, at, delta_rp=0.):
        """Rescale Xi coordinates using ap/at.

        Parameters
        ----------
        r : ND array
            Array of radius coords of Xi
        mu : ND array
            Array of mu = rp/r coords of Xi
        ap : float
            Alpha parallel
        at : float
            Alpha transverse
        delta_rp : float, optional
            Delta radius_parallel - nuisance correction for wrong redshift,
            used for discrete tracers, by default 0.

        Returns
        -------
        ND Array
            Rescaled radii
        ND Array
            Rescaled mu
        """
        mask = r != 0
        rp = r[mask] * mu[mask] + delta_rp
        rt = r[mask] * np.sqrt(1 - mu[mask]**2)
        rescaled_rp = ap * rp
        rescaled_rt = at * rt

        rescaled_r = np.zeros(len(r))
        rescaled_mu = np.zeros(len(mu))
        rescaled_r[mask] = np.sqrt(rescaled_rp**2 + rescaled_rt**2)
        rescaled_mu[mask] = rescaled_rp / rescaled_r[mask]

        return rescaled_r, rescaled_mu

    def compute_bias_evol(self, params):
        """Compute bias evolution for the correlation function.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        # Compute the bias evolution
        bias_evol = self._get_tracer_evol(params, self._tracer1['name'])
        bias_evol *= self._get_tracer_evol(params, self._tracer2['name'])

        return bias_evol

    def _get_tracer_evol(self, params, tracer_name):
        """Compute tracer bias evolution.

        Parameters
        ----------
        params : dict
            Computation parameters
        tracer_name : string
            Name of tracer

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        handle_name = 'z evol {}'.format(tracer_name)

        if handle_name in self._config:
            evol_model = self._config.get(handle_name, 'standard')
        else:
            evol_model = self._config.get('z evol', 'standard')

        # Compute the bias evolution using the right model
        if 'croom' in evol_model:
            bias_evol = self._bias_evol_croom(params, tracer_name)
        else:
            bias_evol = self._bias_evol_std(params, tracer_name)

        return bias_evol

    def _bias_evol_std(self, params, tracer_name):
        """Bias evolution standard model.

        Parameters
        ----------
        params : dict
            Computation parameters
        tracer_name : string
            Tracer name

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        p0 = params['alpha_{}'.format(tracer_name)]
        bias_z = self._rel_z_evol**p0
        return bias_z

    def _bias_evol_croom(self, params, tracer_name):
        """Bias evolution Croom model for QSO, see Croom et al. 2005.

        Parameters
        ----------
        params : dict
            Computation parameters
        tracer_name : string
            Tracer name

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        assert tracer_name == "QSO"
        p0 = params["croom_par0"]
        p1 = params["croom_par1"]
        bias_z = (p0 + p1*(1. + self._z)**2) / (p0 + p1 * (1 + self._z_eff)**2)
        return bias_z

    def compute_growth(self, z_grid=None, z_fid=None,
                       Omega_m=None, Omega_de=None):
        """Compute growth factor.

        Implements eq. 7.77 from S. Dodelson's Modern Cosmology book.

        Returns
        -------
        ND Array
            Growth factor
        """
        # Check the defaults
        if z_grid is None:
            z_grid = self._z
        if z_fid is None:
            z_fid = self._z_fid
        if Omega_m is None:
            Omega_m = self._Omega_m
        if Omega_de is None:
            Omega_de = self._Omega_de

        # Check if we have dark energy
        if Omega_de is None:
            growth = (1 + z_fid) / (1. + z_grid)
            return growth**2

        # Compute the growth at each redshift on the grid
        growth = utils.growth_function(z_grid, Omega_m, Omega_de)
        # Scale to the fiducial redshift
        growth /= utils.growth_function(z_fid, Omega_m, Omega_de)

        return growth**2

    def compute_growth_old(self, z_grid=None, z_fid=None, Omega_m=None, Omega_de=None):
        def hubble(z, Omega_m, Omega_de):
            return np.sqrt(Omega_m*(1+z)**3 + Omega_de + (1-Omega_m-Omega_de)*(1+z)**2)

        def dD1(a, Omega_m, Omega_de):
            z = 1/a-1
            return 1./(a*hubble(z, Omega_m, Omega_de))**3

        # Calculate D1 in 100 values of z between 0 and zmax, then interpolate
        nbins = 100
        zmax = 5.
        z = zmax * np.arange(nbins, dtype=float) / (nbins-1)
        D1 = np.zeros(nbins, dtype=float)
        pars = (Omega_m, Omega_de)
        for i in range(nbins):
            a = 1/(1+z[i])
            D1[i] = 5/2.*Omega_m*hubble(z[i], *pars)*quad(dD1, 0, a, args=pars)[0]

        D1 = interp1d(z, D1)

        growth = D1(z_grid) / D1(z_fid)
        return growth**2

    def compute_qso_radiation(self, params):
        """Model the contribution of QSO radiation to the cross
        (the transverse proximity effect)

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        1D
            Xi QSO radiation model
        """
        assert 'QSO' in [self._tracer1['name'], self._tracer2['name']]
        assert self._tracer1['name'] != self._tracer2['name']

        # Compute the shifted r and mu grids
        delta_rp = params.get(self._delta_rp_name, 0.)
        rp = self._r * self._mu + delta_rp
        rt = self._r * np.sqrt(1 - self._mu**2)
        r_shift = np.sqrt(rp**2 + rt**2)
        mu_shift = rp / r_shift

        # Get the QSO radiation model parameters
        strength = params['qso_rad_strength']
        asymmetry = params['qso_rad_asymmetry']
        lifetime = params['qso_rad_lifetime']
        decrease = params['qso_rad_decrease']

        # Compute the QSO radiation model
        xi_rad = strength / (r_shift**2) * (1 - asymmetry * (1 - mu_shift**2))
        xi_rad *= np.exp(-r_shift * ((1 + mu_shift) / lifetime + 1 / decrease))
        return xi_rad

    def compute_xi_relativistic(self, pk, PktoXi_obj, params):
        """Calculate the cross-correlation contribution from
        relativistic effects (Bonvin et al. 2014).

        Parameters
        ----------
        pk : ND Array
            Input power spectrum
        PktoXi_obj : vega.PktoXi
            An instance of the transform object used to turn Pk into Xi
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output xi relativistic
        """
        assert 'continuous' in [self._tracer1['type'], self._tracer2['type']]
        assert self._tracer1['type'] != self._tracer2['type']

        # Get rescaled Xi coordinates
        delta_rp = params.get(self._delta_rp_name, 0.)
        ap, at = self._scale_params.get_ap_at(params, metal_corr=self._metal_corr)
        rescaled_r, rescaled_mu = self._rescale_coords(self._r, self._mu, ap, at, delta_rp)

        # Compute the correlation function
        xi_rel = PktoXi_obj.pk_to_xi_relativistic(rescaled_r, rescaled_mu, pk, params)

        return xi_rel

    def compute_xi_asymmetry(self, pk, PktoXi_obj, params):
        """Calculate the cross-correlation contribution from
        standard asymmetry (Bonvin et al. 2014).

        Parameters
        ----------
        pk : ND Array
            Input power spectrum
        PktoXi_obj : vega.PktoXi
            An instance of the transform object used to turn Pk into Xi
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output xi asymmetry
        """
        assert 'continuous' in [self._tracer1['type'], self._tracer2['type']]
        assert self._tracer1['type'] != self._tracer2['type']

        # Get rescaled Xi coordinates
        delta_rp = params.get(self._delta_rp_name, 0.)
        ap, at = self._scale_params.get_ap_at(params, metal_corr=self._metal_corr)
        rescaled_r, rescaled_mu = self._rescale_coords(self._r, self._mu, ap, at, delta_rp)

        # Compute the correlation function
        xi_asy = PktoXi_obj.pk_to_xi_asymmetry(rescaled_r, rescaled_mu, pk, params)

        return xi_asy

    def compute_desi_instrumental_systematics(self, params, bin_size_rp):
        """Compute DESI instrumental systematics model
        TODO add link to Satya's paper describing this

        Parameters
        ----------
        params : dict
            Computation parameters
        bin_size_rp : float
            Bin size along the line-of-sight

        Returns
        -------
        1D Array
            Output correction
        """
        if self._tracer1['type'] != self._tracer2['type']:
            raise ValueError('DESI instrumental systematics model only applies '
                             'to auto-correlation functions.')

        rp = self._r * self._mu
        rt = self._r * np.sqrt(1 - self._mu**2)

        # b = 0.0003189935987295203
        b = params.get('desi_inst_sys_amp', 0.0003189935987295203)

        w = (rp > 0) & (rp < bin_size_rp)
        correction = np.zeros(rt.shape)

        if self.desi_instrumental_systematics_interp is None:

            # See in the cvs table directory the code to generate the table.
            # This is the correlation function induced by the sky model white noise.
            path = "instrumental_systematics/desi-instrument-syst-for-forest-auto-correlation.csv"
            table_filename = utils.find_file(path)
            print("Reading desi_instrumental_systematics table", table_filename)
            syst_table = Table.read(table_filename)
            self.desi_instrumental_systematics_interp = interp1d(
                syst_table["RT"], syst_table["XI"], kind='linear')

        correction[w] = b * self.desi_instrumental_systematics_interp(rt[w])

        return correction