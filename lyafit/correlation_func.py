import numpy as np
from . import utils


class CorrelationFunction:
    """Correlation function computation and handling

    # ! Slow operations should be kept in init as that is only called once
    # ! Compute is called many times and should be fast
    # * Extensions should have their separate method of the form
    # * 'compute_extension' that can be called from outside
    """
    def __init__(self, config, fiducial, coords_grid, tracer1, tracer2):
        """Initialize the config, coordinate grid, tracer info,
        and everything else needed to compute the correlation function

        Parameters
        ----------
        config : ConfigParser
            model section of config file
        fiducial : dict
            fiducial config
        coords_grid : dict
            Dictionary with coordinate grid - r, mu, z
        tracer1 : dict
            Config of tracer 1
        tracer2 : dict
            Config of tracer 2
        """
        self._config = config
        self._r = coords_grid['r']
        self._mu = coords_grid['mu']
        self._z = coords_grid['z']
        self._ell_max = config.getint('ell_max', 6)
        self._tracer1 = tracer1
        self._tracer2 = tracer2
        self._z_eff = fiducial['z_eff']

        # Check if we need delta rp
        self._delta_rp_name = None
        if tracer1['type'] == 'discrete':
            self._delta_rp_name = 'drp_'+tracer1['name']
        elif tracer2['type'] == 'discrete':
            self._delta_rp_name = 'drp_'+tracer2['name']

        # Precompute growth
        self._z_fid = fiducial['z_fiducial']
        self._Omega_m = fiducial['Omega_m']
        self._Omega_de = fiducial['Omega_de']
        self.xi_growth = self.compute_growth(self._z, self._z_fid,
                                             self._Omega_m, self._Omega_de)

    def compute(self, k, muk, pk, params):
        """Compute correlation function for input P(k)

        Parameters
        ----------
        k : 1D Array
            Wavenumber grid of power spectrum
        muk : ND Array
            k_parallel / k
        pk : ND Array
            Power spectrum
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            Output correlation function
        """
        # Compute the core
        xi = self.compute_core(k, muk, pk, params)

        # Add bias evolution
        xi *= self.compute_bias_evol(params)

        # Add growth
        xi *= self.xi_growth

        return xi

    def compute_core(self, k, muk, pk, params):
        """Compute the core of the correlation function
        This does the Hankel transform of the input P(k),
        sums the necessary multipoles and rescales the coordinates

        Parameters
        ----------
        k : 1D Array
            Wavenumber grid of power spectrum
        muk : ND Array
            k_parallel / k
        pk : ND Array
            Power spectrum
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
            delta_rp = params[self._delta_rp_name]

        # Get rescaled Xi coordinates
        ap, at = utils.cosmo_fit_func(params)
        rescaled_r, rescaled_mu = self._rescale_coords(self._r, self._mu,
                                                       ap, at, delta_rp)

        # Compute correlation function
        xi = utils.Pk2Xi(rescaled_r, rescaled_mu, k, pk, muk, self._ell_max)

        return xi

    @staticmethod
    def _rescale_coords(r, mu, ap, at, delta_rp=0.):
        """Rescale Xi coordinates using ap/at

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
        rp = r * mu + delta_rp
        rt = r * np.sqrt(1 - mu**2)
        rescaled_rp = ap * rp
        rescaled_rt = at * rt
        rescaled_r = np.sqrt(rescaled_rp**2 + rescaled_rt**2)
        rescaled_mu = rescaled_rp / rescaled_r

        return rescaled_r, rescaled_mu

    def compute_bias_evol(self, params):
        """Compute bias evolution for the correlation function

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
        """Compute tracer bias evolution

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
            evol_model = self._config.get(handle_name)
        else:
            evol_model = self._config.get('z evol')

        # Compute the bias evolution using the right model
        if 'croom' in evol_model:
            bias_evol = self._bias_evol_croom(params, tracer_name)
        else:
            bias_evol = self._bias_evol_std(params, tracer_name)

        return bias_evol

    def _bias_evol_std(self, params, tracer_name):
        """Bias evolution standard model

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
        bias_z = ((1. + self._z) / (1 + self._z_eff))**p0
        return bias_z

    def _bias_evol_croom(self, params, tracer_name):
        """Bias evolution Croom model for QSO
        See Croom et al. 2005

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
        """Compute growth factor
        Implements eq. 7.77 from S. Dodelson's Modern Cosmology book

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

        # Check if z_grid is a float - the cf is approximated at z eff
        if isinstance(z_grid, float):
            growth = utils.growth_function(z_grid, Omega_m, Omega_de)
        else:
            # If it's a grid it should be 1D
            assert z_grid.ndim == 1
            growth = np.zeros(len(z_grid))
            # Compute the growth at each redshift on the grid
            for i, z in enumerate(z_grid):
                growth[i] = utils.growth_function(z, Omega_m, Omega_de)

        # Scale to the fiducial redshift
        growth = growth / utils.growth_function(z_fid, Omega_m, Omega_de)

        return growth**2


# ### QSO radiation model
# def xi_qso_radiation(r, mu, tracer1, tracer2, **pars):
#     assert (tracer1['name']=="QSO" or tracer2['name']=="QSO") and (tracer1['name']!=tracer2['name'])

#     if tracer1['type']=='discrete':
#         drp = pars['drp_'+tracer1['name']]
#     elif tracer2['type']=='discrete':
#         drp = pars['drp_'+tracer2['name']]
#     rp = r*mu + drp
#     rt = r*sp.sqrt(1-mu**2)
#     r_shift = sp.sqrt(rp**2.+rt**2.)
#     mu_shift = rp/r_shift

#     xi_rad = pars["qso_rad_strength"]/(r_shift**2.)
#     xi_rad *= 1.-pars["qso_rad_asymmetry"]*(1.-mu_shift**2.)
#     xi_rad *= sp.exp(-r_shift*( (1.+mu_shift)/pars["qso_rad_lifetime"] + 1./pars["qso_rad_decrease"]) )

#     return xi_rad

# def xi_relativistic(r, mu, k, pk_lin, tracer1, tracer2, **pars):
#     """Calculate the cross-correlation contribution from relativistic effects (Bonvin et al. 2014).

#     Args:
#         r (float): r coordinates
#         mu (float): mu coordinates
#         k (float): wavenumbers
#         pk_lin (float): linear matter power spectrum
#         tracer1: dictionary of tracer1
#         tracer2: dictionary of tracer2
#         pars: dictionary of fit parameters

#     Returns:
#         sum of dipole and octupole correlation terms (float)

#     """
#     assert (tracer1['type']=="continuous" or tracer2['type']=="continuous") and (tracer1['type']!=tracer2['type'])

#     if tracer1['type']=='discrete':
#         drp = pars['drp_'+tracer1['name']]
#     elif tracer2['type']=='discrete':
#         drp = pars['drp_'+tracer2['name']]

#     ap, at = utils.cosmo_fit_func(pars)
#     rp = r*mu + drp
#     rt = r*sp.sqrt(1-mu**2)
#     arp = ap*rp
#     art = at*rt
#     ar = sp.sqrt(arp**2+art**2)
#     amu = arp/ar

#     xi_rel = utils.Pk2XiRel(ar, amu, k, pk_lin, pars)
#     return xi_rel

# def xi_asymmetry(r, mu, k, pk_lin, tracer1, tracer2, **pars):
#     """Calculate the cross-correlation contribution from standard asymmetry (Bonvin et al. 2014).

#     Args:
#         r (float): r coordinates
#         mu (float): mu coordinates
#         k (float): wavenumbers
#         pk_lin (float): linear matter power spectrum
#         tracer1: dictionary of tracer1
#         tracer2: dictionary of tracer2
#         pars: dictionary of fit parameters

#     Returns:
#         sum of dipole and octupole correlation terms (float)

#     """
#     assert (tracer1['type']=="continuous" or tracer2['type']=="continuous") and (tracer1['type']!=tracer2['type'])

#     if tracer1['type']=='discrete':
#         drp = pars['drp_'+tracer1['name']]
#     elif tracer2['type']=='discrete':
#         drp = pars['drp_'+tracer2['name']]

#     ap, at = utils.cosmo_fit_func(pars)
#     rp = r*mu + drp
#     rt = r*sp.sqrt(1-mu**2)
#     arp = ap*rp
#     art = at*rt
#     ar = sp.sqrt(arp**2+art**2)
#     amu = arp/ar

#     xi_asy = utils.Pk2XiAsy(ar, amu, k, pk_lin, pars)
#     return xi_asy

# def broadband_sky(r, mu, name=None, bin_size_rp=None, *pars, **kwargs):
#     '''
#         Broadband function interface.
#         Calculates a Gaussian broadband in rp,rt for the sky residuals
#     Arguments:
#         - r,mu (array or float): where to calcualte the broadband
#         - bin_size_rp (array): Bin size of the distortion matrix along the line-of-sight
#         - name: (string) name ot identify the corresponding parameters,
#                     typically the dataset name and whether it's multiplicative
#                     of additive
#         - *pars: additional parameters that are ignored (for convenience)
#         **kwargs (dict): dictionary containing all the polynomial
#                     coefficients. Any extra keywords are ignored
#     Returns:
#         - cor (array of float): Correlation function
#     '''

#     rp = r*mu
#     rt = r*sp.sqrt(1-mu**2)
#     cor = kwargs[name+'-scale-sky']/(kwargs[name+'-sigma-sky']*sp.sqrt(2.*sp.pi))*sp.exp(-0.5*(rt/kwargs[name+'-sigma-sky'])**2)
#     w = (rp>=0.) & (rp<bin_size_rp)
#     cor[~w] = 0.

#     return cor

# def broadband(r, mu, deg_r_min=None, deg_r_max=None,
#         ddeg_r=None, deg_mu_min=None, deg_mu_max=None,
#         ddeg_mu=None, deg_mu=None, name=None,
#         rp_rt=False, bin_size_rp=None, *pars, **kwargs):
#     '''
#     Broadband function interface.
#     Calculates a power-law broadband in r and mu or rp,rt
#     Arguments:
#         - r,mu: (array or float) where to calcualte the broadband
#         - deg_r_min: (int) degree of the lowest-degree monomial in r or rp
#         - deg_r_max: (int) degree of the highest-degree monomual in r or rp
#         - ddeg_r: (int) degree step in r or rp
#         - deg_mu_min: (int) degree of the lowest-degree monomial in mu or rt
#         - deg_mu_max: (int) degree of the highest-degree monmial in mu or rt
#         - ddeg_mu: (int) degree step in mu or rt
#         - name: (string) name ot identify the corresponding parameters,
#                     typically the dataset name and whether it's multiplicative
#                     of additive
#         - rt_rp: (bool) use r,mu (if False) or rp,rt (if True)
#         - *pars: additional parameters that are ignored (for convenience)
#         **kwargs: (dict) dictionary containing all the polynomial
#                     coefficients. Any extra keywords are ignored
#     '''

#     r1 = r/100
#     r2 = mu
#     if rp_rt:
#         r1 = (r/100)*mu
#         r2 = (r/100)*sp.sqrt(1-mu**2)

#     r1_pows = sp.arange(deg_r_min, deg_r_max+1, ddeg_r)
#     r2_pows = sp.arange(deg_mu_min, deg_mu_max+1, ddeg_mu)
#     BB = [kwargs['{} ({},{})'.format(name,i,j)] for i in r1_pows
#             for j in r2_pows]
#     BB = sp.array(BB).reshape(-1,deg_r_max-deg_r_min+1)

#     return (BB[:,:,None,None]*r1**r1_pows[:,None,None]*\
#             r2**r2_pows[None,:,None]).sum(axis=(0,1,2))
