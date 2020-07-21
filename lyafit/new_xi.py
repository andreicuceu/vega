import numpy as np
import copy
from . import new_utils as utils


class CorrelationFunction:
    """Correlation function computation and handling

    # ! Slow operations should be kept in init as that is only called once
    # ! Compute is called many times and should be fast
    # * Extensions should have their separate method which is
    # * called from init/compute
    """
    def __init__(self, config, r, mu, z, tracer1, tracer2):

        self._config = config
        self._r = r
        self._mu = mu
        self._z = z
        self._ell_max = config['ell_max']
        self._tracer1 = tracer1
        self._tracer2 = tracer2

        # Check if we need delta rp
        self._delta_rp_name = None
        if tracer1['type'] == 'discrete':
            self._delta_rp_name = 'drp_'+tracer1['name']
        elif tracer2['type'] == 'discrete':
            self._delta_rp_name = 'drp_'+tracer2['name']

        # Precompute growth
        self.xi_growth = self.growth_factor()

        pass

    def compute(self, k, muk, dmuk, pk, params):
        self._params = params

        # Check for delta rp
        delta_rp = 0.
        if self._delta_rp_name is not None:
            delta_rp = params[self._delta_rp_name]

        # Get rescaled Xi coordinates
        ap, at = utils.cosmo_fit_func(params)
        rescaled_r, rescaled_mu = self.rescale_coords(self._r, self._mu,
                                                      ap, at, delta_rp)

        # Compute correlation function
        xi_full = utils.Pk2Xi(rescaled_r, rescaled_mu, k, pk, muk, dmuk, self._ell_max)
        self.xi_base = copy.deepcopy(xi_full)

        # Add bias evolution
        tracer1_z_evol = 'z evol {}'.format(self._tracer1['name'])
        tracer2_z_evol = 'z evol {}'.format(self._tracer2['name'])
        if 'croom' in tracer1_z_evol:
            xi_full *= self.bias_evol_croom(self._tracer1['name'])
        else:
            xi_full *= self.bias_evol_std(self._tracer1['name'])
        if 'croom' in tracer2_z_evol:
            xi_full *= self.bias_evol_croom(self._tracer2['name'])
        else:
            xi_full *= self.bias_evol_std(self._tracer2['name'])
        self.xi_evol = copy.deepcopy(xi_full)

        # Add growth
        xi_full *= self.xi_growth

        return xi_full

    @staticmethod
    def rescale_coords(r, mu, ap, at, delta_rp=0.):
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

    def bias_evol_std(self, tracer_name):
        """Bias evolution standard model

        Parameters
        ----------
        tracer_name : string
            Tracer name

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        z_eff = self._config['zeff']
        p0 = self._params['alpha_{}'.format(tracer_name)]
        bias_z = ((1. + self._z) / (1 + z_eff))**p0 
        return bias_z

    def bias_evol_croom(self, tracer_name):
        """Bias evolution Croom model for QSO
        See Croom et al. 2005

        Parameters
        ----------
        tracer_name : string
            Tracer name

        Returns
        -------
        ND Array
            Bias evolution for tracer
        """
        assert tracer_name == "QSO"
        z_eff = self._config['zeff']
        p0 = self._params["croom_par0"]
        p1 = self._params["croom_par1"]
        bias_z = (p0 + p1*(1. + self._z)**2) / (p0 + p1 * (1 + z_eff)**2)
        return bias_z

    def growth_factor(self):
        """Compute growth factor
        Implements eq. 7.77 from S. Dodelson's Modern Cosmology book

        Returns
        -------
        ND Array
            Growth factor
        """
        z_fid = self._config['zfid']
        Omega_m = self._config.get('Omega_m', None)
        Omega_de = self._config.get('Omega_de', None)
        # Check if we have dark energy
        if Omega_de is None:
            growth = (1 + z_fid)/(1. + self._z)
            return growth**2

        # Compute growth at each point in the cf
        if isinstance(self._z, float):
            growth = utils.growth_function(self._z, Omega_m, Omega_de)
        else:
            assert self._z.ndim == 1
            growth = np.zeros(len(self._z))
            for i, z in enumerate(self._z):
                growth[i] = utils.growth_function(z, Omega_m, Omega_de)

        # Scale to fiducial redshift
        growth = growth / utils.growth_function(z_fid, Omega_m, Omega_de)

        # args = (Omega_m, Omega_de)
        # Compute growth for each redshift
        # def growth_func(z):
        #     a = 1 / (1 + z)
        #     growth_int = quad(utils.growth_integrand, 0, a, args=args)[0]
        #     hubble = utils.hubble(z, Omega_m, Omega_de)
        #     return 5./2. * Omega_m * hubble * growth_int
            # a = 1 / (1 + z)
            # growth_int = quad(utils.growth_integrand, 0, a, args=args)[0]
            # hubble = utils.hubble(z, Omega_m, Omega_de)

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
