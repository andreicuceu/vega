import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from . import utils


class CorrelationFunction:
    """Correlation function computation and handling.

    # ! Slow operations should be kept in init as that is only called once

    # ! Compute is called many times and should be fast

    Extensions should have their separate method of the form
    'compute_extension' that can be called from outside
    """
    def __init__(self, config, fiducial, coords_grid, scale_params,
                 tracer1, tracer2, bb_config=None, metal_corr=False):
        """

        Parameters
        ----------
        config : ConfigParser
            model section of config file
        fiducial : dict
            fiducial config
        coords_grid : dict
            Dictionary with coordinate grid - r, mu, z
        scale_params : ScaleParameters
            ScaleParameters object
        tracer1 : dict
            Config of tracer 1
        tracer2 : dict
            Config of tracer 2
        bb_config : list, optional
            list with configs of broadband terms, by default None
        metal_corr : bool, optional
            Whether this is a metal correlation, by default False
        """
        self._config = config
        self._r = coords_grid['r']
        self._mu = coords_grid['mu']
        self._z = coords_grid['z']
        self._multipole = config.getint('single_multipole', -1)
        self._tracer1 = tracer1
        self._tracer2 = tracer2
        self._z_eff = fiducial['z_eff']
        self._rel_z_evol = (1. + self._z) / (1 + self._z_eff)
        self._scale_params = scale_params
        self._metal_corr = metal_corr

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
            self.xi_growth = self.compute_growth(self._z, self._z_fid, self._Omega_m,
                                                 self._Omega_de)
        else:
            self.xi_growth = self.compute_growth_old(self._z, self._z_fid, self._Omega_m,
                                                     self._Omega_de)

        # Initialize the broadband
        self.has_bb = False
        if bb_config is not None:
            self._init_broadband(bb_config)
            self.has_bb = True

        # Check for QSO radiation modeling and check if it is QSOxLYA
        # Does this work for the QSO auto as well?
        self.radiation_flag = False
        if 'radiation effects' in self._config:
            self.radiation_flag = self._config.getboolean('radiation effects')
            if self.radiation_flag:
                names = [self._tracer1['name'], self._tracer2['name']]
                if not ('QSO' in names and 'LYA' in names):
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

    def _init_broadband(self, bb_config):
        """Initialize the broadband terms.

        Parameters
        ----------
        bb_config : list
            list with configs of broadband terms
        """
        self.bb_terms = {}
        self.bb_terms['pre-add'] = []
        self.bb_terms['post-add'] = []
        self.bb_terms['pre-mul'] = []
        self.bb_terms['post-mul'] = []

        # First pick up the normal broadband terms
        normal_broadbands = [el for el in bb_config
                             if el['func'] != 'broadband_sky']
        for index, config in enumerate(normal_broadbands):
            # Create the name for the parameters of this term
            name = 'BB-{}-{} {} {} {}'.format(config['cf_name'], index,
                                              config['type'], config['pre'],
                                              config['rp_rt'])

            # Create the broadband term dictionary
            bb = {}
            bb['name'] = name
            bb['func'] = config['func']
            bb['rp_rt'] = config['rp_rt']
            bb['r_config'] = config['r_config']
            bb['mu_config'] = config['mu_config']
            self.bb_terms[config['pre'] + "-" + config['type']].append(bb)

        # Next pick up the sky broadban terms
        sky_broadbands = [el for el in bb_config
                          if el['func'] == 'broadband_sky']
        for index, config in enumerate(sky_broadbands):
            assert config['rp_rt'] == 'rp,rt'
            # Create the name for the parameters of this term
            name = 'BB-{}-{}-{}'.format(config['cf_name'],
                                        index + len(normal_broadbands),
                                        config['func'])

            # Create the broadband term dictionary
            bb = {}
            bb['name'] = name
            bb['func'] = config['func']
            bb['bin_size_rp'] = config['bin_size_rp']
            self.bb_terms[config['pre'] + "-" + config['type']].append(bb)

    def compute_broadband(self, params, pos_type):
        """Compute the broadband terms for
        one position (pre-distortion/post-distortion)
        and one type (multiplicative/additive).

        Parameters
        ----------
        params : dict
            Computation parameters
        pos_type : string
            String with position and type, must be one of:
            'pre-mul' or 'pre-add' or 'post-mul' or 'post-add'

        Returns
        -------
        1d Array
            Output broadband
        """
        assert pos_type in ['pre-mul', 'pre-add', 'post-mul', 'post-add']
        corr = None

        # Loop over the right pos/type configuration
        for bb_term in self.bb_terms[pos_type]:
            # Check if it's sky or normal broadband
            if bb_term['func'] != 'broadband_sky':
                # Initialize the broadband and check
                # if we need to add or multiply
                if corr is None:
                    corr = self.broadband(bb_term, params)
                    if 'mul' in pos_type:
                        corr = 1 + corr
                elif 'mul' in pos_type:
                    corr *= 1 + self.broadband(bb_term, params)
                else:
                    corr += self.broadband(bb_term, params)
            else:
                # Initialize the broadband and check
                # if we need to add or multiply
                if corr is None:
                    corr = self.broadband_sky(bb_term, params)
                    if 'mul' in pos_type:
                        corr = 1 + corr
                elif 'mul' in pos_type:
                    corr *= 1 + self.broadband_sky(bb_term, params)
                else:
                    corr += self.broadband_sky(bb_term, params)

        # Give defaults if corr is still None
        if corr is None:
            if 'mul' in pos_type:
                corr = 1.
            else:
                corr = 0.
        return corr

    def broadband_sky(self, bb_term, params):
        """Compute sky broadband term.

        Calculates a Gaussian broadband in rp,rt for the sky residuals.

        Parameters
        ----------
        bb_term : dict
            broadband term config
        params : dict
            Computation parameters

        Returns
        -------
        1d Array
            Output broadband
        """
        rp = self._r * self._mu
        rt = self._r * np.sqrt(1 - self._mu**2)
        scale = params[bb_term['name'] + '-scale-sky']
        sigma = params[bb_term['name'] + '-sigma-sky']

        corr = scale / (sigma * np.sqrt(2. * np.pi))
        corr *= np.exp(-0.5 * (rt / sigma)**2)
        w = (rp >= 0.) & (rp < bb_term['bin_size_rp'])
        corr[~w] = 0.

        return corr

    def broadband(self, bb_term, params):
        """Compute broadband term.

        Calculates a power-law broadband in r and mu or rp,rt.

        Parameters
        ----------
        bb_term : dict
            broadband term config
        params : dict
            Computation parameters

        Returns
        -------
        1d Array
            Output broadband
        """
        r1 = self._r / 100.
        r2 = self._mu
        if bb_term['rp_rt'] == 'rp,rt':
            r1 = self._r / 100. * self._mu
            r2 = self._r / 100. * np.sqrt(1 - self._mu**2)

        r_min, r_max, dr = bb_term['r_config']
        mu_min, mu_max, dmu = bb_term['mu_config']
        r1_powers = np.arange(r_min, r_max + 1, dr)
        r2_powers = np.arange(mu_min, mu_max + 1, dmu)

        bb_params = []
        for i in r1_powers:
            for j in r2_powers:
                bb_params.append(params['{} ({},{})'.format(
                        bb_term['name'], i, j)])
        bb_params = np.array(bb_params).reshape(-1, r_max - r_min + 1)

        corr = (bb_params[:, :, None, None] * r1**r1_powers[:, None, None] *
                r2**r2_powers[None, :, None]).sum(axis=(0, 1, 2))
        return corr

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
