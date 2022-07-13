import numpy as np
import copy
from pkg_resources import resource_filename
from numba import njit, float64
from . import utils


class PowerSpectrum:
    """Power Spectrum computation and handling.

    # ! Slow operations should be kept in init as that is only called once

    # ! Compute is called many times and should be fast

    Extensions should have their separate method of the form
    'compute_extension' that can be called from outside
    """
    def __init__(self, config, fiducial, tracer1, tracer2, dataset_name=None):
        """

        Parameters
        ----------
        config : dict
            pk config object
        fiducial : dict
            fiducial config
        tracer1 : dict
            Config of tracer 1
        tracer2 : dict
            Config of tracer 1
        dataset_name : string
            Name of dataset, by default None
        """
        self._config = config
        self.tracer1_name = copy.deepcopy(tracer1['name'])
        self.tracer2_name = copy.deepcopy(tracer2['name'])
        self.tracer1_type = copy.deepcopy(tracer1['type'])
        self.tracer2_type = copy.deepcopy(tracer2['type'])

        self._name = dataset_name
        self.k_grid = fiducial['k']
        self._bin_size_rp = config.getfloat('bin_size_rp')
        self._bin_size_rt = config.getfloat('bin_size_rt')
        self.use_Gk = self._config.getboolean('model binning', True)

        # Check for the old config
        pk_model = self._config.get('model-pk', None)

        # Get the HCD model and check for UV
        self.hcd_model = None
        self._add_uv = False
        if pk_model is None:
            # Get the new config
            self.hcd_model = self._config.get('model-hcd', None)
            self._add_uv = self._config.getboolean('add uv', False)
        else:
            # Get the old config
            if 'hcd' in pk_model:
                self.hcd_model = pk_model
            if 'uv' in pk_model:
                self._add_uv = True

        # Check if we need fvoigt model
        self._Fvoigt_data = None
        if 'fvoigt_model' in self._config.keys():
            fvoigt_model = self._config.get('fvoigt_model')
            if '/' not in fvoigt_model:
                path = '{}/fvoigt_models/Fvoigt_{}.txt'.format(
                                resource_filename('vega', 'models'),
                                fvoigt_model)
            else:
                path = fvoigt_model
            self._Fvoigt_data = np.loadtxt(path)

        # Initialize some stuff we need
        self.pk_Gk = None
        self._pk_fid = fiducial['pk_full'] * ((1 + fiducial['z_fiducial'])
                                              / (1. + fiducial['z_eff']))**2

        # Initialize the mu_k grid
        num_bins_muk = self._config.getint('num_bins_muk', 1000)
        muk_grid = (np.arange(num_bins_muk) + 0.5) / num_bins_muk
        self.muk_grid = muk_grid[:, None]

        self.k_par_grid = self.k_grid * self.muk_grid
        self.k_trans_grid = self.k_grid * np.sqrt(1 - self.muk_grid**2)
        self._arinyo_pars = None
        self._peak_nl_pars = None
        self._L0_hcd_cache = None
        self._F_hcd_cache = None

    def compute(self, pk_lin, params, fast_metals=False):
        """Computes a power spectrum for the tracers using the input
        linear P(k) and parameters.

        Parameters
        ----------
        pk_lin : 1D array
            Linear Power Spectrum
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Power spectrum
        """
        # params = params

        # Get the core biases and betas
        bias_beta = utils.bias_beta(params, self.tracer1_name, self.tracer2_name)
        bias1, beta1, bias2, beta2 = bias_beta

        # Add UV model
        if self._add_uv:
            if self.tracer1_name == 'LYA':
                bias1, beta1 = self.compute_bias_beta_uv(bias1, beta1, params)
            if self.tracer2_name == 'LYA':
                bias2, beta2 = self.compute_bias_beta_uv(bias2, beta2, params)

        # Add HCD model
        if self.hcd_model is not None:
            if self.tracer1_name == 'LYA':
                bias1, beta1 = self.compute_bias_beta_hcd(bias1, beta1, params)
            if self.tracer2_name == 'LYA':
                bias2, beta2 = self.compute_bias_beta_hcd(bias2, beta2, params)

        # Compute kaiser model
        pk_full = pk_lin * self.compute_kaiser(bias1, beta1, bias2, beta2, fast_metals)

        # add non linear small scales
        if 'small scale nl' in self._config.keys():
            if 'arinyo' in self._config.get('small scale nl'):
                pk_full *= self.compute_dnl_arinyo(params)
            elif 'mcdonald' in self._config.get('small scale nl'):
                pk_full *= self.compute_dnl_mcdonald()
            else:
                print('small scale nl: must be either mcdonald or arinyo')
                raise ValueError('Incorrect \'small scale nl\' specified')

        # model the effect of binning
        if self.use_Gk:
            if self.pk_Gk is None:
                self.pk_Gk = self.compute_Gk(params)
            pk_full *= self.pk_Gk

        # add non linear large scales
        if params['peak']:
            pk_full *= self.compute_peak_nl(params)

        # add full shape smoothing
        if 'fullshape smoothing' in self._config:
            smoothing_type = self._config.get('fullshape smoothing')
            if 'gauss' in smoothing_type:
                pk_full *= self.compute_fullshape_gauss_smoothing(params)
            elif 'exp' in smoothing_type:
                pk_full *= self.compute_fullshape_exp_smoothing(params)
            else:
                raise ValueError('"fullshape smoothing" must be of type'
                                 ' "gauss" or "exp".')

        # add velocity dispersion
        if 'velocity dispersion' in self._config:
            smoothing_type = self._config.get('velocity dispersion')
            if 'gauss' in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_gauss(params)
            elif 'lorentz' in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_lorentz(params)
            elif 'lorentz_gauss' in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_lorentz(params)
                pk_full *= self.compute_velocity_dispersion_gauss(params)
            else:
                raise ValueError('"velocity dispersion" must be of type'
                                 ' "gauss" or "lorentz".')

        return pk_full

    def compute_kaiser(self, bias1, beta1, bias2, beta2, fast_metals=False):
        """Compute Kaiser model.

        Parameters
        ----------
        bias1 : float
            Bias for tracer 1
        beta1 : float
            Beta for tracer 1
        bias2 : float
            Bias for tracer 2
        beta2 : float
            Beta for tracer 2

        Returns
        -------
        ND Array
            Kaiser term
        """
        pk = (1 + beta1 * self.muk_grid**2)
        pk = pk * (1 + beta2 * self.muk_grid**2)

        if not fast_metals:
            pk *= bias1 * bias2
        return pk

    def compute_bias_beta_uv(self, bias, beta, params):
        """ Compute effective biases that include UV modeling.

        Parameters
        ----------
        bias : float
            Bias for tracer
        beta : float
            Beta for tracer
        params : dict
            Computation parameters

        Returns
        -------
        (float, float)
            Effective bias and beta
        """
        bias_gamma = params["bias_gamma"]
        bias_prim = params["bias_prim"]
        lambda_uv = params["lambda_uv"]

        W = np.arctan(self.k_grid * lambda_uv) / (self.k_grid * lambda_uv)
        beta_eff = beta / (1 + bias_gamma / bias * W / (1 + bias_prim * W))
        bias_eff = bias + bias_gamma * W / (1 + bias_prim * W)

        return bias_eff, beta_eff

    def compute_bias_beta_hcd(self, bias, beta, params):
        """ Compute effective biases that include HCD modeling.

        Parameters
        ----------
        bias : float
            Bias for tracer
        beta : float
            Beta for tracer
        params : dict
            Computation parameters

        Returns
        -------
        (float, float)
            Effective bias and beta
        """
        # Check if we have an HCD bias for each component
        hcd_bias_name = "bias_hcd_{}".format(self._name)
        bias_hcd = params.get(hcd_bias_name, None)
        if bias_hcd is None:
            bias_hcd = params['bias_hcd']

        # Get the other parameters
        beta_hcd = params["beta_hcd"]
        L0_hcd = params["L0_hcd"]

        # TODO Maybe improve the names we search for
        # Check which model we need
        if L0_hcd != self._L0_hcd_cache or self._F_hcd is None:
            if 'Rogers' in self.hcd_model:
                self._F_hcd = self._hcd_Rogers2018(L0_hcd, self.k_par_grid)
            elif 'mask' in self.hcd_model:
                assert self._Fvoigt_data is not None
                self._F_hcd = self._hcd_no_mask(L0_hcd)
            elif 'sinc' in self.hcd_model:
                self._F_hcd = self._hcd_sinc(L0_hcd)

            self._L0_hcd_cache = L0_hcd

        bias_eff = bias + bias_hcd * self._F_hcd
        beta_eff = (bias * beta + bias_hcd * beta_hcd * self._F_hcd)
        beta_eff /= (bias + bias_hcd * self._F_hcd)

        return bias_eff, beta_eff

    def _hcd_sinc(self, L0):
        """HCD sinc model.

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        return utils.sinc(self.k_par_grid * L0)

    @staticmethod
    @njit(float64[:, :](float64, float64[:, :]))
    def _hcd_Rogers2018(L0, k_par_grid):
        """Model the effect of HCD systems with the Fourier transform
        of a Lorentzian profile. Motivated by Rogers et al. (2018).

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        f_hcd = np.exp(-L0 * k_par_grid)
        return f_hcd

    def _hcd_no_mask(self, L0):
        """Use Fvoigt function to fit the DLA in the autocorrelation Lyman-alpha
        without masking them ! (L0 = 1)

        (If you want to mask them use Fvoigt_exp.txt and L0 = 10 as eBOSS DR14)

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        ND Array
            F_hcd
        """
        k_data = self._Fvoigt_data[:, 0]
        F_data = self._Fvoigt_data[:, 1]

        if self.tracer1_name == self.tracer2_name:
            F_hcd = np.interp(L0 * self.k_par_grid, k_data, F_data,
                              left=0, right=0)
        else:
            F_hcd = np.interp(L0 * self.k_par_grid, k_data, F_data)

        return F_hcd

    def compute_peak_nl(self, params):
        """Compute the non-linear gaussian correction for the peak component.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor for the peak
        """
        sigma_par = params.get('sigmaNL_par', None)
        sigma_trans = params.get('sigmaNL_per', None)
        growth_rate = params.get('growth_rate')

        if sigma_par is None and sigma_trans is not None:
            sigma_par = sigma_trans * (1 + growth_rate)
        elif sigma_trans is None and sigma_par is not None:
            sigma_trans = sigma_par / (1 + growth_rate)
        elif sigma_par is None and sigma_trans is None:
            raise ValueError('No parameters for peak NL found.'
                             ' Add sigmaNL_par and/or sigmaNL_par.')

        if self._peak_nl_pars is None:
            self._peak_nl_pars = np.array([sigma_par, sigma_trans]) + 1

        if not np.allclose(np.array([sigma_par, sigma_trans]),
                           self._peak_nl_pars):
            peak_nl = self.k_par_grid**2 * sigma_par**2
            peak_nl += self.k_trans_grid**2 * sigma_trans**2
            self._peak_nl_pars = np.array([sigma_par, sigma_trans])
            self._peak_nl_cache = np.exp(-peak_nl / 2)

        return self._peak_nl_cache

    def compute_dnl_mcdonald(self):
        """Non linear term from McDonald 2003.

        Returns
        -------
        ND Array
            D_NL factor
        """
        assert self.tracer1_name == "LYA"
        assert self.tracer2_name == "LYA"

        kvel = 1.22 * (1 + self.k_grid / 0.923)**0.451
        dnl = (self.k_grid / 6.4)**0.569 - (self.k_grid / 15.3)**2.01
        dnl = dnl - (self.k_grid * self.muk_grid / kvel)**1.5
        return np.exp(dnl)

    def compute_dnl_arinyo(self, params):
        """Non linear term from Arinyo et al 2015.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            D_NL factor
        """
        two_lya_flag = "LY" in self.tracer1_name and "LY" in self.tracer2_name
        one_lya_flag = "LY" in self.tracer1_name or "LY" in self.tracer2_name

        q1 = params["dnl_arinyo_q1"]
        kv = params["dnl_arinyo_kv"]
        av = params["dnl_arinyo_av"]
        bv = params["dnl_arinyo_bv"]
        kp = params["dnl_arinyo_kp"]

        if self._arinyo_pars is None:
            self._arinyo_pars = np.array([q1, kv, av, bv, kp]) + 1
        if not np.allclose(np.array([q1, kv, av, bv, kp]), self._arinyo_pars):
            growth = q1 * self.k_grid**3 * self._pk_fid / (2 * np.pi**2)
            pec_velocity = (self.k_grid / kv)**av * np.abs(self.muk_grid)**bv
            pressure = (self.k_grid / kp) * (self.k_grid / kp)
            dnl = np.exp(growth * (1 - pec_velocity) - pressure)

            self._arinyo_pars = np.array([q1, kv, av, bv, kp])
            if two_lya_flag:
                self._arinyo_dnl_cache = dnl
            elif one_lya_flag:
                self._arinyo_dnl_cache = np.sqrt(dnl)
            else:
                return np.ones(dnl.shape)
                # raise ValueError("Arinyo NL term called for correlation with no Lyman-alpha")

        return self._arinyo_dnl_cache

    def compute_Gk(self, params):
        """Model the effect of binning of the cf.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            G(k)
        """
        bin_size_rp = params.get("par binsize {}".format(self._name), self._bin_size_rp)
        bin_size_rt = params.get("per binsize {}".format(self._name), self._bin_size_rt)

        Gk = 1.
        if bin_size_rp != 0:
            Gk = Gk * utils.sinc(self.k_par_grid * bin_size_rp / 2)
        if bin_size_rt != 0:
            Gk = Gk * utils.sinc(self.k_trans_grid * bin_size_rt / 2)
        return Gk

    def compute_fullshape_gauss_smoothing(self, params):
        """Compute a Gaussian smoothing for the full correlation function.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        sigma_par = params.get('par_sigma_smooth', None)
        sigma_trans = params.get('per_sigma_smooth', None)

        if sigma_par is None and sigma_trans is None:
            raise ValueError('Asked for fullshape gaussian smoothing without setting the'
                             ' smoothing parameters (par_sigma_smooth and/or per_sigma_smooth).')
        elif sigma_par is None:
            sigma_par = sigma_trans
        elif sigma_trans is None:
            sigma_trans = sigma_par

        gauss_smoothing = self.k_par_grid**2 * sigma_par**2
        gauss_smoothing += self.k_trans_grid**2 * sigma_trans**2
        return np.exp(-gauss_smoothing / 2)**2

    def compute_fullshape_exp_smoothing(self, params):
        """ Compute a Gaussian and exp smoothing for the full
        correlation function (usefull for london_mocks_v6.0).

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        # Get the parameters
        sigma_par_sq = params['par_sigma_smooth']**2
        sigma_trans_sq = params['per_sigma_smooth']**2
        exp_par_sq = params['par_exp_smooth']**2
        exp_trans_sq = params['per_exp_smooth']**2

        # Compute the smoothing components
        gauss_smoothing = self.k_par_grid**2 * sigma_par_sq
        gauss_smoothing += self.k_trans_grid**2 * sigma_trans_sq
        exp_smoothing = np.abs(self.k_par_grid) * exp_par_sq
        exp_smoothing += np.abs(self.k_trans_grid) * exp_trans_sq

        return np.exp(-gauss_smoothing / 2) * np.exp(-exp_smoothing)

    def compute_velocity_dispersion_gauss(self, params):
        """Compute a gaussian smoothing factor to model velocity dispersion.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        assert 'discrete' in [self.tracer1_type, self.tracer2_type]

        smoothing = np.ones(self.k_par_grid.shape)
        if self.tracer1_type == 'discrete':
            sigma = params['sigma_velo_disp_gauss_' + self.tracer1_name]
            smoothing *= np.exp(-0.25 * (self.k_par_grid * sigma)**2)
        if self.tracer2_type == 'discrete':
            sigma = params['sigma_velo_disp_gauss_' + self.tracer2_name]
            smoothing *= np.exp(-0.25 * (self.k_par_grid * sigma)**2)

        return smoothing

    def compute_velocity_dispersion_lorentz(self, params):
        """Compute a lorentzian smoothing factor to model velocity dispersion.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        ND Array
            Smoothing factor
        """
        assert 'discrete' in [self.tracer1_type, self.tracer2_type]

        smoothing = np.ones(self.k_par_grid.shape)
        if self.tracer1_type == 'discrete':
            sigma = params['sigma_velo_disp_lorentz_' + self.tracer1_name]
            smoothing *= 1. / np.sqrt(1 + (self.k_par_grid * sigma)**2)
        if self.tracer2_type == 'discrete':
            sigma = params['sigma_velo_disp_lorentz_' + self.tracer2_name]
            smoothing *= 1. / np.sqrt(1 + (self.k_par_grid * sigma)**2)

        return smoothing
