import numpy as np
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
        self._tracer1 = tracer1
        self._tracer2 = tracer2
        self._name = dataset_name
        self.k_grid = fiducial['k']

        # Check for the old config
        pk_model = self._config.get('model-pk', None)

        # Get the HCD model and check for UV
        self._hcd_model = None
        self._add_uv = False
        if pk_model is None:
            # Get the new config
            self._hcd_model = self._config.get('model-hcd', None)
            self._add_uv = self._config.getboolean('add uv', False)
        else:
            # Get the old config
            if 'hcd' in pk_model:
                self._hcd_model = pk_model
            if 'uv' in pk_model:
                self._add_uv = True

        # Check if we need fvoigt model
        self._Fvoigt_data = None
        if 'fvoigt_model' in self._config.keys():
            fvoigt_model = self._config.get('fvoigt_model')
            path = '{}/fvoigt_models/Fvoigt_{}.txt'.format(
                            resource_filename('vega', 'models'),
                            fvoigt_model)
            self._Fvoigt_data = np.loadtxt(path)

        # Initialize some stuff we need
        self._pk_Gk = None
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

    def compute(self, pk_lin, params):
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
        bias_beta = utils.bias_beta(params, self._tracer1, self._tracer2)
        bias1, beta1, bias2, beta2 = bias_beta

        # Add UV model
        if self._add_uv:
            if self._tracer1['name'] == 'LYA':
                bias1, beta1 = self.compute_bias_beta_uv(bias1, beta1, params)
            if self._tracer2['name'] == 'LYA':
                bias2, beta2 = self.compute_bias_beta_uv(bias2, beta2, params)

        # Add HCD model
        if self._hcd_model is not None:
            if self._tracer1['name'] == 'LYA':
                bias1, beta1 = self.compute_bias_beta_hcd(bias1, beta1, params)
            if self._tracer2['name'] == 'LYA':
                bias2, beta2 = self.compute_bias_beta_hcd(bias2, beta2, params)

        # Compute kaiser model
        pk_full = pk_lin * self.compute_kaiser(bias1, beta1, bias2, beta2)

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
        if self._pk_Gk is None:
            self._pk_Gk = self.compute_Gk(params)
        pk_full *= self._pk_Gk

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
                raise ValueError('"fullshape smoothing" must be of type \
                                  "gauss" or "exp".')

        # add velocity dispersion
        if 'velocity dispersion' in self._config:
            smoothing_type = self._config.get('velocity dispersion')
            if 'gauss' in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_gauss(params)
            elif 'lorentz' in smoothing_type:
                pk_full *= self.compute_velocity_dispersion_lorentz(params)
            else:
                raise ValueError('"velocity dispersion" must be of type \
                                  "gauss" or "lorentz".')

        return self.k_grid, self.muk_grid, pk_full

    def compute_kaiser(self, bias1, beta1, bias2, beta2):
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
        pk = bias1 * bias2
        pk = pk * (1 + beta1 * self.muk_grid**2)
        pk = pk * (1 + beta2 * self.muk_grid**2)
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

        # ! The sinc model is default right now, but we could specifically
        # ! ask for it and raise an error if we don't find it.
        # ! It's done like this to maintain backwards compatibility
        # TODO Maybe improve the names we search for
        # Check which model we need
        if L0_hcd != self._L0_hcd_cache or self._F_hcd is None:
            if 'Rogers' in self._hcd_model:
                self._F_hcd = self._hcd_Rogers2018(L0_hcd, self.k_par_grid)
            elif 'mask' in self._hcd_model:
                assert self._Fvoigt_data is not None
                self._F_hcd = self._hcd_no_mask(L0_hcd)
            else:
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

        if self._tracer1['name'] == self._tracer2['name']:
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
        sigma_par = params['sigmaNL_par']
        sigma_trans = params['sigmaNL_per']
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
        assert self._tracer1['name'] == "LYA"
        assert self._tracer2['name'] == "LYA"

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
        assert self._tracer1['name'] == "LYA"
        assert self._tracer2['name'] == "LYA"
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
            self._arinyo_dnl_cache = dnl

        return self._arinyo_dnl_cache

    # @staticmethod
    # @jit(nopython=True)
    # def _dnl_arinyo(k_grid, muk_grid, pk_fid, q1, kv, av, bv, kp):

        return dnl

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
        L_par = params["par binsize {}".format(self._name)]
        L_per = params["per binsize {}".format(self._name)]

        Gk = utils.sinc(self.k_par_grid * L_par / 2)
        Gk *= utils.sinc(self.k_trans_grid * L_per / 2)
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
        sigma_par_sq = params['par_sigma_smooth']**2
        sigma_trans_sq = params['per_sigma_smooth']**2
        gauss_smoothing = self.k_par_grid**2 * sigma_par_sq
        gauss_smoothing += self.k_trans_grid**2 * sigma_trans_sq
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
        assert 'discrete' in [self._tracer1['type'], self._tracer2['type']]

        smoothing = np.ones(self.k_par_grid.shape)
        if self._tracer1['type'] == 'discrete':
            sigma = params['sigma_velo_disp_gauss_' + self._tracer1['name']]
            smoothing *= np.exp(-0.25 * (self.k_par_grid * sigma)**2)
        if self._tracer2['type'] == 'discrete':
            sigma = params['sigma_velo_disp_gauss_' + self._tracer2['name']]
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
        assert 'discrete' in [self._tracer1['type'], self._tracer2['type']]

        smoothing = np.ones(self.k_par_grid.shape)
        if self._tracer1['type'] == 'discrete':
            sigma = params['sigma_velo_disp_lorentz_' + self._tracer1['name']]
            smoothing *= 1. / np.sqrt(1 + (self.k_par_grid * sigma)**2)
        if self._tracer2['type'] == 'discrete':
            sigma = params['sigma_velo_disp_lorentz_' + self._tracer2['name']]
            smoothing *= 1. / np.sqrt(1 + (self.k_par_grid * sigma)**2)

        return smoothing
