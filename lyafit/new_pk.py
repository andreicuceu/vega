import numpy as np
from pkg_resources import resource_filename
from . import new_utils as utils


class PowerSpectrum:
    """Power Spectrum computation and handling

    # ! Slow operations should be kept in init as that is only called once
    # ! Compute is called many times and should be fast
    # * Extensions should have their separate method of the form
    # * 'compute_extension' that can be called from outside
    """
    def __init__(self, config, fiducial, tracer1, tracer2, dataset_name=None):
        """Initialize power spectrum

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
        self.k = fiducial['k']

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
            path = '{}\\fvoigt_models\\Fvoigt_{}.txt'.format(
                            resource_filename('lyafit', 'models'),
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

    def compute(self, pk_lin, params):
        """Computes a power spectrum for the tracers using the input
        linear P(k) and parameters

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

        # TODO full gauss smoothing
        # TODO vel dispersion

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

        return self.k, self.muk_grid, pk_full

    def compute_kaiser(self, bias1, beta1, bias2, beta2):
        """Compute Kaiser model

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
        """ Compute effective biases that include UV modeling

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

        W = np.arctan(self.k * lambda_uv) / (self.k * lambda_uv)
        beta_eff = beta / (1 + bias_gamma / bias * W / (1 + bias_prim * W))
        bias_eff = bias + bias_gamma * W / (1 + bias_prim * W)

        return bias_eff, beta_eff

    def compute_bias_beta_hcd(self, bias, beta, params):
        """ Compute effective biases that include HCD modeling

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
        if 'Rogers' in self._hcd_model:
            F_hcd = self._hcd_Rogers2018(L0_hcd)
        elif 'mask' in self._hcd_model:
            assert self._Fvoigt_data is not None
            F_hcd = self._hcd_no_mask(L0_hcd)
        else:
            F_hcd = self._hcd_sinc(L0_hcd)

        bias_eff = bias + bias_hcd * F_hcd
        beta_eff = (bias * beta + bias_hcd * beta_hcd * F_hcd)
        beta_eff /= (bias + bias_hcd * F_hcd)

        return bias_eff, beta_eff

    def _hcd_sinc(self, L0):
        """HCD sinc model

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        1D Array
            F_hcd
        """
        kp = self.k * self.muk_grid
        return utils.sinc(kp * L0)

    def _hcd_Rogers2018(self, L0):
        """Model the effect of HCD systems with the Fourier transform
        of a Lorentzian profile. Motivated by Rogers et al. (2018).

        Parameters
        ----------
        L0 : float
            Characteristic length scale of HCDs

        Returns
        -------
        1D Array
            F_hcd
        """
        kp = self.k * self.muk_grid
        return np.exp(-L0 * kp)

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
        1D Array
            F_hcd
        """
        kp = self.k * self.muk_grid
        k_data = self._Fvoigt_data[:, 0]
        F_data = self._Fvoigt_data[:, 1]

        if self._tracer1['name'] == self._tracer2['name']:
            F_hcd = np.interp(L0 * kp, k_data, F_data, left=0, right=0)
        else:
            F_hcd = np.interp(L0 * kp, k_data, F_data)

        return F_hcd

    def compute_peak_nl(self, params):
        """Compute the non-linear gaussian correction for the peak component

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            pk
        """
        kp = self.k * self.muk_grid
        kt = self.k * np.sqrt(1 - self.muk_grid**2)
        st2 = params['sigmaNL_per']**2
        sp2 = params['sigmaNL_par']**2
        return np.exp(-(kp**2 * sp2 + kt**2 * st2) / 2)

    def compute_dnl_mcdonald(self):
        """Non linear term from McDonald 2003

        Returns
        -------
        1D Array
            D_NL factor
        """
        assert self._tracer1['name'] == "LYA"
        assert self._tracer2['name'] == "LYA"

        kvel = 1.22 * (1 + self.k / 0.923)**0.451
        dnl = (self.k / 6.4)**0.569 - (self.k / 15.3)**2.01
        dnl = dnl - (self.k * self.muk_grid / kvel)**1.5
        return np.exp(dnl)

    def compute_dnl_arinyo(self, params):
        """Non linear term from Arinyo et al 2015

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            D_NL factor
        """
        assert self._tracer1['name'] == "LYA"
        assert self._tracer2['name'] == "LYA"
        q1 = params["dnl_arinyo_q1"]
        kv = params["dnl_arinyo_kv"]
        av = params["dnl_arinyo_av"]
        bv = params["dnl_arinyo_bv"]
        kp = params["dnl_arinyo_kp"]

        growth = q1 * self._k**3 * self._pk_fid / (2 * np.pi**2)
        pec_velocity = (self.k / kv)**av * np.abs(self.muk_grid)**bv
        pressure = (self.k / kp) * (self.k / kp)
        dnl = np.exp(growth * (1 - pec_velocity) - pressure)
        return dnl

    def compute_Gk(self, params):
        """Model the effect of binning of the cf

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        1D Array
            G(k)
        """
        L_par = params["par binsize {}".format(self._name)]
        L_per = params["per binsize {}".format(self._name)]

        kp = self.k * self.muk_grid
        kt = self.k * np.sqrt(1 - self.muk_grid**2)
        return utils.sinc(kp * L_par / 2) * utils.sinc(kt * L_per / 2)
