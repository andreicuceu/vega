import numpy as np


class ScaleParameters:
    """Class for handling scale parameters.
    Can choose between three parametrisations:
    Standard anisotropic BAO: ap = rp'/rp, at = rt'/rt
    Standard full-shape AP: aiso = at * cbrt(ap / at), 1 + epsilon = cbrt(ap / at)
    Lya AP: phi = at/ap, alpha = sqrt(ap * at)
    DESI convenction: aiso = cbrt(ap * at**2), aap = ap / at
    See 2.1 of https://arxiv.org/pdf/2103.14075.pdf for more details.
    """
    _parametrisations = [
        'ap_at', 'aiso_epsilon', 'phi_alpha', 'aiso_aap']

    def __init__(self, config):
        """Initialize scale parameters from the cosmo-fit type config

        Parameters
        ----------
        config : ConfigParser
            [cosmo-fit type] section of the main config
        """
        self.full_shape = config.getboolean('full-shape', False)
        self.full_shape_alpha = config.getboolean('full-shape-alpha', False)
        self.smooth_scaling = config.getboolean('smooth-scaling', False)
        self.metal_scaling = config.getboolean('metal-scaling', False)
        self.two_alpha_smooth = config.getboolean('two-alpha-smooth', False)

        if self.full_shape_alpha and self.two_alpha_smooth:
            raise ValueError(
                'The "full-shape-alpha" and "two-alpha-smooth" options are incompatible.')

        if self.metal_scaling and self.two_alpha_smooth:
            raise ValueError(
                'The "metal-scaling" and "two-alpha-smooth" options are incompatible.')

        self.parametrisation = config.get('cosmo fit func', 'ap_at')
        if self.parametrisation not in ScaleParameters._parametrisations:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    def get_ap_at(self, params, corr_name=None, metal_corr=False):
        """Main compute function for extracting the right ap/at

        Parameters
        ----------
        params : dict
            Computation parameters
        corr_name : str, optional
            Name of the correlation, by default None. Only used for the two-alpha-smooth option
        metal_corr : bool, optional
            Whether we are working with a metal correlation, by default False


        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        if metal_corr and not self.metal_scaling:
            return self.default()

        if self.full_shape:
            return self.get_fullshape_params(params, corr_name, )
        elif params['peak']:
            return self.get_bao_params(params)
        elif self.smooth_scaling:
            return self.get_fullshape_params(params, corr_name)

        return self.default()

    @staticmethod
    def default():
        """Default values for alpha_par and alpha_perp

        Returns
        -------
        float, float
            1., 1.
        """
        return 1., 1.

    def get_bao_params(self, params):
        """Used for the peak component in both BAO and full-shape fits.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        if self.parametrisation == 'ap_at':
            return self.ap_at(params)
        elif self.parametrisation == 'aiso_epsilon':
            return self.aiso_epsilon(params)
        elif self.parametrisation == 'phi_alpha':
            return self.phi_alpha(params)
        elif self.parametrisation == 'aiso_aap':
            return self.aiso_aap(params)
        else:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    def get_fullshape_params(self, params, corr_name=None):
        """Full-shape ap/at naming. If full-shape-alpha is False it only works with the
        phi_alpha parametrisation.

        Parameters
        ----------
        params : dict
            Computation parameters
        corr_name : str, optional
            Name of the correlation, by default None. Only used for the two-alpha-smooth option

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        if self.parametrisation != 'phi_alpha' and not self.full_shape_alpha:
            raise ValueError('Only the "phi_alpha" parametrisation works with split full-shape. '
                             'Set full-shape-alpha to True for other parametrisations.')

        if self.parametrisation == 'ap_at':
            return self.ap_at(params, ap_name='ap_full', at_name='at_full')

        elif self.parametrisation == 'aiso_epsilon':
            return self.aiso_epsilon(params, aiso_name='aiso_full', epsilon_name='epsilon_full')

        elif self.parametrisation == 'phi_alpha':
            return self.get_fullshape_phi_alpha(params, corr_name)

        else:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    def get_fullshape_phi_alpha(self, params, corr_name=None):
        """Full-shape phi/alpha parametrisation. If two-alpha-smooth is False it only works with the
        peak component.

        Parameters
        ----------
        params : dict
            Computation parameters
        corr_name : str, optional
            Name of the correlation, by default None. Only used for the two-alpha-smooth option

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        phi_name = 'phi_full' if self.full_shape else 'phi_smooth'

        if self.full_shape_alpha:
            alpha_name = 'alpha_full'
        elif params['peak']:
            alpha_name = 'alpha'
        elif self.two_alpha_smooth:
            alpha_name = f'alpha_smooth_{corr_name}'
        else:
            alpha_name = 'alpha_smooth'

        return self.phi_alpha(params, phi_name=phi_name, alpha_name=alpha_name)

    @staticmethod
    def ap_at(params, ap_name='ap', at_name='at'):
        """Return alpha parallel and alpha transverse from parameters.

        Parameters
        ----------
        params : dict
            Computation parameters
        ap_name : str, optional
            Name of the alpha parallel parameter, by default 'ap'
        at_name : str, optional
            Name of the alpha transverse parameter, by default 'at'

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        return params[ap_name], params[at_name]

    @staticmethod
    def aiso_epsilon(params, aiso_name='aiso', epsilon_name='epsilon'):
        """Compute alpha_isotropic / epsilon parametrisation, and return ap/at.

        Parameters
        ----------
        params : dict
            Computation parameters
        aiso_name : str, optional
            Name of the isotropic alpha parameter, by default 'aiso'
        epsilon_name : str, optional
            Name of the epsilon parameter, by default 'epsilon'

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        aiso = params[aiso_name]
        epsilon = params[epsilon_name]
        ap = aiso * (1 + epsilon)**2
        at = aiso / (1 + epsilon)

        return ap, at

    @staticmethod
    def aiso_aap(params, name_addon=''):
        aiso = params['aiso' + name_addon]
        aap = params['aap' + name_addon]
        at = aiso / np.cbrt(aap)
        ap = at * aap

        return ap, at

    @staticmethod
    def phi_alpha(params, phi_name='phi', alpha_name='alpha'):
        """Compute phi / alpha parametrisation, and return ap/at.
        See 2.1 of https://arxiv.org/pdf/2103.14075.pdf for more details.

        Parameters
        ----------
        params : dict
            Computation parameters
        phi_name : str, optional
            Name of the phi parameter, by default 'phi'
        alpha_name : str, optional
            Name of the alpha parameter, by default 'alpha'

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        phi = params[phi_name]
        alpha = params[alpha_name]
        ap = alpha / np.sqrt(phi)
        at = alpha * np.sqrt(phi)
        return ap, at
