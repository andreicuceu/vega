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

        self.parametrisation = config.get('cosmo fit func', 'ap_at')
        if self.parametrisation not in ScaleParameters._parametrisations:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    def get_ap_at(self, params, metal_corr=False):
        """Main compute function for extracting the right ap/at

        Parameters
        ----------
        params : dict
            Computation parameters
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
            return self.fullshape_ap_at(params)
        elif params['peak']:
            return self.standard_ap_at(params)
        elif self.smooth_scaling:
            return self.fullshape_ap_at(params)

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

    def standard_ap_at(self, params):
        """Standard ap/at naming. Used for the peak component in BAO studies,
        or for the full-shape if that option is True.

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
            return params['ap'], params['at']
        elif self.parametrisation == 'aiso_epsilon':
            return self.aiso_epsilon(params)
        elif self.parametrisation == 'phi_alpha':
            return self.phi_alpha(params)
        elif self.parametrisation == 'aiso_aap':
            return self.aiso_aap(params)
        else:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    def fullshape_ap_at(self, params):
        """Full-shape ap/at naming. If full-shape-alpha is False it only works with the
        phi_alpha parametrisation.

        Parameters
        ----------
        params : dict
            Computation parameters

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        if self.parametrisation != 'phi_alpha' and not self.full_shape_alpha:
            raise ValueError('Only the "phi_alpha" parametrisation works with split full-shape. '
                             'Set full-shape-alpha to True for other parametrisations.')

        if self.parametrisation == 'ap_at':
            return params['ap_full'], params['at_full']

        elif self.parametrisation == 'aiso_epsilon':
            return self.aiso_epsilon(params, name_addon='_full')

        elif self.parametrisation == 'phi_alpha':
            if self.full_shape:
                name_addon = '_full'
            else:
                assert self.smooth_scaling
                name_addon = '_smooth'

            return self.phi_alpha(params, name_addon, self.full_shape_alpha)

        else:
            raise ValueError('Unknown parametrisation {}.'.format(self.parametrisation))

    @staticmethod
    def aiso_epsilon(params, name_addon=''):
        """Compute alpha_isotropic / epsilon parametrisation, and return ap/at.

        Parameters
        ----------
        params : dict
            Computation parameters
        name_addon : str, optional
            Name addon for full shape, by default ''

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        aiso = params['aiso' + name_addon]
        epsilon = params['epsilon' + name_addon]
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
    def phi_alpha(params, name_addon='', fullshape_alpha=False):
        """Compute phi / alpha parametrisation, and return ap/at.
        See 2.1 of https://arxiv.org/pdf/2103.14075.pdf for more details.

        Parameters
        ----------
        params : dict
            Computation parameters
        name_addon : str, optional
            Name addon for full shape or smooth scaling, by default ''
        fullshape_alpha : bool, optional
            Whether to have only one isotropic alpha for the full shape, by default False

        Returns
        -------
        float, float
            alpha parallel, alpha transverse
        """
        phi = params['phi' + name_addon]

        if name_addon == '_full' and not fullshape_alpha:
            if params['peak']:
                alpha = params['alpha']
            else:
                alpha = params['alpha_smooth']
        else:
            alpha = params['alpha' + name_addon]

        ap = alpha / np.sqrt(phi)
        at = alpha * np.sqrt(phi)
        return ap, at
