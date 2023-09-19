import numpy as np
from astropy.io import fits
from scipy.integrate import quad
from numba import njit, float64
import os.path
from pathlib import Path
from functools import lru_cache
from scipy.interpolate import interp1d

import vega


@njit
def sinc(x):
    return np.sin(x)/x


def _tracer_bias_beta(params, name):
    """Get the bias and beta values for a tracer

    Parameters
    ----------
    params : dict
        Computation parameters
    name : string
        Name of tracer

    Returns
    -------
    float, float
        bias, beta
    """
    growth_rate = params.get("growth_rate", 0.970386)

    bias = params.get('bias_' + name, None)
    bias_eta = params.get('bias_eta_' + name, None)
    beta = params.get('beta_' + name, None)

    err_msg = ("For each tracer, you need to specify two of these three:"
               " (bias, bias_eta, beta)."
               " If all three are given, we use bias and beta. "
               f"Offending tracer: {name}")

    if bias is None:
        assert bias_eta is not None and beta is not None, err_msg
        bias = bias_eta * growth_rate / beta

    if bias_eta is None:
        assert bias is not None and beta is not None, err_msg

    if beta is None:
        assert bias is not None and bias_eta is not None, err_msg
        beta = bias_eta * growth_rate / bias

    return bias, beta


def bias_beta(params, tracer1_name, tracer2_name):
    """Get bias and beta values for the two tracers

    Parameters
    ----------
    params : dict
        Computation parameters
    tracer1 : dict
        Config of tracer 1
    tracer2 : dict
        Config of tracer 2

    Returns
    -------
    float, float, float, float
        bias_1, beta_1, bias_2, beta_2
    """
    bias1, beta1 = _tracer_bias_beta(params, tracer1_name)
    if tracer1_name == tracer2_name:
        bias2, beta2 = bias1, beta1
    else:
        bias2, beta2 = _tracer_bias_beta(params, tracer2_name)

    return bias1, beta1, bias2, beta2


def convert_instance_to_dictionary(inst):
    dic = dict((name, getattr(inst, name)) for name in dir(inst) if not name.startswith('__'))
    return dic


@njit(float64(float64, float64, float64))
def hubble(z, Omega_m, Omega_de):
    """Hubble parameter in LCDM + curvature
    No H0/radiation/neutrinos

    Parameters
    ----------
    z : float
        Redshift
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0

    Returns
    -------
    float
        Hubble parameter
    """
    Omega_k = 1 - Omega_m - Omega_de
    e_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_de + Omega_k * (1 + z)**2)
    return e_z


@njit(float64(float64, float64, float64))
def growth_integrand(a, Omega_m, Omega_de):
    """Integrand for the growth factor

    Parameters
    ----------
    a : float
        Scale factor
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0

    Returns
    -------
    float
        Growth integrand
    """
    z = 1 / a - 1
    inv_int = (a * hubble(z, Omega_m, Omega_de))**3
    return 1./inv_int


@lru_cache(maxsize=32)
def get_growth_interp(Omega_m, Omega_de):
    """Build growth function interpolation
    This function should be cached.

    Parameters
    ----------
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0

    Returns
    -------
    scipy.interpolation.interp1d
        Growth function interpolation
    """
    # Initialize redshift and growth arrays
    z_grid = np.linspace(0, 10, 1000)
    growth = np.zeros(1000)

    # Compute growth at each redshift
    for i, z in enumerate(z_grid):
        a = 1 / (1 + z)
        args = (Omega_m, Omega_de)
        growth_int = quad(growth_integrand, 0, a, args=args)[0]
        hubble_par = hubble(z, Omega_m, Omega_de)
        growth[i] = 5./2. * Omega_m * hubble_par * growth_int

    # Return growth interpolation
    return interp1d(z_grid, growth, kind='cubic')


def growth_function(z, Omega_m, Omega_de):
    """Compute growth factor at redshift z

    Parameters
    ----------
    z : float or array
        redshift
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0

    Returns
    -------
    float or array
        Growth function
    """
    # Get cached interpolation
    growth_interp = get_growth_interp(Omega_m, Omega_de)
    return growth_interp(z)


def find_file(path):
    """ Find files on the system.

    Checks if it's an absolute path or something inside vega,
    and returns a proper path.

    Relative paths are checked from the vega main path,
    vega/models and tests

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to vega
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.is_file():
        return input_path

    # Get the vega path and check inside vega (this returns vega/vega)
    vega_path = Path(os.path.dirname(vega.__file__))

    # Check if it's a model
    model = vega_path / 'models' / input_path
    if model.is_file():
        return model

    # Check if it's something used for tests
    test = vega_path.parents[0] / 'tests' / input_path
    if test.is_file():
        return test

    # Check from the main vega folder
    in_vega = vega_path.parents[0] / input_path
    if in_vega.is_file():
        return in_vega

    raise RuntimeError('The path/file does not exists: ', input_path)

def get_zeff(corr_items):
    """Compute effective redshift of all correlations

    Parameters
    ----------
    corr_items : List[vega.correlation_item.CorrelationItem]
        List of correlation items.

    Returns
    -------
    float
        Effective redshift
    """
    zeff_list = []
    weights = []
    
    for corr_item in corr_items:
        rmin = corr_item.config['cuts'].getfloat('r-min', 10)
        rmax = corr_item.config['cuts'].getfloat('r-max', 300)

        hdul = fits.open(corr_item.config['data'].get('filename'))

        r_arr = np.sqrt(hdul[1].data['RP']**2 + hdul[1].data['RT']**2)
        cells = (r_arr > rmin) * (r_arr < rmax)

        zeff = np.average(hdul[1].data['Z'][cells], weights=hdul[1].data['NB'][cells])
        weight = np.sum(hdul[1].data['NB'][cells])

        hdul.close()

        zeff_list.append(zeff)
        weights.append(weight)

    zeff = np.average(zeff_list, weights=weights)
    return zeff

class VegaBoundsError(Exception):
    pass
