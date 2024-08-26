import numpy as np
from scipy.integrate import quad
from numba import njit, float64
import os.path
from pathlib import Path
from functools import lru_cache
from scipy.interpolate import interp1d
from astropy.table import Table
import vega

@njit
def jitted_interp(z, input_z, data):
    return np.interp(z, input_z, data)

@njit
def sinc(x):
    return np.sin(x)/x

@njit
def _line_prof(A,mu,sig,wave):
    return A*np.exp(-0.5*(wave-mu)**2/sig**2) * (1/(2*np.sqrt(np.pi)*sig))

#@njit    
def gen_cont(x,dv=1):
    #tuning the amplitudes of peaks by fitting mocks with 250km/s error
    amps=[30,1.5,1.5,0.5,1.5,1,1.5,5,7,25] #25
    #emission line means
    bs=[1025.7,1063,1073,1082,1084,1118,1128,1175,1205,1215.6]
    #emission line default widths
    cs=[10,5.5,3.5,5,5,4,4,7,8.5,15] #15
    
    #fdv = np.exp(dv/3e5)    
    #cs = [np.sqrt(c**2+(b*(fdv-1))**2) for b,c in zip(bs,cs)]

    cs = [np.sqrt(c**2+(b*(dv/3e5))**2) for b,c in zip(bs,cs)]
    line_props = Table({'amp':amps,'lambda_line':bs,'width':cs})
          
    #flux of smooth component
    smooth_level = 1
    scale_factor = 1 
    
    #gaussian peaks of emission lines onto smooth components
    continuum = smooth_level
    #lyb
    continuum += _line_prof(*list(line_props)[0],x)
    continuum += _line_prof(*list(line_props)[1],x)
    continuum += _line_prof(*list(line_props)[2],x)
    continuum += _line_prof(*list(line_props)[3],x)
    continuum += _line_prof(*list(line_props)[4],x)
    continuum += _line_prof(*list(line_props)[5],x)
    continuum += _line_prof(*list(line_props)[6],x)
    #CIII]
    continuum += _line_prof(*list(line_props)[7],x)
    #lya
    continuum += _line_prof(*list(line_props)[8],x)

    continuum += _line_prof(*list(line_props)[9],x)
    
    return continuum/scale_factor

#@njit
def gen_gamma(lrest,sigma_v):
    gamma_fun = (gen_cont(lrest,sigma_v)/gen_cont(lrest,0)) - 1
    return gamma_fun


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


def compute_masked_invcov(cov_mat, data_mask):
    """Compute the masked inverse of the covariance matrix

    Parameters
    ----------
    cov_mat : Array
        Covariance matrix
    data_mask : Array
        Mask of the data
    invert_full_cov : bool, optional
        Flag to invert the full covariance matrix, by default False
    """
    masked_cov = cov_mat[:, data_mask]
    masked_cov = masked_cov[data_mask, :]

    try:
        np.linalg.cholesky(cov_mat)
        print('LOG: Full matrix is positive definite')
    except np.linalg.LinAlgError:
        print('WARNING: Full matrix is not positive definite')

    try:
        np.linalg.cholesky(masked_cov)
        print('LOG: Reduced matrix is positive definite')
    except np.linalg.LinAlgError:
        print('WARNING: Reduced matrix is not positive definite')

    return np.linalg.inv(masked_cov)


def compute_log_cov_det(cov_mat, data_mask):
    """Compute the log of the determinant of the covariance matrix

    Parameters
    ----------
    cov_mat : Array
        Covariance matrix
    data_mask : Array
        Mask of the data

    Returns
    -------
    float
        Log of the determinant of the covariance matrix
    """
    masked_cov = cov_mat[:, data_mask]
    masked_cov = masked_cov[data_mask, :]
    return np.linalg.slogdet(masked_cov)[1]


class VegaBoundsError(Exception):
    pass
