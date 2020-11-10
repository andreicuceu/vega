import numpy as np
# import scipy as sp
from numpy import fft
from scipy import special
from scipy.integrate import quad
from numba import jit, float64
from scipy import interpolate
import os.path
from pathlib import Path

import vega

@jit(nopython=True)
def sinc(x):
    return np.sin(x)/x


def Pk2Mp(ar, k, pk, ell_vals, muk, dmuk, tform=None):
    """
    Implementation of FFTLog from A.J.S. Hamilton (2000)
    assumes log(k) are equally spaced
    """

    k0 = k[0]
    l = np.log(k.max()/k0)
    r0 = 1.

    N = len(k)
    emm = N*fft.fftfreq(N)
    r = r0*np.exp(-emm*l/N)
    dr = abs(np.log(r[1]/r[0]))
    s = np.argsort(r)
    r = r[s]

    xi = np.zeros([len(ell_vals),len(ar)])

    for ell in ell_vals:
        if tform == "rel":
            pk_ell = pk
            n = 1.
        elif tform == "asy":
            pk_ell = pk
            n = 2.
        else:
            pk_ell = np.sum(dmuk*L(muk,ell)*pk,axis=0)*(2*ell+1)*(-1)**(ell//2)/2/np.pi**2
            n = 2.
        mu = ell+0.5
        q = 2-n-0.5
        x = q+2*np.pi*1j*emm/l
        lg1 = special.loggamma((mu+1+x)/2)
        lg2 = special.loggamma((mu+1-x)/2)

        um = (k0*r0)**(-2*np.pi*1j*emm/l)*2**x*np.exp(lg1-lg2)
        um[0] = np.real(um[0])
        an = fft.fft(pk_ell*k**n*np.sqrt(np.pi/2))
        an *= um
        xi_loc = fft.ifft(an)
        xi_loc = xi_loc[s]
        xi_loc /= r**(3-n)
        xi_loc[-1] = 0
        spline = interpolate.splrep(np.log(r)-dr/2,np.real(xi_loc),k=3,s=0)
        xi[ell//2,:] = interpolate.splev(np.log(ar),spline)

    return xi


def pk_to_xi(r_grid, mu_grid, k_grid, muk_grid, pk, ell_max):
    """Compute the correlation function from an input power spectrum

    Parameters
    ----------
    r_grid : 1D Array
        Grid of r coordinates for the output correlation
    mu_grid : 1D Array
        Grid of mu coordinates for the output correlation
    k_grid : 1D Array
        Grid of k coordinates for the input power spectrum
    muk_grid : ND Array
        Grid of muk = kp/k coordinates for the input power spectrum
    pk : ND Array
        Input power spectrum
    ell_max : int
        Maximum multipole to sum over

    Returns
    -------
    1D Array
        Output correlation function
    """
    # Check what multipoles we need and compute them
    dmuk = 1 / len(muk_grid)
    ell_vals = np.arange(0, ell_max + 1, 2)

    xi = Pk2Mp(r_grid, k_grid, pk, ell_vals, muk_grid, dmuk)

    # Add the Legendre polynomials and sum over the multipoles
    for ell in ell_vals:
        xi[ell//2, :] *= L(mu_grid, ell)
    return np.sum(xi, axis=0)


def pk_to_xi_relativistic(r_grid, mu_grid, k_grid, muk_grid, pk, params):
    """Calculate the cross-correlation contribution from
    relativistic effects (Bonvin et al. 2014).

    Parameters
    ----------
    r_grid : 1D Array
        Grid of r coordinates for the output correlation
    mu_grid : 1D Array
        Grid of mu coordinates for the output correlation
    k_grid : 1D Array
        Grid of k coordinates for the input power spectrum
    muk_grid : ND Array
        Grid of muk = kp/k coordinates for the input power spectrum
    pk : ND Array
        Input power spectrum
    params : dict
        Computation parameters

    Returns
    -------
    1D Array
        Output xi relativistic
    """
    # Compute the dipole and octupole terms
    ell_vals = [1, 3]
    dmuk = 1 / len(muk_grid)
    xi = Pk2Mp(r_grid, k_grid, pk, ell_vals, muk_grid, dmuk, tform='rel')

    # Get the relativistic parameters and sum over the monopoles
    A_rel_1 = params['Arel1']
    A_rel_3 = params['Arel3']
    xi_rel = A_rel_1 * xi[1//2, :] * L(mu_grid, 1)
    xi_rel += A_rel_3 * xi[3//2, :] * L(mu_grid, 3)
    return xi_rel


def pk_to_xi_asymmetry(r_grid, mu_grid, k_grid, muk_grid, pk, params):
    """Calculate the cross-correlation contribution from
    standard asymmetry (Bonvin et al. 2014).

    Parameters
    ----------
    r_grid : 1D Array
        Grid of r coordinates for the output correlation
    mu_grid : 1D Array
        Grid of mu coordinates for the output correlation
    k_grid : 1D Array
        Grid of k coordinates for the input power spectrum
    muk_grid : ND Array
        Grid of muk = kp/k coordinates for the input power spectrum
    pk : ND Array
        Input power spectrum
    params : dict
        Computation parameters

    Returns
    -------
    1D Array
        Output xi asymmetry
    """
    # Compute the monopole and quadrupole terms
    ell_vals = [0, 2]
    dmuk = 1 / len(muk_grid)
    xi = Pk2Mp(r_grid, k_grid, pk, ell_vals, muk_grid, dmuk, tform='asy')

    # Get the asymmetry parameters and sum over the monopoles
    A_asy_0 = params['Aasy0']
    A_asy_2 = params['Aasy2']
    A_asy_3 = params['Aasy3']
    xi_asy = (A_asy_0 * xi[0, :] - A_asy_2 * xi[1, :]) * r_grid * L(mu_grid, 1)
    xi_asy += A_asy_3 * xi[1, :] * r_grid * L(mu_grid, 3)
    return xi_asy


### Legendre Polynomial
def L(mu, ell):
    return special.legendre(ell)(mu)

def bias_beta(kwargs, tracer1, tracer2):

    growth_rate = kwargs["growth_rate"]

    beta1 = kwargs["beta_{}".format(tracer1['name'])]
    bias1 = kwargs["bias_eta_{}".format(tracer1['name'])]
    bias1 *= growth_rate/beta1

    beta2 = kwargs["beta_{}".format(tracer2['name'])]
    bias2 = kwargs["bias_eta_{}".format(tracer2['name'])]
    bias2 *= growth_rate/beta2

    return bias1, beta1, bias2, beta2

def ap_at(pars):
    if pars['peak'] or pars['full-shape']:
        return pars['ap'], pars['at']

    if pars['smooth_scaling']:
        return pars['ap_sb'], pars['at_sb']

    return 1., 1.


def ap_at_custom(pars):
    if pars['peak'] or pars['full-shape']:
        ap = pars['ap']
        at = pars['at']
    elif pars['smooth_scaling']:
        phi = pars['phi_smooth']
        gamma = pars['gamma_smooth']
        ap = np.sqrt(gamma / phi)
        at = np.sqrt(gamma * phi)
    else:
        ap = 1.
        at = 1.

    return ap, at


def phi_gamma(pars):
    if pars['peak'] or pars['full-shape']:
        phi = pars['phi']
        gamma = pars['gamma']
    elif pars['smooth_scaling']:
        phi = pars['phi_smooth']
        gamma = pars['gamma_smooth']
    else:
        phi = 1.
        gamma = 1.

    ap = np.sqrt(gamma / phi)
    at = np.sqrt(gamma * phi)
    return ap, at

def aiso_epsilon(pars):
    if pars['peak'] or pars['full-shape']:
        aiso = pars['aiso']
        eps = pars['1+epsilon']
        ap = aiso*eps*eps
        at = aiso/eps
    else:
        ap = 1.
        at = 1.
    return ap, at

def convert_instance_to_dictionary(inst):
    dic = dict((name, getattr(inst, name)) for name in dir(inst) if not name.startswith('__'))
    return dic


@jit(float64(float64, float64, float64))
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


@jit(float64(float64, float64, float64))
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


def growth_function(z, Omega_m, Omega_de):
    """Compute growth factor at redshift z

    Parameters
    ----------
    z : float
        redshift
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0
    """
    a = 1 / (1 + z)
    args = (Omega_m, Omega_de)
    growth_int = quad(growth_integrand, 0, a, args=args)[0]
    hubble_par = hubble(z, Omega_m, Omega_de)
    return 5./2. * Omega_m * hubble_par * growth_int


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
    input_path = Path(path)

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
