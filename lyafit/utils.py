import numpy as np
import scipy as sp
from numpy import fft
from scipy import special
from scipy.integrate import quad
from numba import jit, float64
import scipy.interpolate

from . import myGamma

def sinc(x):
    return sp.sin(x)/x

def Pk2Mp(ar, k, pk, ell_vals, muk, dmuk, tform=None):
    """
    Implementation of FFTLog from A.J.S. Hamilton (2000)
    assumes log(k) are equally spaced
    """

    k0 = k[0]
    l=sp.log(k.max()/k0)
    r0=1.

    N=len(k)
    emm=N*fft.fftfreq(N)
    r=r0*sp.exp(-emm*l/N)
    dr=abs(sp.log(r[1]/r[0]))
    s=sp.argsort(r)
    r=r[s]

    xi=sp.zeros([len(ell_vals),len(ar)])

    for ell in ell_vals:
        if tform=="rel":
            pk_ell=pk
            n=1.
        elif tform=="asy":
            pk_ell=pk
            n=2.
        else:
            pk_ell=sp.sum(dmuk*L(muk,ell)*pk,axis=0)*(2*ell+1)*(-1)**(ell//2)/2/sp.pi**2
            n=2.
        mu=ell+0.5
        q=2-n-0.5
        x=q+2*sp.pi*1j*emm/l
        lg1=myGamma.LogGammaLanczos((mu+1+x)/2)
        lg2=myGamma.LogGammaLanczos((mu+1-x)/2)

        um=(k0*r0)**(-2*sp.pi*1j*emm/l)*2**x*sp.exp(lg1-lg2)
        um[0]=sp.real(um[0])
        an=fft.fft(pk_ell*k**n*sp.sqrt(sp.pi/2))
        an*=um
        xi_loc=fft.ifft(an)
        xi_loc=xi_loc[s]
        xi_loc/=r**(3-n)
        xi_loc[-1]=0
        spline=sp.interpolate.splrep(sp.log(r)-dr/2,sp.real(xi_loc),k=3,s=0)
        xi[ell//2,:]=sp.interpolate.splev(sp.log(ar),spline)

    return xi

def Pk2Xi(ar, mur, k, pk, muk, ell_max=None):
    dmuk = 1 / len(muk)

    ell_vals=[ell for ell in range(0,ell_max+1,2)]
    xi=Pk2Mp(ar, k, pk, ell_vals, muk, dmuk)
    for ell in ell_vals:
        xi[ell//2,:]*=L(mur,ell)
    return sp.sum(xi,axis=0)

# def Pk2XiRel(ar,mur,k,pk,kwargs):
#     """Calculate the cross-correlation contribution from relativistic effects (Bonvin et al. 2014).

#     Args:
#         ar (float): r coordinates
#         mur (float): mu coordinates
#         k (float): wavenumbers
#         pk (float): linear matter power spectrum
#         kwargs: dictionary of fit parameters

#     Returns:
#         sum of dipole and octupole correlation terms (float)

#     """
#     ell_vals=[1,3]
#     xi=Pk2Mp(ar,k,pk,ell_vals,tform="rel")
#     return kwargs["Arel1"]*xi[1//2,:]*L(mur,1) + kwargs["Arel3"]*xi[3//2,:]*L(mur,3)

# def Pk2XiAsy(ar,mur,k,pk,kwargs):
#     """Calculate the cross-correlation contribution from standard asymmetry (Bonvin et al. 2014).

#     Args:
#         ar (float): r coordinates
#         mur (float): mu coordinates
#         k (float): wavenumbers
#         pk (float): linear matter power spectrum
#         kwargs: dictionary of fit parameters

#     Returns:
#         sum of dipole and octupole correlation terms (float)

#     """
#     ell_vals=[0,2]
#     xi=Pk2Mp(ar,k,pk,ell_vals,tform="asy")
#     return (kwargs["Aasy0"]*xi[0//2,:] - kwargs["Aasy2"]*xi[2//2,:])*ar*L(mur,1) + kwargs["Aasy3"]*xi[2//2,:]*ar*L(mur,3)

### Legendre Polynomial
def L(mu,ell):
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

def phi_am(pars):
    if pars['peak'] or pars['full-shape']:
        phi = pars['phi']
        am = pars['am']
    elif pars['smooth_scaling']:
        phi = pars['phi_sb']
        am = pars['am_sb']
    else:
        phi = 1.
        am = 1.

    ap = 2 * am / (1 + phi)
    at = ap * phi
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
