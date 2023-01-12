#!/usr/bin/env python

import os
import argparse

import camb
import mcfit
import fitsio
import numpy as np

from scipy import constants
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline


def pk_to_xi(k, Pk, ell=0, extrap=True):
    xi = mcfit.P2xi(k, l=ell, lowring=True)
    rr, CF = xi(Pk, extrap=extrap)
    return InterpolatedUnivariateSpline(rr, CF)


def xi_to_pk(r, xi, ell=0, extrap=False):
    P = mcfit.xi2P(r, l=ell, lowring=True)
    kk, Pk = P(xi, extrap=extrap)
    return InterpolatedUnivariateSpline(kk, Pk)


def main(ini, out, fid_H0, fid_Ok, fid_wl, z_ref):
    minkh = 1.e-4
    maxkh = 1.1525e3
    npoints = 814

    print(f'INFO: running CAMB on {ini}')
    pars = camb.read_ini(os.path.expandvars(ini))
    pars.Transfer.kmax = maxkh
    if z_ref is not None:
        pars.Transfer.PK_redshifts[0] = z_ref
    if fid_H0 is not None:
        pars.H0 = fid_H0
    if fid_Ok is not None:
        pars.omk = fid_Ok
    if fid_wl is not None:
        pars.DarkEnergy.w = fid_wl

    results = camb.get_results(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=pars.Transfer.kmax,
                                                 npoints=npoints)
    pk = pk[1]
    pars = results.Params
    pars2 = results.get_derived_params()

    cat = {}
    cat['H0'] = pars.H0
    cat['ombh2'] = pars.ombh2
    cat['omch2'] = pars.omch2
    cat['omnuh2'] = pars.omnuh2
    cat['OK'] = pars.omk
    cat['OL'] = results.get_Omega('de')
    cat['ORPHOTON'] = results.get_Omega('photon')
    cat['ORNEUTRI'] = results.get_Omega('neutrino')
    cat['OR'] = cat['ORPHOTON'] + cat['ORNEUTRI']
    cat['OM'] = (cat['ombh2'] + cat['omch2'] + cat['omnuh2']) / (cat['H0'] / 100.)**2
    cat['W'] = pars.DarkEnergy.w
    cat['TCMB'] = pars.TCMB
    cat['NS'] = pars.InitPower.ns
    cat['ZREF'] = pars.Transfer.PK_redshifts[0]
    cat['SIGMA8_ZREF'] = results.get_sigma8()[0]
    cat['SIGMA8_Z0'] = results.get_sigma8()[1]
    cat['F_ZREF'] = results.get_fsigma8()[0] / results.get_sigma8()[0]
    cat['F_Z0'] = results.get_fsigma8()[1] / results.get_sigma8()[1]
    cat['ZDRAG'] = pars2['zdrag']
    cat['RDRAG'] = pars2['rdrag']

    c = constants.speed_of_light / 1000.
    h = cat['H0'] / 100.
    dh = c / (results.hubble_parameter(cat['ZREF']) / h)
    dm = (1. + cat['ZREF']) * results.angular_diameter_distance(cat['ZREF']) * h
    cat['DH'] = dh
    cat['DM'] = dm
    cat['DHoRD'] = cat['DH'] / (cat['RDRAG'] * h)
    cat['DMoRD'] = cat['DM'] / (cat['RDRAG'] * h)

    # Get the Side-Bands
    # Follow 2.2.1 of Kirkby et al. 2013: https://arxiv.org/pdf/1301.3456.pdf
    coef_Planck2015 = (cat['H0'] / 67.31) * (cat['RDRAG'] / 147.334271564563)
    sb1_rmin = 50. * coef_Planck2015
    sb1_rmax = 82. * coef_Planck2015
    sb2_rmin = 150. * coef_Planck2015
    sb2_rmax = 190. * coef_Planck2015
    xi = pk_to_xi(k, pk)
    r = np.logspace(-7., 3.5, 10000)
    xi = xi(r)

    def f_xiSB(r, am3, am2, am1, a0, a1):
        par = [am3, am2, am1, a0, a1]
        model = np.zeros((len(par), r.size))
        tw = r != 0.
        model[0, tw] = par[0] / r[tw]**3
        model[1, tw] = par[1] / r[tw]**2
        model[2, tw] = par[2] / r[tw]**1
        model[3, tw] = par[3]
        model[4, :] = par[4] * r
        model = np.array(model)
        return model.sum(axis=0)

    w = ((r >= sb1_rmin) & (r < sb1_rmax)) | ((r >= sb2_rmin) & (r < sb2_rmax))
    sigma = 0.1 * np.ones(xi.size)
    sigma[(r >= sb1_rmin - 2.) & (r < sb1_rmin + 2.)] = 1.e-6
    sigma[(r >= sb2_rmax - 2.) & (r < sb2_rmax + 2.)] = 1.e-6
    popt, pcov = curve_fit(f_xiSB, r[w], xi[w], sigma=sigma[w])

    model = f_xiSB(r, *popt)
    xiSB = xi.copy()
    ww = (r >= sb1_rmin) & (r < sb2_rmax)
    xiSB[ww] = model[ww]

    pkSB = xi_to_pk(r, xiSB, extrap=True)
    pkSB = pkSB(k)
    pkSB *= pk[-1] / pkSB[-1]

    out = fitsio.FITS(args.out, 'rw', clobber=True)
    head = [{'name': k, 'value': float(v)} for k, v in cat.items()]
    out.write([k, pk, pkSB], names=['K', 'PK', 'PKSB'], header=head, extname='PK')
    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--ini', type=str, required=True,
                        help='Input config file for CAMB')

    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output FITS file')

    parser.add_argument('--fid-H0', type=float, default=None, required=False,
                        help='Hubble parameter, if not given use the one from the config file')

    parser.add_argument('--fid-Ok', type=float, default=None, required=False,
                        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-wl', type=float, default=None, required=False,
                        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument('--z-ref', type=float, default=None, required=False,
                        help='Power-spectrum redshift, default use the one from the config file')

    args = parser.parse_args()

    main(args.ini, args.out, args.fid_H0, args.fid_Ok, args.fid_wl, args.z_ref)
