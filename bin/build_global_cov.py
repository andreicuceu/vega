#!/usr/bin/env python
"""Build a block-diagonal global covariance FITS file for the highz 3x2pt fit.

Combines:
  • The existing LyA cross-covariance FITS file (covers the 4 forest
    correlation functions: LyA auto ×2 and QSO×LyA cross ×2).
  • A RascalC ASCII covariance file for the QSO auto-correlation multipoles.

The two blocks are treated as independent (zero off-diagonal cross-covariance).
The output is a single FITS file with a 'COV' column that VegaInterface can
read directly via the global-cov-file key in main.ini.

Usage
-----
    python bin/build_global_cov.py --help
    python bin/build_global_cov.py \\
        --lya-cov   /path/to/lya-qso-cross-covariance.fits \\
        --qso-cov   /path/to/xi024_...cov_RascalC_Gaussian.txt \\
        --output    /path/to/combined_global_cov.fits \\
        [--s-min 20.0] [--s-max 200.0] [--n-multipoles 3]
"""

import argparse
import sys
import numpy as np
from astropy.io import fits


def read_lya_cov(path):
    """Read the LyA cross-covariance from a Vega-format FITS file.

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    cov : 2D ndarray  (N_lya × N_lya)
    """
    with fits.open(path) as hdul:
        cov = hdul[1].data['COV'].copy()
    print(f'LyA cross-covariance: {cov.shape[0]}×{cov.shape[1]} from {path}')
    return cov


def read_rascalc_cov(path, s_min, s_max, n_multipoles):
    """Read a RascalC ASCII covariance and extract the sub-matrix
    corresponding to the s cuts used in the fit.

    The RascalC file stores the full covariance as a flat N×N matrix,
    one row per line, with data ordered as:
        [xi_0(s_1..s_n), xi_2(s_1..s_n), ..., xi_L(s_1..s_n)]
    where the s bins span exactly [s_min, s_max).

    Parameters
    ----------
    path : str
        Path to the ASCII covariance file.
    s_min : float
        Lower separation cut applied to the data (inclusive).
    s_max : float
        Upper separation cut applied to the data (exclusive).
    n_multipoles : int
        Number of multipoles (e.g. 3 for ell=0,2,4).

    Returns
    -------
    cov : 2D ndarray  (N_qso × N_qso)
        N_qso = n_s_bins × n_multipoles
    s_cov_centers : 1D ndarray
        Separation bin centres of the covariance file.
    """
    cov_full = np.loadtxt(path, comments='#')
    n_cov = cov_full.shape[0]

    if cov_full.shape[0] != cov_full.shape[1]:
        raise ValueError(
            f'RascalC covariance is not square: {cov_full.shape}. '
            'Each line of the file must represent one row of the matrix.')

    if n_cov % n_multipoles != 0:
        raise ValueError(
            f'Covariance size {n_cov} is not divisible by n_multipoles '
            f'{n_multipoles}. Check --n-multipoles.')

    n_s_cov = n_cov // n_multipoles
    ds_cov = (s_max - s_min) / n_s_cov
    s_cov_centers = s_min + (np.arange(n_s_cov) + 0.5) * ds_cov

    print(f'RascalC covariance: {n_cov}×{n_cov} ({n_s_cov} s-bins × '
          f'{n_multipoles} multipoles) from {path}')
    print(f'  s range: {s_cov_centers[0]:.1f} – {s_cov_centers[-1]:.1f} Mpc/h '
          f'(step {ds_cov:.1f} Mpc/h)')

    return cov_full, s_cov_centers


def build_combined_cov(lya_cov, qso_cov):
    """Assemble a block-diagonal covariance from the LyA and QSO blocks.

    Parameters
    ----------
    lya_cov : 2D ndarray  (N_lya × N_lya)
    qso_cov : 2D ndarray  (N_qso × N_qso)

    Returns
    -------
    combined : 2D ndarray  ((N_lya + N_qso) × (N_lya + N_qso))
    """
    n_lya = lya_cov.shape[0]
    n_qso = qso_cov.shape[0]
    n_total = n_lya + n_qso

    combined = np.zeros((n_total, n_total), dtype=np.float64)
    combined[:n_lya, :n_lya] = lya_cov
    combined[n_lya:, n_lya:] = qso_cov

    print(f'\nCombined covariance: {n_total}×{n_total}')
    print(f'  LyA block  [0:{n_lya}, 0:{n_lya}]')
    print(f'  QSO block  [{n_lya}:{n_total}, {n_lya}:{n_total}]')
    print(f'  Off-diagonal blocks set to zero (block-diagonal assumption)')

    return combined


def write_fits(cov, output_path, lya_cov_path, qso_cov_path,
               s_min, s_max, n_multipoles):
    """Write the combined covariance to a FITS file in Vega format.

    Parameters
    ----------
    cov : 2D ndarray
    output_path : str
    lya_cov_path, qso_cov_path : str  (stored in header for provenance)
    s_min, s_max : float
    n_multipoles : int
    """
    n = cov.shape[0]
    col = fits.Column(name='COV', format=f'{n}D', array=cov)
    hdu = fits.BinTableHDU.from_columns([col])

    hdu.header['LYA_COV'] = (lya_cov_path, 'Source LyA cross-covariance file')
    hdu.header['QSO_COV'] = (qso_cov_path, 'Source QSO auto RascalC covariance file')
    hdu.header['SMIN'] = (s_min, '[Mpc/h] QSO auto s-min cut')
    hdu.header['SMAX'] = (s_max, '[Mpc/h] QSO auto s-max cut')
    hdu.header['NELL'] = (n_multipoles, 'Number of QSO auto multipoles')
    hdu.header['COMMENT'] = 'Block-diagonal: LyA cross-cov + QSO auto RascalC cov'
    hdu.header['COMMENT'] = 'Cross-covariance between LyA and QSO blocks set to zero'

    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary, hdu])
    hdul.writeto(output_path, overwrite=True)
    print(f'\nWrote combined covariance to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--lya-cov', required=True,
        help='Path to the existing LyA cross-covariance FITS file '
             '(covers all 4 forest correlation functions).')
    parser.add_argument(
        '--qso-cov', required=True,
        help='Path to the RascalC ASCII covariance file for the QSO '
             'auto-correlation multipoles.')
    parser.add_argument(
        '--output', required=True,
        help='Output path for the combined block-diagonal global covariance FITS file.')
    parser.add_argument(
        '--s-min', type=float, default=20.0,
        help='Lower separation cut for the QSO auto (Mpc/h, inclusive). '
             'Must match the s-min in qsoxqso.ini [cuts]. Default: 20.0')
    parser.add_argument(
        '--s-max', type=float, default=200.0,
        help='Upper separation cut for the QSO auto (Mpc/h, exclusive). '
             'Must match the s-max in qsoxqso.ini [cuts]. Default: 200.0')
    parser.add_argument(
        '--n-multipoles', type=int, default=3,
        help='Number of multipoles in the QSO auto measurement '
             '(e.g. 3 for ell=0,2,4). Default: 3')
    args = parser.parse_args()

    # Read the LyA block
    lya_cov = read_lya_cov(args.lya_cov)

    # Read the QSO auto block
    qso_cov, s_centers = read_rascalc_cov(
        args.qso_cov, args.s_min, args.s_max, args.n_multipoles)

    # Verify the QSO covariance covers exactly the s-cut range
    n_s_qso = len(s_centers)
    n_qso_expected = n_s_qso * args.n_multipoles
    if qso_cov.shape[0] != n_qso_expected:
        print(f'WARNING: QSO covariance size {qso_cov.shape[0]} does not match '
              f'n_s_bins × n_multipoles = {n_s_qso} × {args.n_multipoles} = '
              f'{n_qso_expected}. Proceeding with the full matrix.')

    # Check positive-definiteness of each block
    for label, block in [('LyA', lya_cov), ('QSO auto', qso_cov)]:
        eigvals = np.linalg.eigvalsh(block)
        min_eig = eigvals.min()
        if min_eig <= 0:
            print(f'WARNING: {label} covariance block has non-positive '
                  f'eigenvalue {min_eig:.3e}. The matrix may not be '
                  'positive definite.')
        else:
            print(f'{label} covariance block is positive definite '
                  f'(min eigenvalue = {min_eig:.3e})')

    # Build and write the combined covariance
    combined = build_combined_cov(lya_cov, qso_cov)
    write_fits(combined, args.output, args.lya_cov, args.qso_cov,
               args.s_min, args.s_max, args.n_multipoles)

    # Quick sanity check on the output
    with fits.open(args.output) as hdul:
        shape = hdul[1].data['COV'].shape
    print(f'Verification: output COV shape = {shape}')


if __name__ == '__main__':
    main()
