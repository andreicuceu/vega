#!/usr/bin/env python
"""Convert a pycorr 2-point correlation measurement (.npy) into the ASCII
multipole format that Vega reads via `data_type = multipoles`.

The DESI clustering pipeline stores the QSO auto-correlation as a pycorr
`TwoPointCorrelationFunction` (an (s, mu) estimator) in a `.npy` file, e.g.

    allcounts_QSO_GCcomb_z1.77-3.8_default_FKP_lin_nran4_njack60_split20.npy

Vega's direct-multipole reader (`Data._read_multipole_data`) instead expects a
plain text file with one row per s-bin and columns

    s_mid  s_avg  xi_0  xi_2  [xi_4 ...]  std_0  std_2  [std_4 ...]

This script projects the (s, mu) measurement onto Legendre multipoles, rebins
in s to the target bin width, applies the [s-min, s-max) range used in the fit,
and writes the expected columns.  Per-multipole standard deviations are taken
from the pycorr jackknife realizations when available, or (preferred for the
fit) from the diagonal of a RascalC covariance file via --rascalc-cov.

Usage
-----
    python bin/pycorr_to_vega_multipoles.py INPUT.npy OUTPUT.txt \\
        --ells 0,2,4 --smin 20 --smax 200 --ds 4 \\
        [--rascalc-cov xi024_..._cov_RascalC.txt]
"""

import argparse
import numpy as np


def _to_ell_major(arr, n_ells, n_s):
    """Return a multipole array with shape (n_ells, n_s).

    pycorr versions differ in whether they return xi_ell as (n_ells, n_s) or
    (n_s, n_ells); normalise to ell-major here.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size == n_ells * n_s:
            return arr.reshape(n_ells, n_s)
        raise ValueError(
            f'Cannot interpret 1D multipole array of size {arr.size} as '
            f'({n_ells} ells x {n_s} s-bins).')
    if arr.shape == (n_ells, n_s):
        return arr
    if arr.shape == (n_s, n_ells):
        return arr.T
    raise ValueError(
        f'Unexpected multipole array shape {arr.shape}; expected '
        f'({n_ells}, {n_s}) or ({n_s}, {n_ells}).')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Input pycorr .npy correlation file.')
    parser.add_argument('output', help='Output ASCII multipole file for Vega.')
    parser.add_argument(
        '--ells', default='0,2,4',
        help='Comma-separated multipole orders to write. Default: 0,2,4')
    parser.add_argument(
        '--smin', type=float, default=20.0,
        help='Lower edge of the output s range (Mpc/h, inclusive). Default: 20')
    parser.add_argument(
        '--smax', type=float, default=200.0,
        help='Upper edge of the output s range (Mpc/h, exclusive). Default: 200')
    parser.add_argument(
        '--ds', type=float, default=4.0,
        help='Target s bin width (Mpc/h). The input is rebinned by the nearest '
             'integer factor to approximate this width. Default: 4')
    parser.add_argument(
        '--rascalc-cov', default=None,
        help='Optional RascalC ASCII covariance (flat n x n matrix). When given, '
             'the std columns are taken from sqrt(diag) of this covariance '
             'instead of the pycorr jackknife errors.')
    args = parser.parse_args()

    ells = tuple(int(x) for x in args.ells.split(','))
    n_ells = len(ells)

    from pycorr import TwoPointCorrelationFunction  # noqa: F401  (import check)
    result = TwoPointCorrelationFunction.load(args.input)

    # Determine the input s bin width and the rebin factor needed to reach --ds.
    s_edges = np.asarray(result.edges[0], dtype=float)
    ds_in = np.mean(np.diff(s_edges))
    rebin = max(1, int(round(args.ds / ds_in)))
    ds_out = ds_in * rebin
    print(f'Input ds = {ds_in:.4f} Mpc/h; rebin factor = {rebin}; '
          f'output ds ~ {ds_out:.4f} Mpc/h')

    if rebin > 1:
        result = result[::rebin]

    # Project onto Legendre multipoles.  mode='poles' returns the jackknife
    # standard deviation as a third element when realizations are available.
    out = result(ells=ells, mode='poles', return_sep=True)
    if isinstance(out, tuple) and len(out) >= 3:
        s_avg_raw, xiell_raw, xierr_raw = out[0], out[1], out[2]
    else:
        s_avg_raw, xiell_raw = out[0], out[1]
        xierr_raw = None
        print('NOTE: jackknife errors not returned by pycorr; std columns '
              'default to 0 unless --rascalc-cov is given.')

    s_avg_full = np.asarray(s_avg_raw, dtype=float)
    n_s_full = s_avg_full.size
    xiell_full = _to_ell_major(xiell_raw, n_ells, n_s_full)
    if xierr_raw is not None:
        std_full = _to_ell_major(xierr_raw, n_ells, n_s_full)
    else:
        std_full = np.zeros_like(xiell_full)

    # Nominal bin centres from the (rebinned) edges, used for the range cut.
    s_edges = np.asarray(result.edges[0], dtype=float)
    s_mid_full = 0.5 * (s_edges[:-1] + s_edges[1:])
    if s_mid_full.size != n_s_full:
        # Fall back to the measured separations if edges and poles disagree.
        s_mid_full = s_avg_full.copy()

    # Apply the output s range.
    sel = (s_mid_full >= args.smin) & (s_mid_full < args.smax)
    n_s = int(sel.sum())
    if n_s == 0:
        raise ValueError(
            f'No s-bins fall within [{args.smin}, {args.smax}). '
            f'Available s range: {s_mid_full.min():.1f} - {s_mid_full.max():.1f}.')
    s_mid = s_mid_full[sel]
    s_avg = s_avg_full[sel]
    xiell = xiell_full[:, sel]
    std = std_full[:, sel]
    print(f'Selected {n_s} s-bins in [{args.smin}, {args.smax}): '
          f'{s_mid[0]:.1f} - {s_mid[-1]:.1f} Mpc/h')

    # Optionally override std with sqrt(diag) of a RascalC covariance.
    if args.rascalc_cov is not None:
        print(f'Reading RascalC covariance {args.rascalc_cov} for std columns.')
        cov = np.loadtxt(args.rascalc_cov, comments='#')
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f'RascalC covariance is not square: shape {cov.shape}.')
        n_cov = cov.shape[0]
        if n_cov % n_ells != 0:
            raise ValueError(
                f'RascalC covariance size {n_cov} is not divisible by the '
                f'number of multipoles ({n_ells}).')
        n_s_cov = n_cov // n_ells
        std_diag = np.sqrt(np.diag(cov)).reshape(n_ells, n_s_cov)

        # Reconstruct the covariance s-grid and match it to the selected bins.
        ds_cov = (args.smax - args.smin) / n_s_cov
        s_cov_centers = args.smin + (np.arange(n_s_cov) + 0.5) * ds_cov
        cov_idx = np.array(
            [int(np.argmin(np.abs(s_cov_centers - sv))) for sv in s_mid])
        max_mismatch = np.max(np.abs(s_cov_centers[cov_idx] - s_mid))
        if max_mismatch > ds_cov:
            print(f'WARNING: covariance s-grid and data s-grid differ by up to '
                  f'{max_mismatch:.2f} Mpc/h (> bin width {ds_cov:.2f}). '
                  f'Check that --smin/--smax/--ds match the covariance.')
        std = std_diag[:, cov_idx]

    header_cols = (['s_mid', 's_avg']
                   + [f'xi_{ell}' for ell in ells]
                   + [f'std_{ell}' for ell in ells])
    table = np.column_stack(
        [s_mid, s_avg]
        + [xiell[i] for i in range(n_ells)]
        + [std[i] for i in range(n_ells)])
    np.savetxt(args.output, table, header=' '.join(header_cols))
    print(f'Wrote {table.shape[0]} rows x {table.shape[1]} cols to {args.output}')


if __name__ == '__main__':
    main()
