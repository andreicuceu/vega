#!/usr/bin/env python
from vega import VegaInterface
import argparse

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    # Initialize Vega
    vega = VegaInterface(args.config)

    # Run minimizer
    vega.minimize()

    # Run chi2scan
    scan_results = None
    if 'chi2 scan' in vega.main_config:
        scan_results = vega.analysis.chi2_scan()

    # Write output
    if vega.minimizer is not None:
        for par, val in vega.bestfit.values.items():
            vega.params[par] = val
    corr_funcs = vega.compute_model(vega.params)
    vega.output.write_results(corr_funcs, vega.params, vega.minimizer, scan_results, vega.models)
