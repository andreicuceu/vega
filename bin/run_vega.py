#!/usr/bin/env python
from vega import VegaInterface
from vega.minimizer import Minimizer
import argparse

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    # Initialize Vega
    vega = VegaInterface(args.config)

    # Check if we need to run over a Monte Carlo mock
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if run_montecarlo and vega.mc_config is not None:
        # Get the MC seed and forecast flag
        seed = vega.main_config['control'].getfloat('mc_seed', 0)
        forecast = vega.main_config['control'].getboolean('forecast', False)

        # Create the mocks
        vega.monte_carlo_sim(vega.mc_config['params'], seed=seed, forecast=forecast)

        # Set to sample the MC params
        sampling_params = vega.mc_config['sample']
        vega.minimizer = Minimizer(vega.chi2, sampling_params)
    elif run_montecarlo:
        raise ValueError('You asked to run over a Monte Carlo simulation,'
                         ' but no "[monte carlo]" section provided.')

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
