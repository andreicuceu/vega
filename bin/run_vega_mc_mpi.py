#!/usr/bin/env python
import argparse
import sys

from astropy.io import fits
from mpi4py import MPI

from vega import FitResults, VegaInterface
from vega.utils import find_file

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    mpi_comm = MPI.COMM_WORLD
    cpu_rank = mpi_comm.Get_rank()
    num_cpus = mpi_comm.Get_size()

    def print_func(message):
        if cpu_rank == 0:
            print(message)
        sys.stdout.flush()
        mpi_comm.barrier()

    print_func('Initializing Vega')

    # Initialize Vega and get the sampling parameters
    vega = VegaInterface(args.config)
    sampling_params = vega.sample_params['limits']

    # Run monte carlo
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if not run_montecarlo or (vega.mc_config is None):
        raise ValueError('Warning: You called "run_vega_mc_mpi.py" without asking'
                         ' for monte carlo. Add "run_montecarlo = True" to the "[control]" section.')

    print_func('Finished initializing Vega')

    mc_params = vega.mc_config['params']
    mc_start_from_fit = vega.main_config['control'].get('mc_start_from_fit', None)
    # Read existing fit and use the bestfit values for the MC template
    if mc_start_from_fit is not None:
        print_func(f'Reading input fit {mc_start_from_fit}')
        existing_fit = FitResults(find_file(mc_start_from_fit))
        mc_params = existing_fit.params | mc_params

        print_func(f'Set template parameters to {mc_params}.')

    # Do fit on input data and use the bestfit values for the MC template
    elif sampling_params:
        print_func('Running initial fit')
        # run compute_model once to initialize all the caches
        _ = vega.compute_model(run_init=False)

        # Run minimizer
        vega.minimize()

        mc_params = vega.bestfit.values | mc_params

        print_func(f'Set template parameters to {mc_params}.')

    # Check if we need the distortion
    use_distortion = vega.main_config['control'].getboolean('use_distortion', True)
    if not use_distortion:
        for key, data in vega.data.items():
            data._distortion_mat = None
        test_model = vega.compute_model(vega.params, run_init=True)

    # Activate monte carlo mode
    vega.monte_carlo = True

    # Check if we need to run a forecast
    forecast = vega.main_config['control'].getboolean('forecast', False)
    if forecast:
        print('Warning: You called "run_vega_mc_mpi.py" with forecast=True.')
        # raise ValueError('You asked to run a forecast. Use run_vega.py instead.')

    # Get the MC seed and the number of mocks to run
    seed = vega.main_config['control'].getint('mc_seed', 0)
    num_mc_mocks = vega.main_config['control'].getint('num_mc_mocks', 1)
    num_local_mc = num_mc_mocks // num_cpus
    if num_mc_mocks % num_cpus != 0:
        num_local_mc += 1

    # Get fiducial model
    use_measured_fiducial = vega.main_config['control'].getboolean('use_measured_fiducial', False)
    if use_measured_fiducial:
        fiducial_model = {}
        for name in vega.corr_items.keys():
            fiducial_path = vega.main_config['control'].get(f'mc_fiducial_{name}')
            with fits.open(fiducial_path) as hdul:
                fiducial_model[name] = hdul[1].data['DA']
    else:
        fiducial_model = vega.compute_model(mc_params, run_init=False)

    # Run the mocks
    run_mc_fits = vega.main_config['control'].getboolean('run_mc_fits', True)
    local_seed = int(seed + cpu_rank)
    vega.analysis.run_monte_carlo(
        fiducial_model, num_mocks=num_local_mc, seed=local_seed,
        forecast=forecast, run_mc_fits=run_mc_fits
    )

    # Write output
    if num_cpus > 1:
        vega.output.write_monte_carlo(cpu_rank)
    else:
        vega.output.write_monte_carlo()
