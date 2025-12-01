#!/usr/bin/env python
import argparse
import sys

from mpi4py import MPI

from vega import VegaInterface

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
        raise ValueError('Warning: You called "run_vega_mc_mpi.py" without asking '
                         'for monte carlo. Add "run_montecarlo = True" to the "[control]" section.')

    print_func('Finished initializing Vega')

    # Get the fiducial model
    fiducial_model = vega.get_fiducial_for_monte_carlo(print_func=print_func)

    # Activate monte carlo mode
    vega.monte_carlo = True

    # Check if we need to run a forecast
    forecast = vega.main_config['control'].getboolean('forecast', False)
    if forecast:
        raise ValueError('You asked to run a forecast. Use run_vega.py instead.')

    # Get the MC seed and the number of mocks to run
    seed = vega.main_config['control'].getint('mc_seed', 0)
    num_mc_mocks = vega.main_config['control'].getint('num_mc_mocks', 1)
    num_local_mc = num_mc_mocks // num_cpus
    if num_mc_mocks % num_cpus != 0:
        num_local_mc += 1

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
