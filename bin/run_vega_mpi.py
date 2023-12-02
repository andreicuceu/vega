#!/usr/bin/env python
from vega import VegaInterface
from vega.sampler_interface import Sampler
from mpi4py import MPI
import argparse
import sys


if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    mpi_comm = MPI.COMM_WORLD
    cpu_rank = mpi_comm.Get_rank()

    def print_func(message):
        if cpu_rank == 0:
            print(message)
        sys.stdout.flush()
        mpi_comm.barrier()

    print_func('Initializing Vega')

    # Initialize Vega and get the sampling parameters
    vega = VegaInterface(args.config)
    sampling_params = vega.sample_params['limits']

    print_func('Finished initializing Vega')

    # Check if we need the distortion
    use_distortion = vega.main_config['control'].getboolean('use_distortion', True)
    if not use_distortion:
        for key, data in vega.data.items():
            data._distortion_mat = None
        test_model = vega.compute_model(vega.params, run_init=True)

    # Check if we need to run over a Monte Carlo mock
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if run_montecarlo and vega.mc_config is not None:
        # Get the MC seed and forecast flag
        seed = vega.main_config['control'].getfloat('mc_seed', 0)
        forecast = vega.main_config['control'].getboolean('forecast', False)

        # Create the mocks
        vega.monte_carlo_sim(vega.mc_config['params'], seed=seed, forecast=forecast)

        # Set to sample the MC params
        sampling_params = vega.mc_config['sample']['limits']
        print_func('Created Monte Carlo realization of the correlation')
    elif run_montecarlo:
        raise ValueError('You asked to run over a Monte Carlo simulation,'
                         ' but no "[monte carlo]" section provided.')

    # Run sampler
    if vega.has_sampler:
        print_func('Running the sampler')
        sampler = Sampler(vega.main_config['Polychord'],
                          sampling_params, vega.log_lik)

        sampler.run()
    else:
        raise ValueError('Warning: You called "run_vega_mpi.py" without asking'
                         ' for the sampler. Add "sampler = True" to the "[control]" section.')
