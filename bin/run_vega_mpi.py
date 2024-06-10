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
    if not vega.run_sampler:
        raise ValueError('Warning: You called "run_vega_mpi.py" without asking'
                         ' for the sampler. Add "run_sampler = True" to the "[control]" section.')

    if vega.sampler == 'Polychord':
        from vega.samplers.polychord import Polychord

        print_func('Running Polychord')
        sampler = Polychord(vega.main_config['Polychord'], sampling_params)
        sampler.run(vega.log_lik)

    elif vega.sampler == 'PocoMC':
        from vega.samplers.pocomc import PocoMC
        import pocomc
        from multiprocessing import Pool

        print_func('Running PocoMC')
        sampler = PocoMC(vega.main_config['PocoMC'], sampling_params)
        if sampler.use_mpi:
            assert False

        def log_lik(theta):
            params = {name: val for name, val in zip(sampler.names, theta)}
            return vega.log_lik(params)

        mpi_comm.barrier()
        with Pool(sampler.num_cpu) as pool:
            sampler.pocomc_sampler = pocomc.Sampler(
                sampler.prior, log_lik,
                pool=pool, output_dir=sampler.path,
                dynamic=sampler.dynamic, precondition=sampler.precondition,
                n_effective=sampler.n_effective, n_active=sampler.n_active,
            )
            sampler.pocomc_sampler.run(sampler.n_total, sampler.n_evidence, save_every=sampler.save_every)

        # sampler.run(vega.log_lik)
        mpi_comm.barrier()

        if cpu_rank == 0:
            sampler.write_chain()
        mpi_comm.barrier()

    print_func('Finished running sampler')
