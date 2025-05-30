#!/usr/bin/env python
import argparse
import sys

from mpi4py import MPI

from vega import VegaInterface

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, help='Config file')
    args = pars.parse_args()

    mpi_comm = MPI.COMM_WORLD
    cpu_rank = mpi_comm.Get_rank()
    num_cpus = mpi_comm.Get_size()

    def print_func(message):
        if cpu_rank == 0:
            print(message)
        sys.stdout.flush()

    print_func('Initializing Vega')

    # Initialize Vega and get the sampling parameters
    vega = VegaInterface(args.config)
    sampling_params = vega.sample_params['limits']

    # run compute_model once to initialize all the caches
    _ = vega.compute_model(run_init=False)

    print_func('Finished initializing Vega')

    # Check if we need to run over a Monte Carlo mock
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if run_montecarlo and vega.mc_config is not None:
        _ = vega.initialize_monte_carlo(print_func=print_func)
        sampling_params = vega.mc_config['sample']['limits']
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
        sampler = Polychord(vega.main_config['Polychord'], sampling_params, vega.log_lik)
        sampler.run()

    elif vega.sampler == 'PocoMC':
        from vega.samplers.pocomc import PocoMC
        import pocomc
        from multiprocessing import Pool
        from schwimmbad import MPIPool

        print_func('Running PocoMC')
        sampler = PocoMC(vega.main_config['PocoMC'], sampling_params, vega.log_lik)
        if cpu_rank == 0:
            sampler.pocomc_output.mkdir()

        def log_lik(theta):
            params = {name: val for name, val in zip(sampler.names, theta)}
            return vega.log_lik(params)

        if sampler.use_mpi:
            mpi_comm.barrier()
            with MPIPool(mpi_comm) as pool:
                pocomc_sampler = pocomc.Sampler(
                    sampler.prior, log_lik, pool=pool, output_dir=sampler.pocomc_output,
                    dynamic=sampler.dynamic, precondition=sampler.precondition,
                    n_effective=sampler.n_effective, n_active=sampler.n_active,
                )
                pocomc_sampler.run(
                    sampler.n_total, sampler.n_evidence, save_every=sampler.save_every)
        else:
            if num_cpus > 1:
                raise ValueError(
                    'You asked to run PocoMC without MPI, '
                    'but you are using more than one MPI thread.'
                )
            with Pool(sampler.num_cpu) as pool:
                pocomc_sampler = pocomc.Sampler(
                    sampler.prior, log_lik, pool=pool, output_dir=sampler.pocomc_output,
                    dynamic=sampler.dynamic, precondition=sampler.precondition,
                    n_effective=sampler.n_effective, n_active=sampler.n_active,
                )
                pocomc_sampler.run(
                    sampler.n_total, sampler.n_evidence, save_every=sampler.save_every)

        if cpu_rank == 0:
            sampler.write_chain(pocomc_sampler)

    print_func('Finished running sampler')
