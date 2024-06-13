#!/usr/bin/env python
import argparse
import sys

from vega import VegaInterface


def main(config, mpi=False):
    cpu_rank = 0
    if mpi:
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()
        num_cpus = mpi_comm.Get_size()

    def print_func(message):
        if cpu_rank == 0:
            print(message)
        sys.stdout.flush()

    print_func('Initializing Vega')

    # Initialize Vega and get the sampling parameters
    vega = VegaInterface(config)
    sampling_params = vega.sample_params['limits']

    print_func('Finished initializing Vega')

    # Check if we need to run over a Monte Carlo mock
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if run_montecarlo and vega.mc_config is not None:
        # Get the MC seed and forecast flag
        seed = vega.main_config['control'].getint('mc_seed', 0)
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

    # run compute_model once to initialize all the caches
    _ = vega.compute_model(run_init=False)

    if vega.sampler == 'Polychord':
        from vega.samplers.polychord import Polychord

        print_func('Running Polychord')
        sampler = Polychord(vega.main_config['Polychord'], sampling_params, vega.log_lik)
        sampler.run()

        print_func('Finished running sampler')
        return sampler

    elif vega.sampler == 'PocoMC':
        from vega.samplers.pocomc import PocoMC
        import pocomc

        print_func('Running PocoMC')
        sampler_config = PocoMC(vega.main_config['PocoMC'], sampling_params, vega.log_lik)
        if cpu_rank == 0:
            sampler_config.pocomc_output.mkdir()
        if sampler_config.resume_state_path is not None:
            print_func(f'Resuming from state file {sampler_config.resume_state_path}.')

        def log_lik(theta):
            params = {name: val for name, val in zip(sampler_config.names, theta)}
            return vega.log_lik(params)

        if mpi:
            from pocomc.parallel import MPIPool
            mpi_comm.barrier()
            with MPIPool(mpi_comm, use_dill=False) as pool:
                sampler = pocomc.Sampler(
                    sampler_config.prior, log_lik, pool=pool,
                    output_dir=sampler_config.pocomc_output,
                    dynamic=sampler_config.dynamic, precondition=sampler_config.precondition,
                    n_effective=sampler_config.n_effective, n_active=sampler_config.n_active,
                )
                sampler.run(
                    sampler_config.n_total, sampler_config.n_evidence,
                    save_every=sampler_config.save_every,
                    resume_state_path=sampler_config.resume_state_path
                )
        else:
            sampler = pocomc.Sampler(
                sampler_config.prior, log_lik, output_dir=sampler_config.pocomc_output,
                dynamic=sampler_config.dynamic, precondition=sampler_config.precondition,
                n_effective=sampler_config.n_effective, n_active=sampler_config.n_active,
            )
            sampler.run(
                sampler_config.n_total, sampler_config.n_evidence,
                save_every=sampler_config.save_every,
                resume_state_path=sampler_config.resume_state_path
            )

        if cpu_rank == 0:
            sampler_config.write_chain(sampler)

        if mpi:
            print_func('Finished running sampler')
        return sampler, sampler_config


if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, help='Config file')
    args = pars.parse_args()

    main(args.config)

    # mpi_comm = MPI.COMM_WORLD
    # cpu_rank = mpi_comm.Get_rank()
    # num_cpus = mpi_comm.Get_size()

    # def print_func(message):
    #     if cpu_rank == 0:
    #         print(message)
    #     sys.stdout.flush()

    # print_func('Initializing Vega')

    # # Initialize Vega and get the sampling parameters
    # vega = VegaInterface(args.config)
    # sampling_params = vega.sample_params['limits']

    # print_func('Finished initializing Vega')

    # # Check if we need the distortion
    # use_distortion = vega.main_config['control'].getboolean('use_distortion', True)
    # if not use_distortion:
    #     for key, data in vega.data.items():
    #         data._distortion_mat = None
    #     test_model = vega.compute_model(vega.params, run_init=True)

    # # Check if we need to run over a Monte Carlo mock
    # run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    # if run_montecarlo and vega.mc_config is not None:
    #     # Get the MC seed and forecast flag
    #     seed = vega.main_config['control'].getint('mc_seed', 0)
    #     forecast = vega.main_config['control'].getboolean('forecast', False)

    #     # Create the mocks
    #     vega.monte_carlo_sim(vega.mc_config['params'], seed=seed, forecast=forecast)

    #     # Set to sample the MC params
    #     sampling_params = vega.mc_config['sample']['limits']
    #     print_func('Created Monte Carlo realization of the correlation')
    # elif run_montecarlo:
    #     raise ValueError('You asked to run over a Monte Carlo simulation,'
    #                      ' but no "[monte carlo]" section provided.')

    # # Run sampler
    # if not vega.run_sampler:
    #     raise ValueError('Warning: You called "run_vega_mpi.py" without asking'
    #                      ' for the sampler. Add "run_sampler = True" to the "[control]" section.')

    # if vega.sampler == 'Polychord':
    #     from vega.samplers.polychord import Polychord

    #     print_func('Running Polychord')
    #     sampler = Polychord(vega.main_config['Polychord'], sampling_params, vega.log_lik)
    #     sampler.run()

    # elif vega.sampler == 'PocoMC':
    #     from vega.samplers.pocomc import PocoMC
    #     import pocomc
    #     from pocomc.parallel import MPIPool
    #     from multiprocessing import Pool
    #     # from schwimmbad import MPIPool

    #     print_func('Running PocoMC')
    #     sampler = PocoMC(vega.main_config['PocoMC'], sampling_params, vega.log_lik)
    #     if cpu_rank == 0:
    #         sampler.pocomc_output.mkdir()

    #     def log_lik(theta):
    #         params = {name: val for name, val in zip(sampler.names, theta)}
    #         return vega.log_lik(params)

    #     if sampler.use_mpi:
    #         mpi_comm.barrier()
    #         with MPIPool(mpi_comm, use_dill=False) as pool:
    #             pocomc_sampler = pocomc.Sampler(
    #                 sampler.prior, log_lik, pool=pool, output_dir=sampler.pocomc_output,
    #                 dynamic=sampler.dynamic, precondition=sampler.precondition,
    #                 n_effective=sampler.n_effective, n_active=sampler.n_active,
    #             )
    #             pocomc_sampler.run(
    #                 sampler.n_total, sampler.n_evidence, save_every=sampler.save_every)
    #     else:
    #         if num_cpus > 1:
    #             raise ValueError(
    #                 'You asked to run PocoMC without MPI, '
    #                 'but you are using more than one MPI thread.'
    #             )
    #         with Pool(sampler.num_cpu) as pool:
    #             pocomc_sampler = pocomc.Sampler(
    #                 sampler.prior, log_lik, pool=pool, output_dir=sampler.pocomc_output,
    #                 dynamic=sampler.dynamic, precondition=sampler.precondition,
    #                 n_effective=sampler.n_effective, n_active=sampler.n_active,
    #             )
    #             pocomc_sampler.run(
    #                 sampler.n_total, sampler.n_evidence, save_every=sampler.save_every)

    #     if cpu_rank == 0:
    #         sampler.write_chain(pocomc_sampler)

    # print_func('Finished running sampler')
