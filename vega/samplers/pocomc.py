from pathlib import Path

import numpy as np
import pocomc
from mpi4py import MPI
from schwimmbad import MPIPool
# from mpi4py.futures import MPIPoolExecutor
from multiprocessing import Pool
from scipy.stats import uniform

from vega.samplers.sampler_interface import Sampler


class PocoMC(Sampler):
    """ Interface between Vega and the PocoMC sampler """

    def __init__(self, args):
        super().__init__(args)

    def get_sampler_settings(self, sampler_config, num_params, num_derived):
        # Initialize the pocomc settings
        self.precondition = sampler_config.getboolean('precondition', True)
        self.dynamic = sampler_config.getboolean('dynamic', False)
        self.n_effective = sampler_config.getint('n_effective', 512)
        self.n_active = sampler_config.getint('n_active', 256)
        self.n_total = sampler_config.getint('n_total', 1024)
        self.n_evidence = sampler_config.getint('n_evidence', 0)
        self.save_every = sampler_config.getint('save_every', 3)

        self.use_mpi = sampler_config.getboolean('use_mpi', False)
        self.num_cpu = sampler_config.getint('num_cpu', 64)

        self.prior = pocomc.Prior(
            [uniform(self.limits[par][0], self.limits[par][1]-self.limits[par][0])
             for par in self.limits]
        )

    def vec_log_lik(self, theta):
        params = {name: val for name, val in zip(self.names, theta)}
        return self.vega.log_lik(params)

    def run(self):
        if self.use_mpi:
            self._run_mpi()
        else:
            self._run_multiprocessing()

        self.print_func('Finished running sampler')

    def _run_mpi(self):
        """ Run the PocoMC sampler """
        with MPIPool(self.mpi_comm) as pool:
            self.pocomc_sampler = pocomc.Sampler(
                self.prior, self.vec_log_lik, pool=pool, output_dir=self.path,
                dynamic=self.dynamic, precondition=self.precondition,
                n_effective=self.n_effective, n_active=self.n_active,
            )
            self.pocomc_sampler.run(self.n_total, self.n_evidence, save_every=self.save_every)

    def _run_multiprocessing(self):
        """ Run the PocoMC sampler """
        with Pool(self.num_cpu) as pool:
            self.pocomc_sampler = pocomc.Sampler(
                self.prior, self.vec_log_lik, pool=pool, output_dir=self.path,
                dynamic=self.dynamic, precondition=self.precondition,
                n_effective=self.n_effective, n_active=self.n_active,
            )
            self.pocomc_sampler.run(self.n_total, self.n_evidence, save_every=self.save_every)

    def write_chain(self):
        # Get the weighted posterior samples
        samples, weights, logl, logp = self.pocomc_sampler.posterior()

        # Write the chain
        chain_path = Path(self.path) / (self.name + '.txt')
        chain = np.column_stack((weights, logl, samples))
        print(f'Writing chain to {chain_path}')
        np.savetxt(chain_path, chain, header='Weights, Log Likelihood, ' + ', '.join(self.names))

        # Write stats
        stats_path = Path(self.path) / (self.name + '.stats')
        stats = np.column_stack((weights, logl, logp))
        np.savetxt(stats_path, stats, header='Weights, Log Likelihood, Log Prior')

        # Print Evidence
        logZ, logZerr = self.pocomc_sampler.evidence()
        print(f'log(Z) = {logZ} +/- {logZerr}')
