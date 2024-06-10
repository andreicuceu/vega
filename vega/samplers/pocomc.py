from pathlib import Path

import numpy as np
import pocomc as pc
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from scipy.stats import uniform

from vega.samplers.sampler_interface import Sampler


class PocoMC(Sampler):
    """ Interface between Vega and the PocoMC sampler """

    def __init__(self, pocomc_setup, limits, log_lik_func):
        super().__init__(pocomc_setup, limits, log_lik_func)

    def get_sampler_settings(self, sampler_config, num_params, num_derived):
        # Initialize the pocomc settings
        self.precondition = sampler_config.getboolean('precondition', True)
        self.dynamic = sampler_config.getboolean('dynamic', True)
        self.n_effective = sampler_config.getint('n_effective', 512)
        self.n_active = sampler_config.getint('n_active', 256)
        self.n_total = sampler_config.getint('n_total', 1024)
        self.n_evidence = sampler_config.getint('n_evidence', 0)
        self.save_every = sampler_config.getint('save_every', 3)

        self.prior = pc.Prior(
            [uniform(self.limits[par][0], self.limits[par][1]-self.limits[par][0])
             for par in self.limits]
        )

    def run(self):
        """ Run the PocoMC sampler """
        mpi_comm = MPI.COMM_WORLD
        num_mpi_threads = mpi_comm.Get_size()
        with MPIPoolExecutor(num_mpi_threads) as pool:
            self.pocomc_sampler = pc.Sampler(
                self.prior, self.log_lik, pool=pool,
                output_dir=self.path, save_every=self.save_every,
                dynamic=self.dynamic, precondition=self.precondition,
                n_effective=self.n_effective, n_active=self.n_active,
            )
            self.pocomc_sampler.run(self.n_total, self.n_evidence)

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
