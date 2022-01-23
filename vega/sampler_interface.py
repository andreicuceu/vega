import numpy as np
import os.path
import sys
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from pathlib import Path
from vega.postprocess.param_utils import build_names
from mpi4py import MPI


class Sampler:
    ''' Interface between Vega and the nested sampler PolyChord '''

    def __init__(self, polychord_setup, limits, log_lik_func):
        """

        Parameters
        ----------
        polychord_setup : ConfigParser
            Polychord section from the main config
        limits : dict
            Dictionary with the prior limits of the sampled parameters
        log_lik_func : f(params)
            Log Likelihood function to be passed to Polychord
        """
        self.limits = limits
        self.names = limits.keys()
        self.num_params = len(limits)
        self.num_derived = 0
        self.log_lik = log_lik_func
        self.getdist_latex = polychord_setup.getboolean('getdist_latex', True)

        # Check limits are well defined
        for lims in self.limits.values():
            if None in lims:
                raise ValueError('Sampler needs well defined prior limits.'
                                 ' You passed a None. Please give numbers, or'
                                 ' just say par_name = True to use defaults.')

        # Initialize the Polychord settings
        self.settings, self.parnames_path = self.get_polychord_settings(
                polychord_setup, self.num_params, self.num_derived)

    @staticmethod
    def get_polychord_settings(polychord_setup, num_params, num_derived):
        """Extract polychord settings and create the settings object.

        Parameters
        ----------
        polychord_setup : ConfigParser
            Polychord section from the main config
        num_params : int
            Number of sampled parameters
        num_derived : int
            Number of derived parameters

        Returns
        -------
        PolyChordSettings
            Settings object for running Polychord
        """
        # Seed and path/name
        seed = polychord_setup.getint('seed', int(0))
        path = os.path.expandvars(polychord_setup.get('path'))
        name = polychord_setup.get('name')

        # The key config parameters
        num_live = polychord_setup.getint('num_live', int(25*num_params))
        num_repeats = polychord_setup.getint('num_repeats', int(5*num_params))
        precision = polychord_setup.getfloat('precision', float(0.001))

        # Resume should almost always be true
        resume = polychord_setup.getboolean('resume', True)
        write_dead = polychord_setup.getboolean('write_dead', True)

        # Useful for plotting as it gives you more posterior samples
        boost_posterior = polychord_setup.getfloat('boost_posterior',
                                                   float(0.0))

        # Do we do clustering, useful for multimodal distributions
        do_clustering = polychord_setup.getboolean('do_clustering', False)
        cluster_posteriors = polychord_setup.getboolean('cluster_posteriors',
                                                        False)

        # Perform maximisation at the end of the chain
        maximise = polychord_setup.getboolean('maximise', False)

        # These control different sampling speeds
        # grade_frac : List[float]
        #    (Default: [1])
        #    The amount of time to spend in each speed.
        #    If any of grade_frac are <= 1, then polychord will time each
        #    sub-speed, and then choose num_repeats for the number of slowest
        #    repeats, and spend the proportion of time indicated by grade_frac.
        #    Otherwise this indicates the number of repeats to spend in
        #    each speed.
        # grade_dims : List[int]
        #     (Default: nDims)
        #     The number of parameters within each speed.

        # Initialize the settings
        settings = PolyChordSettings(num_params, num_derived, base_dir=path,
                                     file_root=name, seed=seed, nlive=num_live,
                                     num_repeats=num_repeats,
                                     precision_criterion=precision,
                                     write_resume=resume, read_resume=resume,
                                     boost_posterior=boost_posterior,
                                     do_clustering=do_clustering,
                                     cluster_posteriors=cluster_posteriors,
                                     equals=False, write_dead=write_dead,
                                     maximise=maximise,
                                     write_live=False, write_prior=False)

        # Check the path and get the paramnames path
        output_path = Path(path)
        err_msg = ("The PolyChord 'path' does not correspond to an existing"
                   " folder. Create the output folder before running.")
        assert output_path.exists(), err_msg
        parnames_path = output_path / (name + '.paramnames')

        return settings, parnames_path

    def write_parnames(self):
        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()

        if cpu_rank == 0:
            print('Writing parameter names')
            sys.stdout.flush()
            latex_names = build_names(list(self.names))
            with open(self.parnames_path, 'w') as f:
                for name, latex in latex_names.items():
                    if self.getdist_latex:
                        f.write('%s    %s\n' % (name, latex))
                    else:
                        f.write('%s    $%s$\n' % (name, latex))
            print('Finished writing parameter names')
            sys.stdout.flush()

        mpi_comm.barrier()

    def run(self):
        """Run Polychord. We need to pass three functions:

        log_lik: takes a list of parameter values and
            returns tuple: (log_lik, list of derived)

        prior: takes a unit hypercube and converts it to the
            physical parameters

        dumper: Optional function if we want to get some output while
            the chain is running. For now it's empty
        """
        # Write parameter names
        self.write_parnames()

        def log_lik(theta):
            """ Wrapper for likelihood. No derived for now """
            params = {}
            for i, name in enumerate(self.names):
                params[name] = theta[i]

            log_lik = self.log_lik(params)
            return log_lik, []

        def prior(hypercube):
            """ Uniform prior """
            prior = []
            for i, limits in enumerate(self.limits.values()):
                prior.append(UniformPrior(limits[0], limits[1])(hypercube[i]))
            return prior

        def dumper(live, dead, logweights, logZ, logZ_err):
            """ Dumper empty for now"""
            pass

        pypolychord.run_polychord(log_lik, self.num_params, self.num_derived,
                                  self.settings, prior, dumper)
