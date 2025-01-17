import pypolychord
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings

from vega.samplers.sampler_interface import Sampler


class Polychord(Sampler):
    ''' Interface between Vega and the nested sampler PolyChord '''

    def __init__(self, sampler_config, limits, log_lik_func):
        super().__init__(sampler_config, limits, log_lik_func)

    def get_sampler_settings(self, sampler_config, num_params, num_derived):
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
        seed = sampler_config.getint('seed', int(0))

        # The key config parameters
        num_live = sampler_config.getint('num_live', int(25*num_params))
        num_repeats = sampler_config.getint('num_repeats', int(5*num_params))
        precision = sampler_config.getfloat('precision', float(0.001))

        # Resume should almost always be true
        resume = sampler_config.getboolean('resume', True)
        write_dead = sampler_config.getboolean('write_dead', True)

        # Useful for plotting as it gives you more posterior samples
        boost_posterior = sampler_config.getfloat('boost_posterior', float(0.0))

        # Do we do clustering, useful for multimodal distributions
        do_clustering = sampler_config.getboolean('do_clustering', False)
        cluster_posteriors = sampler_config.getboolean('cluster_posteriors', False)

        # Perform maximisation at the end of the chain
        maximise = sampler_config.getboolean('maximise', False)

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
        self.settings = PolyChordSettings(
            num_params, num_derived, base_dir=self.path,
            file_root=self.name, seed=seed, nlive=num_live,
            num_repeats=num_repeats,
            precision_criterion=precision,
            write_resume=resume, read_resume=resume,
            boost_posterior=boost_posterior,
            do_clustering=do_clustering,
            cluster_posteriors=cluster_posteriors,
            equals=False, write_dead=write_dead,
            maximise=maximise,
            write_live=False, write_prior=False
        )

    def run(self):
        """Run Polychord. We need to pass three functions:

        log_lik: takes a list of parameter values and
            returns tuple: (log_lik, list of derived)

        prior: takes a unit hypercube and converts it to the
            physical parameters

        dumper: Optional function if we want to get some output while
            the chain is running. For now it's empty
        """
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

        pypolychord.run_polychord(
            log_lik, self.num_params, self.num_derived, self.settings, prior, dumper)
