import os.path
import sys
from pathlib import Path

from mpi4py import MPI

from vega import VegaInterface
from vega.parameters.param_utils import build_names


class Sampler:
    ''' Interface between Vega and the nested sampler PolyChord '''

    def __init__(self, args):
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
        self.mpi_comm = MPI.COMM_WORLD
        self.cpu_rank = self.mpi_comm.Get_rank()

        self.print_func('Initializing Vega')

        # Initialize Vega and get the sampling parameters
        self.vega = VegaInterface(args.config)
        sampling_params = self.vega.sample_params['limits']

        self.print_func('Finished initializing Vega')

        # Check if we need to run over a Monte Carlo mock
        run_montecarlo = self.vega.main_config['control'].getboolean('run_montecarlo', False)
        if run_montecarlo and self.vega.mc_config is not None:
            # Get the MC seed and forecast flag
            seed = self.vega.main_config['control'].getint('mc_seed', 0)
            forecast = self.vega.main_config['control'].getboolean('forecast', False)

            # Create the mocks
            self.vega.monte_carlo_sim(self.vega.mc_config['params'], seed=seed, forecast=forecast)

            # Set to sample the MC params
            sampling_params = self.vega.mc_config['sample']['limits']
            self.print_func('Created Monte Carlo realization of the correlation')
        elif run_montecarlo:
            raise ValueError('You asked to run over a Monte Carlo simulation,'
                             ' but no "[monte carlo]" section provided.')

        # Run sampler
        if not self.vega.run_sampler:
            raise ValueError(
                'Warning: You called "run_vega_mpi.py" without asking'
                ' for the sampler. Add "run_sampler = True" to the "[control]" section.'
            )

        self.limits = sampling_params
        self.names = list(sampling_params.keys())
        self.num_params = len(sampling_params)
        self.num_derived = 0
        # self.log_lik = log_lik_func

        if self.vega.sampler == 'Polychord':
            sampler_config = self.vega.main_config['Polychord']
        elif self.vega.sampler == 'PocoMC':
            sampler_config = self.vega.main_config['PocoMC']

        self.getdist_latex = sampler_config.getboolean('getdist_latex', True)

        # Check limits are well defined
        for lims in self.limits.values():
            if None in lims:
                raise ValueError('Sampler needs well defined prior limits.'
                                 ' You passed a None. Please give numbers, or'
                                 ' just say par_name = True to use defaults.')

        # Check the path and get the paramnames path
        self.path = os.path.expandvars(sampler_config.get('path'))
        self.name = sampler_config.get('name')

        output_path = Path(self.path)
        err_msg = ("The sampler 'path' does not correspond to an existing"
                   " folder. Create the output folder before running.")
        assert output_path.exists(), err_msg
        parnames_path = output_path / (self.name + '.paramnames')

        # Write parameter names
        self.write_parnames(parnames_path)

        # Initialize the sampler settings
        self.get_sampler_settings(sampler_config, self.num_params, self.num_derived)

    def print_func(self, message):
        if self.cpu_rank == 0:
            print(message)
        sys.stdout.flush()

    def write_parnames(self, parnames_path):
        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()

        if cpu_rank == 0:
            print('Writing parameter names')
            sys.stdout.flush()
            latex_names = build_names(list(self.names))
            with open(parnames_path, 'w') as f:
                for name, latex in latex_names.items():
                    if self.getdist_latex:
                        f.write('%s    %s\n' % (name, latex))
                    else:
                        f.write('%s    $%s$\n' % (name, latex))
            print('Finished writing parameter names')
            sys.stdout.flush()

        mpi_comm.barrier()

    def get_sampler_settings(self, sampler_config, num_params, num_derived):
        raise NotImplementedError('This method should be implemented in the child class')

    def run(self, log_lik_func):
        raise NotImplementedError('This method should be implemented in the child class')
