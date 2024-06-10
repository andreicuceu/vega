import os.path
import sys
from pathlib import Path

from mpi4py import MPI

from vega.parameters.param_utils import build_names


class Sampler:
    ''' Interface between Vega and the nested sampler PolyChord '''

    def __init__(self, sampler_config, limits, log_lik_func):
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

    def run(self):
        raise NotImplementedError('This method should be implemented in the child class')
