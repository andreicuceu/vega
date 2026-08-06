#!/usr/bin/env python
import numpy as np
from vega import VegaInterface
from vega.minimizer import Minimizer
from mpi4py import MPI
from astropy.io import fits
import argparse
import sys


def run_monte_carlo(
    self, mocks, start1=None, end1=None, start2=None, end2=None
):
    """Run Monte Carlo simulations

    Parameters
    ----------
    fiducial_model : dict
        Fiducial model for the correlation functions
    num_mocks : int, optional
        Number of mocks to run, by default 1
    seed : int, optional
        Starting seed, by default 0
    scale : float/dict, optional
        Scaling for the covariance, by default None
    """
    assert self.mc_config is not None, 'No Monte Carlo config provided'

    sample_params = self.mc_config['sample']
    minimizer = Minimizer(self._chi2_func, sample_params)

    self.mc_bestfits = {}
    self.mc_covariances = []
    self.mc_chisq = []
    self.mc_valid_minima = []
    self.mc_valid_hesse = []
    self.mc_mocks = {'global': []}
    self.mc_failed_mask = []

    for i, mock in enumerate(mocks):
        if start1 is None or end1 is None or start2 is None or end2 is None:
            self.current_mc_mock = mock
            self.mc_mocks['global'].append(mock)
        else:
            slice1 = mock[start1:end1]
            slice2 = mock[start2:end2]
            sliced_mock = np.r_[slice1, slice2]
            self.current_mc_mock = sliced_mock
            self.mc_mocks['global'].append(sliced_mock)

        try:
            # Run minimizer
            minimizer.minimize()
            self.mc_failed_mask.append(False)
        except ValueError:
            print('WARNING: Minimizer failed for mock {}'.format(i))
            self.mc_failed_mask.append(True)
            self.mc_chisq.append(np.nan)
            self.mc_valid_minima.append(False)
            self.mc_valid_hesse.append(False)
            continue

        sys.stdout.flush()

        for param, value in minimizer.values.items():
            if param not in self.mc_bestfits:
                self.mc_bestfits[param] = []
            self.mc_bestfits[param].append([value, minimizer.errors[param]])

        self.mc_covariances.append(minimizer.covariance)
        self.mc_chisq.append(minimizer.fmin.fval)
        self.mc_valid_minima.append(minimizer.fmin.is_valid)
        self.mc_valid_hesse.append(not minimizer.fmin.hesse_failed)

    for param in self.mc_bestfits.keys():
        self.mc_bestfits[param] = np.array(self.mc_bestfits[param])

    self.has_monte_carlo = True


if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    mpi_comm = MPI.COMM_WORLD
    cpu_rank = mpi_comm.Get_rank()
    num_cpus = mpi_comm.Get_size()

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
    sys.stdout.flush()

    # Check if we need the distortion
    use_distortion = vega.main_config['control'].getboolean('use_distortion', True)
    if not use_distortion:
        for key, data in vega.data.items():
            data._distortion_mat = None
        test_model = vega.compute_model(vega.params, run_init=True)

    # Run monte carlo
    run_montecarlo = vega.main_config['control'].getboolean('run_montecarlo', False)
    if not run_montecarlo or (vega.mc_config is None):
        raise ValueError(
            'Warning: You called "run_vega_mc_fits_mpi.py" without asking'
            ' for monte carlo. Add "run_montecarlo = True" to the "[control]" section.'
        )

    # Activate monte carlo mode
    vega.monte_carlo = True

    # Get the MC seed and the number of mocks to run
    # seed = vega.main_config['control'].getint('mc_seed', 0)
    # num_mc_mocks = vega.main_config['control'].getint('num_mc_mocks', 1)
    mock_path = vega.main_config['control'].get('mc_mocks')
    with fits.open(mock_path) as hdul:
        mocks = hdul['MOCKS'].data['global']

    num_tasks_per_proc = mocks.shape[0] // num_cpus
    remainder = mocks.shape[0] % num_cpus
    if cpu_rank < remainder:
        start = int(cpu_rank * (num_tasks_per_proc + 1))
        stop = int(start + num_tasks_per_proc + 1)
    else:
        start = int(cpu_rank * num_tasks_per_proc + remainder)
        stop = int(start + num_tasks_per_proc)

    # Get slice values
    slice_start1 = vega.main_config['control'].getint('slice_start1')
    slice_end1 = vega.main_config['control'].getint('slice_end1')
    slice_start2 = vega.main_config['control'].getint('slice_start2')
    slice_end2 = vega.main_config['control'].getint('slice_end2')

    # Run the mocks
    print(f'Proc #{cpu_rank} running MC mocks: {start} to {stop}')
    sys.stdout.flush()
    run_monte_carlo(
        vega.analysis, mocks[start:stop], slice_start1, slice_end1, slice_start2, slice_end2
    )

    # Write output
    if num_cpus > 1:
        vega.output.write_monte_carlo(cpu_rank)
    else:
        vega.output.write_monte_carlo()
