#!/usr/bin/env python
import argparse

from vega.scripts.run_vega_sampler import run_vega_mpi

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega in parallel.')

    pars.add_argument('config', type=str, help='Config file')
    args = pars.parse_args()

    _ = run_vega_mpi(args.config, mpi=True)
