#!/usr/bin/env python
import argparse

from vega import run_vega

if __name__ == '__main__':
    pars = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run Vega.')

    pars.add_argument('config', type=str, default=None, help='Config file')
    args = pars.parse_args()

    run_vega(args.config)
