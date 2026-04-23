#!/usr/bin/env python
import argparse
from vega import compress_data

"""
This script will take a Vega config and compute the CCA matrix from the supplied data-param and param-param covariance matrices.

Parameters
----------
data : str
    Path to input config file.
outdir : str
    Path to output file.
name : str, optional
    Optional suffix for output file name.

Returns
-------
None

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make compressed data vector')
    parser.add_argument('--config','-i', type=str, required=True, help='Path to vega config file')
    parser.add_argument('--outdir','-o', type=str, required=True, help='Path to output directory, where new configs' 
    'and compressed data vector will be saved.')
    parser.add_argument('--name', type=str, required=False, help='Optional name for output file')

    args = parser.parse_args()

    # Read the config files
    cfg_parser = configparser.ConfigParser()
    cfg_parser.optionxform = lambda option: option

    #load config
    cfg_parser.read(args.config)

    #init vega
    vega = VegaInterface(main_cfg_path)
