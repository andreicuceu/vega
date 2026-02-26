#!/usr/bin/env python
import argparse
from vega import compress_data

"""
Make compressed data vector

This script will take in data vectors (cf + xcf) and a global covariance matrix, and, 
using a Vega config file for the fiducial model, will output a compressed data vector.

Parameters
----------
data : str
    Path to input data vector.
cov : str
    Path to input global cov.
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
    parser.add_argument('--cf', type=str, required=True, help='Path to input cf data vector.')
    parser.add_argument('--xcf', type=str, required=True, help='Path to input xcf data vector.')
    parser.add_argument('--cov', type=str, required=True, help='Path to input global cov.')
    parser.add_argument('--outdir','-o', type=str, required=True, help='Path to output directory, where new configs' 
    'and compressed data vector will be saved.')
    parser.add_argument('--config','-i', type=str, required=True, help='Path to vega config file')
    parser.add_argument('--name', type=str, required=False, help='Optional name for output file')

    args = parser.parse_args()

    compress_data(args.config, args.cf, args.xcf, args.cov, args.outdir, name=args.name)