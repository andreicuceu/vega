import argparse
import configparser
from vega import VegaInterface
import os
import numpy as np

def compress_data(config_path, cf_file=None, xcf_file=None, global_cov_file=None, outdir=None):
    """Compute a compressed covariance matrix from input covariance matrix + data vectors.

    Parameters
    ----------
    config : str
        Path to vega config file
    cov : str
        Path to input covariance matrix
    outdir : str
        Path to output compressed covariance matrix

    Returns
    -------
    None
    """
    assert os.path.isfile(config_path), 'Config file does not exist'
    config_templates = '/global/cfs/projectdirs/desi/users/cgordon/DESI/DR2/compression/configs/template'
    main_cfg = config_templates + '/main.ini'
    auto_cfg = config_templates + '/lyaxlya.ini'
    cross_cfg = config_templates + '/lyaxqso.ini'
    print('INFO: using config files in: ', config_templates)
    print('INFO: If you want to change masks, fiducial models etc.'
                'use a different config setup')

    # Get the config file paths
    auto_cfg_path = os.path.join(outdir, 'lyaxlya.ini')
    cross_cfg_path = os.path.join(outdir, 'lyaxqso.ini')
    main_cfg_path = os.path.join(outdir, 'main.ini')

    # Read the config files
    auto_cfg_parser = configparser.ConfigParser()
    auto_cfg_parser.optionxform = lambda option: option
    cross_cfg_parser = configparser.ConfigParser()
    cross_cfg_parser.optionxform = lambda option: option
    main_cfg_parser = configparser.ConfigParser()
    main_cfg_parser.optionxform = lambda option: option

    auto_cfg_parser.read(auto_cfg)
    cross_cfg_parser.read(cross_cfg)
    main_cfg_parser.read(main_cfg)

    # Change data options
    auto_cfg_parser['data']['filename'] = cf_file
    cross_cfg_parser['data']['filename'] = xcf_file

    # Change global covariance file
    main_cfg_parser['data sets']['ini files'] = auto_cfg_path + ' ' + cross_cfg_path
    main_cfg_parser['data sets']['global-cov-file'] = global_cov_file

    #check if outdir exists, if not create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(auto_cfg_path, 'w') as f:
        auto_cfg_parser.write(f, space_around_delimiters=True)
    with open(cross_cfg_path, 'w') as f:
        cross_cfg_parser.write(f, space_around_delimiters=True)
    with open(main_cfg_path, 'w') as f:
        main_cfg_parser.write(f, space_around_delimiters=True)

    # Initialize Vega
    vega = VegaInterface(main_cfg_path)

    _xi_compressed = vega.score

    print('Writing compressed data vector to: ', os.path.join(outdir, 'xi_compressed.npz'))
    # Save the compressed covariance matrix as npz file
    np.savez(os.path.join(outdir, 'xi_compressed.npz'), xi_t = _xi_compressed)
