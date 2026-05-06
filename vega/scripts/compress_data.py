import argparse
import configparser
from vega import VegaInterface
import os
import numpy as np

def compress_data(config_path, cf_file, xcf_file, outdir, name=None):
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
    assert os.path.isdir(config_path), 'Config file does not exist'
    # config_templates = '/global/cfs/projectdirs/desi/users/cgordon/DESI/DR2/compression/configs/template'
    main_cfg = config_path + '/main.ini'
    auto_cfg = config_path + '/lyaxlya.ini'
    cross_cfg = config_path + '/lyaxqso.ini'
    print('INFO: using config files in: ', config_path)
    print('INFO: If you want to change masks, fiducial models etc.'
                'use a different config setup')

    # Set the new config file paths
    auto_cfg_path = os.path.join(outdir, 'lyaxlya.ini')
    cross_cfg_path = os.path.join(outdir, 'lyaxqso.ini')
    main_cfg_path = os.path.join(outdir, 'main.ini')

    # for f in [auto_cfg_path, cross_cfg_path, main_cfg_path]:
    #     if not os.path.exists(f):
    #         raise FileNotFoundError(f'File {f} does not exist')

    # Read the template config files
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
    # main_cfg_parser['data sets']['global-cov-file'] = global_cov_file

    #check if outdir exists, if not create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Write the new config files
    with open(auto_cfg_path, 'w') as f:
        auto_cfg_parser.write(f, space_around_delimiters=True)
    with open(cross_cfg_path, 'w') as f:
        cross_cfg_parser.write(f, space_around_delimiters=True)
    with open(main_cfg_path, 'w') as f:
        main_cfg_parser.write(f, space_around_delimiters=True)

    # Initialize Vega
    vi = VegaInterface(main_cfg_path)

    # Compress the data vector
    _xi_compressed = vi.compress(vi._full_datavec)

    if name is not None:
        name = '_' + name

    print('Writing compressed data vector to: ', os.path.join(outdir, 'xi_compressed{}.npz'.format(name)))
    # Save the compressed covariance matrix as npz file
    np.savez(os.path.join(outdir, 'xi_compressed{}.npz'.format(name)), xi_t = _xi_compressed)