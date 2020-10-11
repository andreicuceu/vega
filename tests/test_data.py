import pytest
import numpy as np
import configparser
from astropy.io import fits

from vega.data import Data
from vega.utils import find_file
from vega import correlation_item


def test_data():
    test_config_path = find_file('configs/main.ini')

    # Read the main config file
    main_config = configparser.ConfigParser()
    main_config.optionxform = lambda option: option
    main_config.read(test_config_path)
    ini_files = main_config['data sets'].get('ini files').split()

    # Initialize the individual components and test each dataset
    for path in ini_files:
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option
        config.read(find_file(path))

        corr_item = correlation_item.CorrelationItem(config)

        data = Data(corr_item)
        hdul = fits.open(find_file(config['data']['filename']))

        assert np.allclose(data.data_vec, hdul[1].data['DA'])

        rp_rt_grid = corr_item.rp_rt_grid
        assert np.allclose(rp_rt_grid[0], hdul[1].data['RP'])
        assert np.allclose(rp_rt_grid[1], hdul[1].data['RT'])
        assert np.allclose(corr_item.z_grid, hdul[1].data['Z'])

        hdul.close()

        assert data.masked_data_vec is not None
