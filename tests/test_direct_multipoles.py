"""Integration tests for direct QSO multipole data"""

import configparser

import numpy as np
import pytest

from vega import correlation_item
from vega.coordinates import MultipoleCoordinates
from vega.data import Data
from vega.utils import find_file


def _load_qsoxqso():
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(find_file("configs/qsoxqso.ini"))
    corr_item = correlation_item.CorrelationItem(config)
    for tracer in (corr_item.tracer1, corr_item.tracer2):
        if tracer.get("weights-path") is not None:
            tracer["weights-path"] = str(find_file(tracer["weights-path"]))
    corr_item.z_eff = 2.304
    data = Data(corr_item)
    return corr_item, data


def test_read_multipole_data_vector_and_coordinates():
    corr_item, data = _load_qsoxqso()

    assert data.is_direct_multipoles
    assert not corr_item.use_multipoles
    assert corr_item.ells_to_model == [0, 2]

    raw = np.loadtxt(find_file("data/qsoauto_xipoles.txt"), comments="#")
    s_min, s_max = 50.0, 90.0
    mask = (raw[:, 0] >= s_min) & (raw[:, 0] < s_max)
    xi_cut = raw[mask, 2:4]
    expected_vec = np.concatenate([xi_cut[:, 0], xi_cut[:, 1]])

    assert data.data_vec.shape == (20,)
    assert np.allclose(data.data_vec, expected_vec)
    assert data.full_data_size == 20
    assert data.nells == 2
    assert np.all(data.data_mask)

    assert isinstance(data.data_coordinates, MultipoleCoordinates)
    assert data.data_coordinates.ells == [0, 2]
    assert data.data_coordinates.s_grid.shape == (10,)
    assert np.all(data.data_coordinates.s_grid >= s_min)
    assert np.all(data.data_coordinates.s_grid < s_max)

    assert data.r_min_cut == s_min
    assert data.r_max_cut == s_max
    assert data.use_multipoles
    assert data._multipole_matrix is not None
    assert data._multipole_matrix.shape == (20, 10 * 100)

    assert corr_item.z_eff_QSO == pytest.approx(2.376196, rel=1e-5)
    assert np.all(data.data_coordinates.z_grid == pytest.approx(corr_item.z_eff_QSO))

    assert data._mp_n_s_full == 10
    assert data._mp_n_ells_file == 2
    assert data._mp_ell_file_indices == [0, 1]
    assert np.all(data._mp_full_s_mask)


def test_multipole_covariance_subset():
    corr_item, data = _load_qsoxqso()

    assert data.cov_mat.shape == (20, 20)
    assert np.allclose(data.cov_mat, data.cov_mat.T)

    cov_full = np.loadtxt(find_file("data/qsoauto_cov_20x20.txt"), comments="#")
    s_data = data.data_coordinates.s_grid
    s_min, s_max = 50.0, 90.0
    n_s_cov = cov_full.shape[0] // 2
    ds_cov = (s_max - s_min) / n_s_cov
    s_cov_centers = s_min + (np.arange(n_s_cov) + 0.5) * ds_cov
    cov_idx = np.array([np.argmin(np.abs(s_cov_centers - sv)) for sv in s_data])
    expected_idx = np.concatenate([cov_idx, cov_idx + n_s_cov])
    expected_cov = cov_full[np.ix_(expected_idx, expected_idx)]

    assert np.allclose(data.cov_mat, expected_cov)
    assert data.masked_data_vec is not None
    assert data.inv_masked_cov is not None
