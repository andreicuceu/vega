"""Tests for catalog redshift weights and bias-evolution factors."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_RW_PATH = Path(__file__).resolve().parents[1] / 'vega' / 'redshift_weights.py'
_spec = importlib.util.spec_from_file_location('vega_redshift_weights', _RW_PATH)
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)


def test_weighted_mean_z():
    z = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 2.0])
    assert rw.weighted_mean_z(z, w) == pytest.approx(2.25)


def test_catalog_bias_evolution_at_pivot_is_unity_for_delta():
    # If all weight is at z_eff, F(alpha) == 1 for any alpha
    z = np.array([2.3, 2.3, 2.3])
    w = np.array([1.0, 2.0, 3.0])
    assert rw.catalog_bias_evolution_factor(z, w, alpha=1.44, z_eff=2.3) == pytest.approx(1.0)


def test_catalog_bias_evolution_hand_computation():
    z = np.array([1.8, 2.5, 3.2])
    w = np.array([1.0, 2.0, 1.0])
    alpha = 1.5
    z_eff = 2.4
    expected = np.sum(w * ((1 + z) / (1 + z_eff))**alpha) / np.sum(w)
    assert rw.catalog_bias_evolution_factor(z, w, alpha, z_eff) == pytest.approx(expected)


def test_catalog_bias_evolution_differs_from_mean_z_evaluation():
    # Broad n(z): Jensen inequality => <((1+z)/(1+zeff))^a> != ((1+<z>)/(1+zeff))^a
    z = np.array([1.8, 2.2, 2.8, 3.5])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    alpha = 1.44
    z_mean = rw.weighted_mean_z(z, w)
    f_cat = rw.catalog_bias_evolution_factor(z, w, alpha, z_mean)
    f_mean = ((1 + z_mean) / (1 + z_mean))**alpha
    assert f_mean == pytest.approx(1.0)
    assert f_cat != pytest.approx(1.0)


def test_rebin():
    v = np.arange(8, dtype=float)
    out = rw.rebin(v, 2)
    assert out.shape == (4,)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(2.5)
