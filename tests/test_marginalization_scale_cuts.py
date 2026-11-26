"""Tests of the bins selected for small-scale marginalization by marginalize-all-rmin-cuts."""

import configparser

import numpy as np
import pytest

from vega.coordinates import RtRpCoordinates


def _cuts(**cuts):
    """[cuts] section of a cross-correlation with r-max = 200 and the given extra cuts."""
    config = configparser.ConfigParser()
    base_cuts = {"rp-min": "-300", "rp-max": "300", "rt-max": "200", "r-max": "200"}
    config.read_dict({"cuts": {**base_cuts, **cuts}})
    return config["cuts"]


@pytest.fixture(scope="module")
def cross_grids():
    """Data and distorted-model grids of a Lyα×QSO cross: |r_par| < 200 and < 300, 4 Mpc/h bins."""
    data = RtRpCoordinates(-200.0, 200.0, 200.0, 100, 50)
    dist_model = RtRpCoordinates(-300.0, 300.0, 200.0, 150, 50)
    return data, dist_model


@pytest.mark.parametrize(
    "extra_cuts",
    [{"r-min": "30"}, {"r-min": "30", "rt-min": "8"}, {"r-min": "20", "rt-min": "16"}],
)
def test_data_and_model_marginalize_the_same_bins(cross_grids, extra_cuts):
    """The marginalized bins are the same separations on the data and distorted-model grids."""
    data, dist_model = cross_grids
    cuts = _cuts(**extra_cuts)
    marginalization = {"all-rmin": True}
    data_mask = data.get_mask_marginalization_scales(cuts, marginalization)
    model_mask = dist_model.get_mask_marginalization_scales(cuts, marginalization)
    assert data_mask.sum() == model_mask.sum()

    # Every marginalized bin is inside the large-scale cuts and removed by a small-scale cut
    rt_min = float(extra_cuts.get("rt-min", 0.0))
    r_min = float(extra_cuts["r-min"])
    for grid, mask in ((data, data_mask), (dist_model, model_mask)):
        assert np.all(grid.r_regular_grid[mask] < 200.0)
        assert np.all((grid.r_regular_grid[mask] <= r_min) | (grid.rt_regular_grid[mask] <= rt_min))


def test_restriction_is_a_no_op_for_r_min_alone(cross_grids):
    """With r-min alone the marginalized bins are exactly those with r <= r-min."""
    data, dist_model = cross_grids
    cuts = _cuts(**{"r-min": "30"})
    for grid in (data, dist_model):
        mask = grid.get_mask_marginalization_scales(cuts, {"all-rmin": True})
        assert np.array_equal(mask, grid.r_regular_grid <= 30.0)
