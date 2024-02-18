from pathlib import Path

import numpy as np
import rasterio as rs

import gridfinder as gf


def test_optimise(p_targets_clean: Path, p_costs: Path, p_dist: Path) -> None:
    targets, costs, start, _ = gf.get_targets_costs(p_targets_clean, p_costs)
    res = gf.optimise(targets, costs, start)
    with rs.open(p_dist) as ds:
        exp = ds.read(1)
    assert np.array_equal(res, exp)
