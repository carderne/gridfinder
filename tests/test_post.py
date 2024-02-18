from pathlib import Path

import numpy as np
import rasterio as rs

import gridfinder as gf


def test_threshold(p_dist: Path, p_guess: Path) -> None:
    res, affine = gf.threshold(p_dist, cutoff=0.0)
    with rs.open(p_guess) as ds:
        exp = ds.read(1)
        exp_affine = ds.transform
    assert np.array_equal(res, exp)
    assert affine == exp_affine


def test_thin(p_guess: Path, p_guess_thin: Path) -> None:
    res, affine = gf.thin(p_guess)
    with rs.open(p_guess_thin) as ds:
        exp = ds.read(1)
        exp_affine = ds.transform
    assert np.array_equal(res, exp)
    assert affine == exp_affine
