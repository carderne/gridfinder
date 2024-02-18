from pathlib import Path

import numpy as np
import rasterio as rs
from geopandas import gpd

import gridfinder as gf


def test_prepare_ntl(p_ntl: Path, p_aoi: Path, p_targets: Path) -> None:
    res, res_affine = gf.prepare_ntl(p_ntl, p_aoi)
    with rs.open(p_targets) as ds:
        exp = ds.read(1)
        exp_affine = ds.transform
    assert np.array_equal(res, exp)
    assert res_affine == exp_affine


def test_drop_zero_pop(
    p_targets: Path, p_pop: Path, p_aoi: Path, p_targets_clean: Path
) -> None:
    aoi = gpd.read_file(p_aoi)
    res = gf.drop_zero_pop(p_targets, p_pop, aoi)
    with rs.open(p_targets_clean) as ds:
        exp = ds.read(1)
    assert np.array_equal(res, exp)


def test_prepare_roads(
    p_roads: Path,
    p_aoi: Path,
    p_targets_clean: Path,
    p_costs: Path,
) -> None:
    with rs.open(p_targets_clean) as ds:
        shape = ds.shape
        affine = ds.transform
    aoi = gpd.read_file(p_aoi)
    roads = gpd.read_file(p_roads)
    res = gf.prepare_roads(roads, aoi, shape, affine)
    with rs.open(p_costs) as ds:
        exp = ds.read(1)
    assert np.array_equal(res, exp)
