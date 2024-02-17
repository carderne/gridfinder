from pathlib import Path

import pytest

p_data = Path(__file__).parents[0] / "data"


@pytest.fixture()
def p_ntl() -> Path:
    return p_data / "ntl.tif"


@pytest.fixture()
def p_aoi() -> Path:
    return p_data / "aoi.geojson"


@pytest.fixture()
def p_targets() -> Path:
    return p_data / "targets.tif"


@pytest.fixture()
def p_targets_clean() -> Path:
    return p_data / "targets_clean.tif"


@pytest.fixture()
def p_pop() -> Path:
    return p_data / "pop.tif"


@pytest.fixture()
def p_roads() -> Path:
    return p_data / "roads.geojson"


@pytest.fixture()
def p_costs() -> Path:
    return p_data / "costs.tif"


@pytest.fixture()
def p_dist() -> Path:
    return p_data / "dist.tif"


@pytest.fixture()
def p_guess() -> Path:
    return p_data / "guess.tif"


@pytest.fixture()
def p_guess_thin() -> Path:
    return p_data / "thin.tif"
