""" Defines reusable test fixtures. """

import os

import pytest

from config import top_level_directory


TEST_RESOURCES = os.path.join(top_level_directory, "tests", "resources")
DEFAULT_CRS = "EPSG:4326"


@pytest.fixture()
def test_resources() -> str:
    return TEST_RESOURCES


@pytest.fixture()
def default_crs() -> str:
    return DEFAULT_CRS
