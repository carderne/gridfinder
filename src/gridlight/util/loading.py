import logging
import re
from contextlib import contextmanager
from os import PathLike
from typing import Union

import rasterio
from accsr.loading import open_file_in_tar

log = logging.getLogger(__name__)

VIIRS_TAR_FILE_REGEX = "^.+\.avg_rade9h.tif$"


@contextmanager
def open_raster_in_tar(
    path: Union[str, PathLike],
    file_regex: Union[str, re.Pattern] = VIIRS_TAR_FILE_REGEX,
):
    with open_file_in_tar(path, file_regex) as raster_in_bytes:
        with rasterio.open(raster_in_bytes) as raster:
            yield raster
