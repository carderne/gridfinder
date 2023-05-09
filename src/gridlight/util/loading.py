import logging
import tarfile
from contextlib import contextmanager
from io import BufferedReader

import rasterio
from rasterio import DatasetReader

log = logging.getLogger(__name__)


# TODO: remove this once we have separated these utils from the scripts repo
@contextmanager
def open_file_in_tar(
    path: str, file_name: str = None, file_index: int = None
) -> BufferedReader:
    """
    Opens an archived file in memory without extracting it on disc.

    :param path:
    :param file_name:
    :param file_index: 1-based index of the file to retrieve
    :return:
    """
    if file_name is not None and file_index is not None:
        raise ValueError("Either file_name or file_index should be passed; not both")
    if file_name is None and file_index is None:
        raise ValueError("One of file_name or file_index has to be passed")

    with tarfile.open(path) as tar:
        archived_files = tar.getnames()
        if file_index is not None:
            if file_index < 1:
                raise IndexError(
                    f"Invalid index {file_index}. NOTE: the parameter file_index is 1 based"
                )
            file_name = archived_files[file_index - 1]  # tar uses 1-based indices
        # tar.extractfile returns None for non-existing files, so we have to raise the Exception ourselves
        elif file_name not in archived_files:
            raise FileNotFoundError(f"No such file in {path}: {file_name}")
        log.debug(f"Yielding {file_name} from {path}")

        with tar.extractfile(file_name) as file:
            yield file


@contextmanager
def open_raster_in_tar(
    path: str, file_name: str = None, file_index: int = None
) -> DatasetReader:
    with open_file_in_tar(
        path, file_name=file_name, file_index=file_index
    ) as raster_in_bytes:
        with rasterio.open(raster_in_bytes) as raster:
            yield raster
