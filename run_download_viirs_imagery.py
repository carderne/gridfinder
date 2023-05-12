import datetime
import logging
import os
import os.path
import tarfile
from contextlib import ExitStack
from typing import Callable, List, Optional, Union

import click
from accsr.remote_storage import RemoteStorage

from config import get_config
from src.gridlight.prepare import combine_rasters_into_single_file
from src.gridlight.util.loading import open_raster_in_tar
from src.gridlight.util.viirs import download_viirs_imagery

log = logging.getLogger(__name__)


def make_tar_archive(
    tar_archive_name: str,
    files_to_tar: List[Union[str, os.PathLike]],
    arcnamer: Optional[Callable] = None,
):
    """
    Tars multiple files into a .gz archive.

    :param tar_archive_name: Local path of the archive.
    :param files_to_tar: Local path to file that you want to put into the archive
    :param arcnamer: A function with the signature (str) -> str,
     which maps the file name to the name in the tar archive.
     If None, the file name will be used
    """

    with tarfile.open(tar_archive_name, "w:gz") as tar:
        for name in files_to_tar:
            tar.add(name, arcname=name if arcnamer is None else arcnamer(name))


@click.command()
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d", "%y%m%d"]),
    default="2020-01-01",
    help="Start date of the range defining the files which shall be downloaded.",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d", "%y%m%d"]),
    default="2020-02-28",
    help="End date of the range defining the files which shall be downloaded.",
)
@click.option(
    "--tiles",
    "-t",
    multiple=True,
    default=["75N060W", "00N060W"],
    type=str,
    help="For each time step, configure which tiles shall be downloaded. "
    "Note that in case of multiple tile they will be merged "
    "into one tif file prior to upload.",
)
@click.option(
    "--dev-mode",
    "-d",
    is_flag=True,
    default=False,
    help="Execute script in development mode. "
    "If true, the files will be pushed to the development remote storage.",
)
def viirs_imagery_to_remote_storage(
    start_date: datetime.date, end_date: datetime.date, tiles: List[str], dev_mode: bool
):
    c = get_config()
    storage = RemoteStorage(
        c.development_remote_storage if dev_mode else c.remote_storage
    )

    downloaded_files = download_viirs_imagery(
        start_date=start_date,
        end_date=end_date,
        tiles=tiles,
        viirs_credentials=c.viirs_repo_access,
    )

    if len(tiles) > 1:
        log.info(f"Merging tiles {tiles} into single file for each time step")

        # for each time step, get all tiles and merge them. Save the
        # output in a new folder coding the included tile in the folder name
        tiles_code = "-".join(tiles)
        output_folder = c.datafile_path(
            f"nightlight_imagery/{tiles_code}",
            stage=c.PROCESSED,
            relative=True,
            check_existence=False,
        )
        os.makedirs(output_folder, exist_ok=True)
        filename_to_tar_archive = {}
        for date, tiles in downloaded_files.items():
            log.info(f"Processing date {date}")
            with ExitStack() as stack:
                rasters = [
                    stack.enter_context(open_raster_in_tar(tile)) for tile in tiles
                ]
                output_file = os.path.join(output_folder, f"{date}.tif")
                combine_rasters_into_single_file(rasters, output_file=output_file)
                filename_to_tar_archive[output_file] = os.path.join(
                    output_folder, f"{date}.tgz"
                )

        for tif_file, tar_archive in filename_to_tar_archive.items():
            make_tar_archive(
                tar_archive, [tif_file], arcnamer=lambda name: name.split("/")[-1]
            )
            storage.push(tar_archive)

    else:
        files = [f_path for sublist in downloaded_files.values() for f_path in sublist]
        for f in files:
            storage.push(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    viirs_imagery_to_remote_storage()
