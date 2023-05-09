import logging.handlers
from os.path import exists
from typing import List

from accsr.remote_storage import RemoteStorage
from libcloud.storage.types import ObjectDoesNotExistError

from config import get_config

log = logging.getLogger(__name__)

__default_remote_storage = None
__develop_remote_storage = None


def get_default_remote_storage():
    global __default_remote_storage
    c = get_config()
    if __default_remote_storage is None:
        __default_remote_storage = RemoteStorage(c.remote_storage)
    return __default_remote_storage


def get_develop_remote_storage():
    global __develop_remote_storage
    c = get_config()
    if __develop_remote_storage is None:
        __develop_remote_storage = RemoteStorage(c.development_remote_storage)
    return __develop_remote_storage


def download_files_from_gcp(
    all_filenames: List[str],
    remote_storage: RemoteStorage,
):
    """
    This method downloads all files from the GCP folder.

    :param all_filenames: A list of filenames specifying which files to download.
    :param remote_storage: The remote storage object from accsr.
    """

    for filename in all_filenames:

        # when the file does not exist check on GCP
        if not exists(filename):
            log.info(f"File not found. Check if raster {filename} is on GCP.")
            try:
                remote_storage.pull(filename, overwrite_existing=False)

            except ObjectDoesNotExistError:
                log.info(f"File {filename} not found on GCP.")


def upload_files_to_gcp(all_filenames: List[str], remote_storage: RemoteStorage):
    for filename in all_filenames:

        # when the file does not exist check on GCP
        if not exists(filename):
            raise FileNotFoundError(f"The file {filename} doesn't exist locally.")
        log.info(f"File was found locally uploading {filename} to GCP.")
        remote_storage.push_file(filename)
