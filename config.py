import datetime
import logging.handlers
import os
from dataclasses import dataclass
from typing import Dict

from accsr.config import ConfigProviderBase, ConfigurationBase, DefaultDataConfiguration
from accsr.remote_storage import RemoteStorageConfig

log = logging.getLogger(__name__)

top_level_directory = os.path.dirname(__file__)


def recursive_dict_update(d: Dict, u: Dict):
    """
    Modifies d by overwriting with non-dict values and updating all dict-values recursively
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Configuration(DefaultDataConfiguration):
    """
    Holds essential configuration entries
    """

    log = log.getChild(__qualname__)

    def __init__(self, config_files=None):
        """
        :param config_files: list of JSON configuration files (relative to root) from which to read.
        """
        if config_files is None:
            config_files = [
                "./config.json",
                "./config_local.json",
                "/app/config/config_local.json",
            ]
        ConfigurationBase.__init__(self, config_files=config_files)

    @property
    def remote_storage(self):
        return RemoteStorageConfig(**self._get_non_empty_entry("remote_storage_config"))

    @property
    def development_remote_storage(self):
        return RemoteStorageConfig(
            **self._get_non_empty_entry("dev_remote_storage_config")
        )

    def viirs_file_path(
        self,
        date: datetime.date,
        tile="75N060W",
        stage="raw",
        relative=False,
        check_existence=False,
    ):
        """
        :param date: date of the viirs recording. Only month and year will be used,
            so 2019.01.01 gives the same path as 2019.01.30
        :param tile:
        :param stage: raw, ground_truth, cleaned or processed
        :param relative: If True, the returned path will be relative the project's top-level directory
        :param check_existence: if True, will raise an error when file does not exist
        """
        viirs_base_full_path = self.datafile_path(
            "nightlight_imagery", stage=stage, relative=False
        )
        image_full_path = os.path.join(
            viirs_base_full_path, f"{tile}/{date.strftime('%Y%m')}.tgz"
        )
        return self._adjusted_path(image_full_path, relative, check_existence)

    @property
    def viirs_repo_access(self):
        return ViirsRepoConfig(**self._get_non_empty_entry("viirs_repo_access"))

    def remote_storage_path(self, filename: str, stage="raw"):
        return self.datafile_path(
            filename=filename, stage=stage, relative=True, check_existence=False
        )

    def _data_basedir(self, stage):
        try:
            basedir = DefaultDataConfiguration._data_basedir(self, stage)
        except KeyError:
            raise KeyError(f"Unknown stage: {stage}")
        return basedir

@dataclass
class ViirsRepoConfig:
    username: str
    password: str

class __ConfigurationProvider(ConfigProviderBase[Configuration]):
    pass

_config_provider = __ConfigurationProvider()

def get_config(reload=False) -> Configuration:
    return _config_provider.get_config(reload)
