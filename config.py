import json
import logging.handlers
import os
from pathlib import Path
from typing import List, Union, Dict


log = logging.getLogger(__name__)

__config_instance = None

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


class __Configuration:
    """
    Holds essential configuration entries
    """

    log = log.getChild(__qualname__)

    PROCESSED = "processed"
    RAW = "raw"
    CLEANED = "cleaned"
    GROUND_TRUTH = "ground_truth"
    DATA = "data"

    def __init__(self, config_files: List[str] = None):
        """
        :param config_files: list of JSON configuration files (relative to root) from which to read.
            If None, reads from './config.json' and './config_local.json' (latter files have precedence)
        """
        if config_files is None:
            config_files = ["config.json", "config_local.json"]
        self.config = {}
        for filename in config_files:
            file_path = os.path.join(top_level_directory, filename)
            if os.path.exists(file_path):
                self.log.info("Reading configuration from %s" % file_path)
                with open(file_path, "r") as f:
                    recursive_dict_update(self.config, json.load(f))
        if not self.config:
            raise Exception(
                "No configuration entries could be read from %s" % config_files
            )

    def _get_non_empty_entry(
        self, key: Union[str, List[str]]
    ) -> Union[float, str, List, Dict]:
        """
        Retrieves an entry from the configuration

        :param key: key or list of keys to go through hierarchically
        :return: the queried json object
        """
        if isinstance(key, str):
            key = [key]
        value = self.config
        for k in key:
            value = value.get(k)
            if value is None:
                raise Exception(f"Value for key '{key}' not set in configuration")
        return value

    def _get_existing_path(self, key: Union[str, List[str]], create=False) -> str:
        """
        Retrieves an existing local path from the configuration

        :param key: key or list of keys to go through hierarchically
        :param create: if True, a directory with the given path will be created on the fly.
        :return: the queried path
        """
        path_string = self._get_non_empty_entry(key)
        path = os.path.abspath(path_string)
        if not os.path.exists(path):
            if isinstance(key, list):
                key = ".".join(key)  # purely for logging
            if create:
                log.info(
                    f"Configured directory {key}='{path}' not found; will create it"
                )
                os.makedirs(path)
            else:
                raise FileNotFoundError(
                    f"Configured directory {key}='{path}' does not exist."
                )
        return path.replace("/", os.sep)

    @property
    def artifacts(self):
        return self._get_existing_path("artifacts", create=True)

    @property
    def visualizations(self):
        return self._get_existing_path("visualizations", create=True)

    @property
    def temp(self):
        return self._get_existing_path("temp", create=True)

    @property
    def data(self):
        return self._get_existing_path("data")

    @property
    def data_raw(self):
        return self._get_existing_path("data_raw")

    @property
    def data_cleaned(self):
        return self._get_existing_path("data_cleaned", create=True)

    @property
    def data_processed(self):
        return self._get_existing_path("data_processed", create=True)

    @property
    def data_ground_truth(self):
        return self._get_existing_path("data_ground_truth")

    def datafile_path(
        self, filename: str, stage="raw", relative=False, check_existence=True
    ):
        """
        :param filename:
        :param stage: raw, ground_truth, cleaned or processed
        :param relative: If True, the returned path will be relative the project's top-level directory
        :param check_existence: if True, will raise an error when file does not exist
        """
        basedir = self._data_basedir(stage)
        full_path = os.path.join(basedir, filename)
        return self._adjusted_path(full_path, relative, check_existence)

    @staticmethod
    def _adjusted_path(path: str, relative: bool, check_existence: bool):
        """
        :param path:
        :param relative: If true, the returned path will be relative the project's top-level directory.
        :param check_existence: if True, will raise an error when file does not exist
        :return: the adjusted path, either absolute or relative
        """
        path = os.path.abspath(path)
        if check_existence and not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")
        if relative:
            return str(Path(path).relative_to(top_level_directory))
        return path

    def _data_basedir(self, stage):
        if stage == self.RAW:
            basedir = self.data_raw
        elif stage == self.CLEANED:
            basedir = self.data_cleaned
        elif stage == self.PROCESSED:
            basedir = self.data_processed
        elif stage == self.GROUND_TRUTH:
            basedir = self.data_ground_truth
        else:
            raise KeyError(f"Unknown stage: {stage}")
        return basedir

    def artifact_path(self, name: str, relative=False, check_existence=True):
        """
        :param name:
        :param relative: If true, the returned path will be relative the project's top-level directory.
        :param check_existence: if True, will raise an error when file does not exist
        :return:
        """
        full_path = os.path.join(self.artifacts, name)
        return self._adjusted_path(full_path, relative, check_existence)


def get_config(reload=False) -> __Configuration:
    """
    :param reload: if True, the configuration will be reloaded from the json files
    :return: the configuration instance
    """
    global __config_instance
    if __config_instance is None or reload:
        __config_instance = __Configuration()
    return __config_instance
