"""
Utilities for retrieving viirs nightlight imagery. The website where we download them from has
specific conventions about formatting dates and tile names, the methods here adhere to these conventions.
"""
import collections
import datetime
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import requests
from accsr.loading import download_file
from bs4 import BeautifulSoup
from dateutil import rrule
from keycloak import KeycloakOpenID

from config import ViirsRepoConfig, get_config

log = logging.getLogger(__name__)


def _year_month(date: datetime.date):
    return date.strftime("%Y%m")


def _validate_tile(tile: str) -> None:
    tile_regex = re.compile(r"^\d\dN\d\d\d[OWE]$")
    if not re.match(tile_regex, tile):
        raise ValueError(f"Invalid tile: {tile}")


def _file_regex(date: datetime.date, tile: str):
    _validate_tile(tile)
    return re.compile(
        rf"^SVDNB_npp_{_year_month(date)}01-{_year_month(date)}\d\d_{tile}_vcmcfg_v10_c\d+\.tgz$"
    )


class ViirsAuthenticator:
    """
    Abstracts the authentication process to the nasa repo.
    You can register to the nasa repo under

    https://eogdata.mines.edu/eog/EOG_sensitive_contents

    >>> authenticator = ViirsAuthenticator(username="foo", password="bar")
    >>> token = authenticator.token

    The received token can be used to make an authorized request to the nasa repo
    using the http request header field `Authorization`.
    """

    # Source: https://eogdata.mines.edu/products/register/
    CLIENT_ID = "eogdata_oidc"
    CLIENT_SECRET_KEY = "2677ad81-521b-4869-8480-6d05b9e57d48"
    AUTH_SERVER_URL = "https://eogauth.mines.edu/auth/"
    AUTH_REALM_NAME = "master"

    def __init__(self, username: str, password: str):
        self.keycloak_openid = KeycloakOpenID(
            server_url=ViirsAuthenticator.AUTH_SERVER_URL,
            client_id=ViirsAuthenticator.CLIENT_ID,
            realm_name=ViirsAuthenticator.AUTH_REALM_NAME,
            client_secret_key=ViirsAuthenticator.CLIENT_SECRET_KEY,
        )
        self.username = username
        self.password = password

    @property
    def token(self) -> str:
        return self.keycloak_openid.token(self.username, self.password)["access_token"]


class ViirsDatafileIterator:
    """
    This iterator takes care of the url construction for the viirs imagery files.
    Given a range of dates and a list of tiles yields all url available for the specification.

    >>> viirs_datafile_iterator = ViirsDatafileIterator(start_date=..., end_date=..., tiles=...)
    >>> for datafile in viirs_datafile_iterator:
    >>>     url = datafile.url
    >>>     date = datafile.date
    >>>     tile = datafile.tile

    :param start_date: Beginning of the date range. Format %Y-%m-%d
    :param end_date: End of the date range. Format %Y-%m-%d
    :param tiles: List of tiles to download for every available timestamp
    """

    @dataclass
    class ViirsDatafile:
        tile: str
        date: datetime.date
        url: str

    VIIRS_ALL_IMAGES_URL = (
        "https://eogdata.mines.edu/wwwdata/viirs_products/dnb_composites/v10"
    )

    TILES = ["75N180W", "75N060W", "75N060E", "00N180W", "00N060W", "00N060E"]

    def __init__(
        self, start_date: datetime.date, end_date: datetime.date, tiles: List[str]
    ):
        assert len(set(tiles)) == len(
            tiles
        ), f"The list of tiles {tiles} contains duplicates."
        assert set(tiles).issubset(
            set(ViirsDatafileIterator.TILES)
        ), f"Tiles {tiles} are not a set. Valid tiles are {ViirsDatafileIterator.TILES}"

        self.start_date = start_date
        self.end_date = end_date
        self.tiles = tiles

        date_range = list(
            rrule.rrule(rrule.MONTHLY, dtstart=self.start_date, until=self.end_date)
        )
        self._data = [
            ViirsDatafileIterator.ViirsDatafile(
                tile, date, ViirsDatafileIterator.get_file_url(date, tile)
            )
            for tile in self.tiles
            for date in date_range
        ]

    @classmethod
    def get_base_url(cls, date: datetime.date) -> str:
        return f"{cls.VIIRS_ALL_IMAGES_URL}/{date.strftime('%Y%m')}/vcmcfg"

    @classmethod
    def get_file_url(cls, date: datetime.date, tile: str) -> str:
        base_url = cls.get_base_url(date)
        page = requests.get(base_url)
        parsed_page = BeautifulSoup(page.content, features="html.parser")
        hrefs = [link.get("href") for link in parsed_page.find_all("a", href=True)]
        matched_hrefs = list(filter(_file_regex(date, tile).match, hrefs))
        matched_hrefs = list(set(matched_hrefs))

        if len(matched_hrefs) != 1:
            raise RuntimeError(
                f"Expected to match exactly one href, instead got: {matched_hrefs}"
            )
        return base_url + "/" + matched_hrefs[0]

    def __iter__(self) -> collections.Iterable:
        return iter(self._data)


def download_viirs_imagery(
    start_date: datetime.date,
    end_date: datetime.date,
    tiles: List[str],
    viirs_credentials: ViirsRepoConfig,
) -> Dict[str, List[str]]:
    """
    Download all file for the specified date range and tiles and store them locally
    under data/raw/nightlight_imagery/<tile[s]>/<date>.tgz.

    :param start_date: Beginning of the date range. Format %Y-%m-%d
    :param end_date: End of the date range. Format %Y-%m-%d
    :param tiles: List of tiles to download for every available timestamp
    :param viirs_credentials: Use config.viirs_repo_access to get the credentials.
      Note, that they must be in your config_local.json under `viirs_repo_access`.
    :return: A dictionary mapping the date to a list of filenames for each tile of that date.
      E.g {"202001": [
            "data/raw/nightlight_imagery/75N180W/202001.tgz",
            "data/raw/nightlight_imagery/75N060W/202001.tgz"
          ]
    """

    c = get_config()
    downloaded_files = defaultdict(list)

    token = ViirsAuthenticator(
        viirs_credentials.username, viirs_credentials.password
    ).token

    for viirs_data_file in ViirsDatafileIterator(start_date, end_date, tiles):
        output_file_name = c.viirs_file_path(
            viirs_data_file.date, tile=viirs_data_file.tile, relative=True
        )
        downloaded_files[_year_month(viirs_data_file.date)].append(output_file_name)

        try:
            download_file(
                viirs_data_file.url,
                output_file_name,
                show_progress=True,
                overwrite_existing=False,
                headers={"Authorization": f"Bearer {token}"},
            )
        except FileExistsError:
            log.info(
                f"Skipping download of {output_file_name} because the file already exists."
            )
    return downloaded_files
