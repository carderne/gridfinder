# Gridlight

This is a fork from Chris Ardernes [Gridfinder repository](https://github.com/carderne/gridfinder).

Gridlight is a module to help assess energy access levels of human settlmenents for any area of interest.

Gridlight uses the nighttime lights (NTL) data from the [Visible Infrared Imaging Radiometer Suite (VIIRS)](https://www.earthdata.nasa.gov/learn/backgrounders/nighttime-lights) instruments aboard the NASA/NOAA Suomi National Polar-orbiting Partnership (Suomi NPP) and NOAA-20 satellites.

The algorithm first predicts a mask representing electricity access for each human settlement. 
Then a minimum spanning tree is calculated which links the previosuly identified areas, using a many-to-many variant Dijkstra algorithm and using existing road networks as a cost function. Adapted from [this work from Facebook](https://github.com/facebookresearch/many-to-many-dijkstra). 

## Input requirements
Gridlight requires the following data sources:
- VIIRS data, monthly and annual composites (more information [here](https://payneinstitute.mines.edu/eog/nighttime-lights/)). The script uses the API access. For this, you need to create a login. You can set this [here](https://eogdata.mines.edu/products/register/).
- Population data. HRSL works well, which can be donwloaded [here](https://ciesin.columbia.edu/data/hrsl/). 
- OSM highway data, most easily available here [geofabrik](https://download.geofabrik.de/)
- Existing grids

## Get started

## Install with pip

Just type 

```
pip install gridfinder
```

## From Github

Clone the repo: 

```
git clone https://github.com/carderne/gridfinder.git
```

For the installation of the Rtree dependency the `libspatialindex-dev` library is necessary. 

On Ubuntu and Mac, install it with:
```shell script
$ sudo apt install libspatialindex-dev
```

On Mac OS X, you need `spatialindex` instead:

```shell script
$ brew install spatialindex
```

This package uses Poetry for dependency management. You can install it [here](https://python-poetry.org/docs/#installation). You can create a virtual environment (called .venv) and install all required dependencies listed in poetry.lock using the following:

```shell script
$ poetry shell
$ poetry install
```

## Creating credentials
There are two credentials which are necessary to run Gridlight. Both have to be stored in a file called config_local.json. 

1. You have to create a login to access the VIIRS data. You can create the login here. Then add the PW and username into the config_local.json file

2. To store the input and resulting data, the package uses cloud buckets. You have to create a bucket and add the credentials into the config file. 

Your config_local.json file should then look like this:

```
{

   "viirs_repo_access": {
        "username": "<USER EMAIL>",
        "password": "<PW>"
    },
   "remote_storage_config": {
        "key": "<KEY>",
        "secret": "<SECRET>",
        "bucket": "<BUCKET_NAME>",
        "base_path": "<BUCKET_BASE_PATH>",
        "provider": "google_storage"
    }
}
```

## Running the script

There are three steps

1. Download the VIIRS data which covers your area of interest.

First we have to download the nightlight data. This is done using the run_download_viirs_imagery.py script. 

```
python run_download_viirs_imagery.py --start-date 2021-01-01 --end-date 2021-12-31
--tiles "75N060W" --tiles "00N060W" 
```

To get a description of all arguments, you can type python run_download_viirs_imagery.py --help

This script downloads the relevant VIIRS data into the data/raw/nightlight_imagery directory and pushes it to the cloud storage bucket 

For the start and end dates, the days don´t matter. The month you specify will be fully downloaded independent on the day specified. 

There are a total of six tiles: They either go from the equator north or south (see images below and link [here](https://eogdata.mines.edu/download_dnb_composites.html)):


- 00N 060E (Asia and Australia, south of the equator)
- 00N 060W (Africa south of the equator) 
- 00N 180W (South America)
- 75N 060E (Most of Asia, north of the equator)
- 75N 060W (Africa north of the equator, including Europe)
- 75N 180W (North America)

You have to give all the tiles which your area of interest covers. For example, if you want to run Gridlight for South Africa, you need the 00N060W tile. If you are interested in Ethiopia, you need both tiles 00N060W and 75N060W. If more than one tile is given as input, the script will fuse them together. So it´s important that you run the download script with all the tiles you need as argument. 

Important: Depending on the start and end dates, the download can take several hours.

2. Add other necessary datasets

The script requires a population as well as a road dataset. Both have to be stored into the data/ground_truth directory. 
- An area of interest file: This has to be a vector file outlining the area of interest you want to run Gridlight for.
- For population, any raster file can be used. The population file is only used to filter out areas which are not populated. The actual population value any of those datasets would have are not taken into consideration. Importantly, a pixel value of 0 refers to 0 population. 
We recommend using HRSL data, which can be downloaded [here](https://ciesin.columbia.edu/data/hrsl/) or from humdata.org (e.g. [here](https://data.humdata.org/dataset/highresolutionpopulationdensitymaps-sle) for Sierra Leone.

- OSM Road data: For the grid prediction, road data can be used to guide the MV power line prediciton. The OSM Highway data can be downloaded here for different countries: [Geofabrik](https://download.geofabrik.de/) 
- If you already have a power grid file, you can specify it. It will be appended to the resulting estimate.


3. Run Gridlight

Now you can run the Gridlight script, referencing the datasets above.

To see all input arguments and a description, type:

python run_gridfinder.py --help

For example, a function call might look like this:
```
python run_gridfinder.py --area-of-interest-data nigeria/nigeria-kano.geojson --roads-data nigeria/nigeria-roads-200101.gpkg --population-data nigeria/population_nga_2018-10-01.tif --grid-truth-data nigeria/nigeriafinal.geojson  --nightlight-data nightlight_imagery/75N060W --nightlight-output nigeria/ntl_clipped
```

## Example output

The script provides a set of outputs. TBD