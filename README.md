# gridfinder

gridfinder uses night-time lights imagery to as an indicator of settlements/towns with grid electricity access. Then a minimum spanning tree is calculated for these connect points, using a many-to-many variant Dijkstra algorithm and using existing road networks as a cost function. Adapted from [this work from Facebook](https://github.com/facebookresearch/many-to-many-dijkstra). Currently gridfinder only uses road networks, but it would be trivial to add other cost parameters such as slope or terrain.

Documentation is available [here](http://customers.pages.aai.lab/tfe/vida/gridfinder/docs/).

The algorithm looks as follows in process, guessing the grid network for Uganda:

[![Animated algorithm](https://raw.githubusercontent.com/carderne/gridfinder/master/gridfinder-animated.gif)](#)

## Input requirements
gridfinder requires the following data sources:
- VIIRS data, monthly and annual composites available [here](https://payneinstitute.mines.edu/eog/nighttime-lights/).
- OSM highway data, most easily available using the [HOT Export Tool](https://export.hotosm.org/en/v3/), otherwise [BBBike](https://extract.bbbike.org/) or [geofabrik](https://download.geofabrik.de/), depending on your needs.

## Model usage

To get to grips with the API and steps in the model, open the Jupyter notebook `example.ipynb`. This repository  includes the input data needed to do a test run for Burundi, so it should be a matter of openening the notebook and running all cells.

## Installation
### Requirements
gridfinder requires Python >= 3.5 with the following packages installed:
 - `numpy` >=1.2.0
 - `scikit-image` >=0.14.1
 - `rasterio` >=1.0.13
 - `geopandas` >=0.4.0

These additional packages may be necessary depending on your configuration:
 - `Rtree` >= 0.8.3
 - `affine` >= 2.2.1
 - `Pillow` >= 5.3.0
 - `pyproj` >= 1.9.5.1
 - `pytz` >= 2018.7
 - `descartes`

 And these for using an interactive notebook:
 - `IPython`
 - `jupyter`
 - `matplotlib`
 - `seaborn`
 - `folium`

### Configuring the appliedAI Nexus repo

The `data_access` dependency is hosted in a private PyPI repository, which needs
to be made known to `pip` in order to be used as a source.

If you are using a plain virtualenv in `venv`, add the following lines to `venv/pip.conf`,
otherwise you can add them to your system-wide pip configuration in `~/.pip/pip.conf`:

```ini
[global]
extra-index-url = https://nexus.admin.aai.sh/repository/aai-pypi/simple
```

Alternatively, these options can also be directly passed to
pip on the command line or through env vars (e.g. in CI builds)
Make sure to use the version for the `data_access` library specified in the requirements.txt file:

```shell script
$ pip install data_access==vv.vv \
    --extra-index-url https://nexus.admin.aai.sh/repository/aai-pypi/simple
```

or

```shell script
export PIP_EXTRA_INDEX_URL=https://nexus.admin.aai.sh/repository/aai-pypi/simple
pip install tfe_clustering==vv.vv
``` 


### Install with pip

    pip install gridfinder

**Note:** On some operating systems (Ubuntu 18.04), you may get an error about `libspatialindex`. To overcome this on Ubuntu, run:

    sudo apt install libspatialindex-dev

### Install from GitHub
Download or clone the repository and install the required packages (preferably in a virtual environment):

    git clone https://github.com/carderne/gridfinder.git
    cd gridfinder
    pip install -r requirements.txt

You can run ```./test.sh``` in the directory, which will do an entire run through using the test data and confirm whether everything is set up properly. (It will fail if jupyter isn't installed!)

