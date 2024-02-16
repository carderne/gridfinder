# gridfinder

gridfinder uses night-time lights imagery to as an indicator of settlements/towns with grid electricity access. Then a minimum spanning tree is calculated for these connect points, using a many-to-many variant Dijkstra algorithm and using existing road networks as a cost function. Adapted from [this work from Facebook](https://github.com/facebookresearch/many-to-many-dijkstra). Currently gridfinder only uses road networks, but it would be trivial to add other cost parameters such as slope or terrain.

The algorithm looks as follows in process, guessing the grid network for Uganda:

[![Animated algorithm](https://raw.githubusercontent.com/carderne/gridfinder/master/gridfinder-animated.gif)](#)

## Input requirements
gridfinder requires the following data sources:
- VIIRS data, monthly and annual composites available [here](https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html).
- OSM highway data, most easily available using the [HOT Export Tool](https://export.hotosm.org/en/v3/), otherwise [BBBike](https://extract.bbbike.org/) or [geofabrik](https://download.geofabrik.de/), depending on your needs.

## Model usage

To get to grips with the API and steps in the model, open the Jupyter notebook `example.ipynb`.
This repository  includes the input data needed to do a test run for Burundi, so it should be a matter of openening the notebook and running all cells.

## Installation
### Install with pip

    pip install gridfinder

**Note:** On some operating systems (Ubuntu 18.04), you may get an error about `libspatialindex`. To overcome this on Ubuntu, run:

    sudo apt install libspatialindex-dev

### Install from GitHub
Download or clone the repository and install the required packages (preferably in a virtual environment):

    git clone https://github.com/carderne/gridfinder.git
    cd gridfinder
    pip install -e '.[dev]'

You can run ```./test.sh``` in the directory, which will do an entire run through using the test data and confirm whether everything is set up properly.
(It will fail if jupyter isn't installed!)
