# gridfinder
Algorithm for guessing MV grid network based on NTL.

gridfinder uses NASA night time lights imagery to as an indicator of settlements/towns with grid electricity access. Then a minimum spanning tree is calculated for these connect points, using the Djikstra algorithm and using existing road networks as a cost function.

The algorithm looks as follows in process, guessing the grid network for Uganda:
[![Animated algorithm](gridfinder-animated.gif)]()

There are more details on my [blog post](https://rdrn.me/night-time-lights-find-grid/).

## Input requirements
gridfinder requires the following data sources:
- NASA VIIRS data, monthly and annual composites available [here](https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html).
- OSM highway data, most easily available using the [HOT Export Tool](https://export.hotosm.org/en/v3/), otherwise [BBBike](https://extract.bbbike.org/) or [geofabrik](https://download.geofabrik.de/), depending on your needs.

## Installation

**Requirements**

gridfinder requires Python >= 3.5 with the following packages installed:

 - `numpy` >=1.2.0
 - `scikit-image` >=0.14.1
 - `rasterio` >=1.0.13
 - `geopandas` >=0.4.0
 - `IPython`
 - `matplotlib`
 - `seaborn`

**Install from GitHub**

Downloads or clone the repository:

```
git clone https://github.com/carderne/gridfinder.git
```

Then ``cd`` into the directory, and install the required packages into a virtual environment:

```
pip install -r requirements.txt
```

Then run ``jupyter notebook`` and open ``gridfinder.ipynb``  or `quickrun.ipynb` to go over the main model usage and API.