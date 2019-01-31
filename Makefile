#!/bin/bash
SHELL := /bin/bash

test:
	# convert the example notebook to a script
	jupyter nbconvert --to script example.ipynb

	# and disable Jupyter display handle output
	sed -i -e 's/jupyter=True/jupyter=False/g' example.py

	# activate virtualenv if present but continue anyway
	source /home/chris/.envs/gridfinder/bin/activate || true

	# for rasterio/Fiona CRS issues
	export GDAL_DATA="/home/travis/virtualenv/python3.6.3/lib/python3.6/site-packages/fiona/gdal_data"

	# run script
	python3 example.py

	# clean up
	rm example.py
	rm -r test_output