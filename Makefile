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
	# export GDAL_DATA=$()
	fio env --gdal-data

	# run script
	python3 example.py