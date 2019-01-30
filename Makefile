#!/bin/bash
SHELL := /bin/bash

test:
	# convert the example notebook to a script
	jupyter nbconvert --to script example.ipynb

	# and disable Jupyter display handle output
	sed -i -e 's/jupyter=True/jupyter=False/g' example.py

	# activate virtualenv if present but continue anyway
	source /home/chris/.envs/gridfinder/bin/activate || true

	python3 example.py