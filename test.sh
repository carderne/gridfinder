# test.sh
#!/bin/bash

# clean output folder
rm -r test_output || true

# convert the example notebook to a script
jupyter nbconvert --to script example.ipynb

# and disable Jupyter display handle output
sed -i -e 's/jupyter=True/jupyter=False/g' example.py

# activate virtualenv if present but continue anyway
source /home/chris/.envs/gridfinder/bin/activate || true

# run script
python example.py

# clean up
rm example.py
