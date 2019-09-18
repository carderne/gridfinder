# test.sh
#!/bin/bash

# clean output folder
rm -r test_output || true

# convert the example notebook to a script
python -m jupyter nbconvert --to script example.ipynb

# disable Jupyter and plt.imshow output
sed -i -e 's/jupyter=True/jupyter=False/g' example.py
sed -i -e 's/plt.imshow(.*)//g' example.py

# activate virtualenv if present but continue anyway
source /home/chris/.envs/gridfinder/bin/activate || true

# run script
python example.py

# clean up
rm example.py
