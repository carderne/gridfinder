from distutils.core import setup

import gridfinder

long_description = """
gridfinder uses NASA night time lights imagery to as an indicator of 
settlements/towns with grid electricity access. Then a minimum spanning 
tree is calculated for these connect points, using the Dijkstra 
algorithm and using existing road networks as a cost function. 
"""

setup(
    name="gridfinder",
    version=gridfinder.__version__,
    author="Chris Arderne",
    author_email="chris@rdrn.me",
    description="Algorithm for guessing MV grid network based on night time lights",
    long_description=long_description,
    url="https://github.com/carderne/gridfinder",
    packages=["gridfinder"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
