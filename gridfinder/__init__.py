"""
gridfinder package contains the following modules:

 - _util.py : helper functions used internally
 - gridfinder.py : main implementation of Dijkstra's algorithm
 - post.py : postprocess the algorithm output and check accuracy
 - prepare.py : transforming input data into the cost and targets arrays
"""

from importlib.metadata import version

from .gridfinder import *  # NoQA
from .post import *  # NoQA
from .prepare import *  # NoQA
from .util import *  # NoQA

__version__ = version("gridfinder")
