"""
gridfinder package contains the following modules:

 - gridfinder.py : main implementation of Dijkstra's algorithm
 - prepare.py : transforming input data into the cost and targets arrays
 - post.py : postprocess the algorithm output and check accuracy
 - _util.py : helper functions used internally
"""

__version__ = "1.1.0"

from .prepare import *
from .gridfinder import *
from .post import *
from ._util import *
