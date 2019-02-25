from distutils.core import setup

import gridfinder

long_description = """
gridfinder uses NASA night time lights imagery to as an indicator of 
settlements/towns with grid electricity access. Then a minimum spanning 
tree is calculated for these connect points, using the Djikstra 
algorithm and using existing road networks as a cost function. 
"""

setup(
    name='gridfinder',
    version=gridfinder.__version__,
    author='Chris Arderne',
    author_email='chris@rdrn.me',
    description='Algorithm for guessing MV grid network based on night time lights',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/carderne/gridfinder',
    packages=['gridfinder'],
    install_requires=[
        'numpy>=1.2.0',
        'scikit-image>=0.14.1',
        'rasterio>=1.0.18',
        'geopandas>=0.4.0',
        'Rtree>=0.8.3',
        'affine>=2.2.1',
        'descartes',
        'Pillow>=5.3.0',
        'pyproj>=1.9.5.1',
        'pytz>=2018.7'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
)