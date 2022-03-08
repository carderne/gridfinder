from setuptools import find_packages, setup

test_requirements = ["pytest"]

setup(
    name="gridfinder",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="1.2.0-dev18",
    description="Library for finding gridlines from nightlight imagery",
    install_requires=open("requirements.txt").readlines()[1:],
    setup_requires=["wheel"],
    tests_require=test_requirements,
    author="appliedAI and Chris Ardene",
)
