from setuptools import find_packages, setup

test_requirements = ["pytest"]

setup(
    name="gridlight",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="1.2.0-dev18",
    description="Library for finding electrified areas and estimate MV gridlines from nightlight imagery",
    install_requires=open("requirements.txt").readlines()[1:],
    setup_requires=["wheel"],
    tests_require=test_requirements,
    author="VIDA.place, AppliedAI and Chris Ardene",
)
