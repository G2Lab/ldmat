from setuptools import setup

setup(
    name="ldmat",
    version="0.1.2",
    author="Rockwell Weiner",
    author_email="rockwellw@gmail.com",
    description=("Efficient Storage and Querying of Linkage Disequilibrium Matrices"),
    license="MIT",
    url="https://github.com/G2Lab/ldmat",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": "ldmat = ldmat:run_cli"},
    install_requires=[
        "click>=8.1.3",
        "h5py>=3.7.0",
        "matplotlib>=3.4.3",
        "numpy>=1.21.3",
        "pandas>=1.3.4",
        "scipy>=1.8.1",
        "seaborn>=0.11.2",
    ],
    include_package_data=True,
)
