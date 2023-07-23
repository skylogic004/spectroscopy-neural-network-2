from setuptools import setup, find_packages
import os

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "spectra_ml",
	version = "1",
	author = "Matthew Dirks",
	author_email = "md4000000@gmail.com",
	description = ("Python library to apply machine learning neural networks and sensor fusion models to spectroscopic data (i.e., spectra)"),
	keywords = "",
	url = "",
	packages=find_packages(),
	long_description=read('README.md'),
	package_data = {'spectra_ml': []},
)
