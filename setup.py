#!/usrbin/env python
from setuptools import setup

description = "schNell - Fast calculation of N_l for GW anisotropies"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(name="schnell",
      version="0.2.0",
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/damonge/schNell",
      author="David Alonso",
      author_email="david.alonso@physics.ox.ac.uk",
      install_requires=requirements,
      packages=['schnell'],
)
