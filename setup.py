#!/usrbin/env python
from setuptools import setup

description = "schNell - Fast calculation of N_l for GW anisotropies"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="schnell",
      version="0.1.0",
      long_description=description,
      long_description_content_type='text/markdown',
      url="https://github.com/damonge/schnell",
      author="David Alonso",
      author_email="david.alonso@physics.ox.ac.uk",
      install_requires=requirements,
      packages=['schnell'],
)
