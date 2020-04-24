Overview
========
`schNell` is a very lightweight python module that can be used to compute basic map-level noise properties for generic networks of gravitational wave interferometers. This includes primarily the noise power spectrum  :math:`N_{\ell}`, but also other things, such as antenna patterns, overlap functions, inverse variance maps etc.

`schNell` is composed of two main classes:

- :class:`~schnell.Detector`\s. These contain information about each individual detector of the network (their positions, noise properties, orientation etc.).
- :class:`~schnell.NoiseCorrelation`\s. These describe the noise-level correlation between pairs of detectors.
- :class:`~schnell.MapCalculator`\s. These objects combine a list of :class:`~schnell.Detector`\s into a network (potentially together with a :class:`~schnell.NoiseCorrelation` object) and compute the corresponding map-level noise properties arising from their correlations.

A quick but thorough description of how these two classes can be used to compute different quantities can be found in `here <https://github.com/damonge/schNell/blob/master/examples/Nell_example.ipynb>`_.


Installation
------------

Installing `schNell` should be as simple as typing::

  pip install schnell [--user]


where the `--user` flag will only be necessary if you don't have admin privileges.
