Overview
========
schNell is a very lightweight python module that can be used to compute basic map-level noise properties for generic networks of gravitational wave interferometers. This includes primarily the noise power spectrum  :math:`N_{\ell}`, but also other things, such as antenna patterns, overlap functions, inverse variance maps etc.

schNell is composed of two main classes:

- :class:`~schnell.Detector`\s. These contain information about each individual detector of the network (their positions, noise properties, orientation etc.).
- :class:`~schnell.MapCalculator`\s. These objects combine a list of :class:`~schnell.Detector`\s into a network and compute the correspondin map-level noise properties arising from their correlations.

A quick but thorough description of how these two classes can be used to compute different quantities can be found in `here <https://github.com/damonge/schNell/blob/master/examples/Nell_example.ipynb>`_.
