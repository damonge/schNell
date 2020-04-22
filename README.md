# schNell

schNell is a very lightweight python module that can be used to compute basic map-level noise properties for generic networks of gravitational wave interferometers. This includes primarily the noise power spectrum  "N_ell", but also other things, such as antenna patterns, overlap functions, inverse variance maps etc.

## Installation
You can install schnell simply by typing
```
pip install schnell [--user]
```
(use `--user` if you don't have admin privileges on your machine).
Or for development versions you can download the repository with git and install from there using `python setup.py install [--user]`.

## Documentation
Documentation can be found on [readthedocs](https://schnell.readthedocs.io/en/latest/).

This example [notebook](https://github.com/damonge/schNell/blob/master/examples/Nell_example.ipynb) on github also showcases the main functionalities of the module.

## License and credits
If you use schNell, we kindly ask you to cite its companion paper (link to appear here soon).

The code is available under the [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.

If you have a problem you've not been able to debug, or a feature request/suggestion, please open an issue on github to discuss it.
