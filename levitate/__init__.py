"""Levitate, a python package for simulating acoustic levitation using ultrasonic transducer arrays.

The API consists of four main modules, and a few supporting modules.
The main modules contain models to handle transducers and transducer arrays, in the `~levitate.transducers` and `~levitate.arrays` modules respectively,
algorithms to calculate physical properties in the `~levitate.algorithms` module, and some numerical optimization functions in the `~levitate.optimization` module.
There is also a `~levitate.visualize` module with some convenience function to show various fields, a few utilities in `~levitate.utils`.
It is possible to use different materials or material properties from the `~levitate.materials` module.

The `~levitate.hardware` module includes definitions with array geometries corresponding to some physical prototypes,
and python-c++ combined setup to control Ultrahaptics physical hardware directly from python.
This implementation of Ultrahaptics control from python is not officially supported by Ultrahaptics, and only enables a very limited subset of the research SDK.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ['transducers', 'arrays', 'hardware', 'materials', 'optimization', 'algorithms', 'utils']

from . import _version
__version__ = _version.__version__
del _version  # Keep the namespace clean. Import from directly form the `_version` module if needed.


from . import *
