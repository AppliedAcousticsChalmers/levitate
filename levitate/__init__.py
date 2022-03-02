"""Levitate, a python package for simulating acoustic levitation using ultrasonic transducer arrays.

The API consists of four main modules, and a few supporting modules.
The main modules contain models to handle transducers and transducer arrays, in the `~levitate.transducers` and `~levitate.arrays` modules respectively,
algorithms to calculate physical properties in the `~levitate.fields` module, and some numerical optimization functions in the `~levitate.optimization` module.
There is also a `~levitate.visualizers` module with some convenience function to show various fields, and some analysis tools in `~levitate.analysis`.
It is possible to use different materials or material properties from the `~levitate.materials` module.

The `~levitate.hardware` module includes definitions with array geometries corresponding to some physical prototypes,
and python-c++ combined setup to control Ultrahaptics physical hardware directly from python.
This implementation of Ultrahaptics control from python is not officially supported by Ultrahaptics, and only enables a very limited subset of the research SDK.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['transducers', 'arrays', 'hardware', 'materials', 'optimization', 'fields', 'analysis']

from . import _version
__version_info__ = _version.version_info
__version__ = _version.version
del _version  # Keep the namespace clean. Import from directly form the `_version` module if needed.


from . import *


def complex(phase, magnitude=1.):
    return np.exp(1j * phase) * magnitude


def phase(complex_amplitude):
    return np.angle(complex_amplitude)


def magnitude(complex_amplitude):
    return np.abs(complex_amplitude)


def phase_magnitude(complex_amplitude):
    return phase(complex_amplitude), magnitude(complex_amplitude)


def rms(complex_amplitude):
    return magnitude(complex_amplitude) / 2**0.5
