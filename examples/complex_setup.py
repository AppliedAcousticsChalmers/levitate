"""Incorporating transducer models, reflectors, and doublesided arrays.

The setup shown here is a doublesided array where the two halves
are standing vertically 3 cm above a reflecting surface.
The two halves are separated by 20 cm, and each side has 50 elements.
The transducers are modeled as circular pistons, and the reflection is
included by using the `ReflectingTransducer ` meta-class.

In this example no optimization is done, but all optimization functions
support complex arrangements like this one.
"""

import numpy as np
import levitate

transducer = levitate.transducers.TransducerReflector(
    levitate.transducers.CircularPiston, effective_radius=3e-3,
    plane_intersect=(0, 0, 0), plane_normal=(0, 0, 1))

array = levitate.arrays.DoublesidedArray(
    levitate.arrays.RectangularArray, separation=200e-3,
    normal=(1, 0, 0), offset=(0, 0, 50e-3),
    shape=(5, 10), transducer=transducer)

phases = array.focus_phases(np.array([25e-3, 0, 40e-3]))
amps = levitate.utils.complex(phases)
array.visualize.zlimits = (0, 0.1)
array.visualize.append('Pressure')
array.visualize.append('Velocity')
array.visualize(amps).write_html(file='complex_setup.html', include_mathjax='cdn')
