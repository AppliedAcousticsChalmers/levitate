"""Incorporating transducer models, reflectors, and doublesided arrays.

The setup shown here is a doublesided array where the two halves
are standing vertically 3 cm above a reflecting surface.
The two halves are separated by 20 cm, and each side has 50 elements.
The transducers are modeled as circular pistons, and the reflection is
included by using the `ReflectingTransducer ` meta-class.

In this example no optimization is done, but all optimization functions
support complex arrangements like this one.
"""

import levitate
from plotly.offline import plot
import numpy as np

transducer = levitate.transducers.ReflectingTransducer(
    levitate.transducers.CircularPiston, effective_radius=3e-3,
    plane_distance=0, plane_normal=(0, 0, 1))

array = levitate.arrays.DoublesidedArray(
    levitate.arrays.RectangularArray, separation=200e-3,
    normal=(1, 0, 0), offset=(0, 0, 50e-3),
    shape=(5, 10), transducer_model=transducer)

array.phases = array.focus_phases(np.array([25e-3, 0, 40e-3]))
array.visualize.zlimits = (0, 0.1)
plot(levitate.visualize.selection_figure(
    (array.visualize.pressure(), 'Pressure'),
    (array.visualize.velocity(), 'Velocity'),
    additional_traces=[array.visualize.transducers()]),
    filename='complex_setup.html', auto_open=False)
