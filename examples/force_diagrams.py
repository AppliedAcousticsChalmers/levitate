"""Visualizing the force around a trap.

Convenient methods to show how the radiation force changes
along the Cartesian axes for different sizes of the beads.
"""

import numpy as np
import levitate
from levitate.visualizers import ForceDiagram

pos = np.array([0, 0, 60e-3])
array = levitate.arrays.RectangularArray(16)
amps = levitate.utils.complex(array.focus_phases(pos) + array.signature(stype='twin'))

diagram = ForceDiagram(array)
radii = [1e-3, 2e-3, 4e-3, 8e-3, 16e-3]
for radius in radii:
    diagram.append([pos, {'radius': radius, 'name': '{} mm'.format(radius * 1e3)}])
diagram(amps).write_html(file='force_diagrams.html', include_mathjax='cdn')
