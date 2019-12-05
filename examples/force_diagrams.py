"""Visualizing the force around a trap.

Convenient methods to show how the radiation force changes
along the Cartesian axes for different sizes of the beads.
"""

import numpy as np
import levitate
import plotly.graph_objects as go

pos = np.array([0, 0, 60e-3])
array = levitate.arrays.RectangularArray(16)
array.phases = array.focus_phases(pos) + array.signature(stype='twin')

radii = [1e-3, 2e-3, 4e-3, 8e-3, 16e-3]
traces = []
for radius in radii:
    traces.extend(array.visualize.force_diagram_traces(
        pos, radius_sphere=radius, label='{:.0f} mm'.format(radius * 1e3)))
layout = array.visualize.force_diagram_layout()

go.Figure(traces, layout).write_html(file='force_diagrams.html', include_mathjax='cdn')
