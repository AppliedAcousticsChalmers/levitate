"""Simple trap optimization.

A very basic use-case, finding the correct phases to levitate a bead centered
5 cm above a 9x9 element rectangular array, then inspecting the resultant field.
"""

import numpy as np
import levitate
import plotly.graph_objects as go

pos = np.array([0, 0, 80e-3])
array = levitate.arrays.RectangularArray(9)
array.phases = array.focus_phases(pos) + array.signature(stype='twin') + 0.2 * np.random.uniform(-np.pi, np.pi, array.num_transducers)

# Create the cost functions and minimize them.
point = levitate.fields.GorkovLaplacian(array) * (-100, -100, -1) + abs(levitate.fields.Pressure(array)) * 1e-3
results = levitate.optimization.minimize(point@pos, array)

# Visualize the field.
array.complex_amplitudes = results
go.Figure(
    [array.visualize.pressure(), array.visualize.transducers(signature_pos=pos)]
).write_html(file='basic_trap.html', include_mathjax='cdn')
