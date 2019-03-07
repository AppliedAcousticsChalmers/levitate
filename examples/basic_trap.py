"""Simple trap optimization.

A very basic use-case, finding the correct phases to levitate a bead centered
5 cm above a 9x9 element rectangular array, then inspecting the resultant field.
"""

import numpy as np
import levitate
from plotly.offline import plot

pos = np.array([0, 0, 80e-3])
array = levitate.arrays.RectangularArray(9)
array.phases = array.focus_phases(pos) + array.twin_signature() + 0.2 * np.random.uniform(-np.pi, np.pi, array.num_transducers)


# Create the cost functions and minimize them.
point = levitate.optimization.CostFunctionPoint(
    pos, array,
    levitate.algorithms.gorkov_laplacian(array, weights=(-100, -100, -1)),
    levitate.algorithms.pressure_squared_magnitude(array, weights=1e-3),
)
results = levitate.optimization.minimize(point, array)

# Visualize the field.
array.complex_amplitudes = results
plot([array.visualize.pressure(), array.visualize.transducers(signature_pos=pos)], filename='basic_trap.html', auto_open=False)
