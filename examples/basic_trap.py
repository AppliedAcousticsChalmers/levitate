"""Simple trap optimization.

A very basic use-case, finding the correct phases to levitate a bead centered
5 cm above a 9x9 element rectangular array, then inspecting the resultant field.
"""

import numpy as np
import levitate
from plotly.offline import plot

array = levitate.arrays.RectangularArray(9)
pos = np.array([0, 0, 50e-3])

# Create the cost functions and minimize them.
gorkov_laplacian = levitate.cost_functions.gorkov_laplacian(array, pos, weights=(-1, -1, -1))
pressure = levitate.cost_functions.pressure(array, pos, weight=1e3)
results = levitate.cost_functions.minimize([gorkov_laplacian, pressure], array)

# Visualize the field.
array.complex_amplitudes = results
plot([array.visualize.pressure(), array.visualize.transducers()])
