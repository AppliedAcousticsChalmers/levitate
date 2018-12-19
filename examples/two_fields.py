"""Superposition of two fields.

A more advanced usage, designed to create a field with a levitation trap
and a haptics focus point.
"""

import numpy as np
import levitate
from plotly.offline import plot

array = levitate.arrays.RectangularArray((21, 12))
trap_pos = np.array([-20e-3, 0, 60e-3])
haptics_pos = np.array([40e-3, 0, 90e-3])


# The fields are superposed using mutual quiet zones, created by minimizing the
# pressure and velocity at the secondary point in each field.
gorkov_laplacian = levitate.cost_functions.second_order_stiffness(array, trap_pos, weights=(1, 1, 1))
trap_pressure_null = levitate.cost_functions.pressure(array, trap_pos, weight=1)
trap_velocity_null = levitate.cost_functions.velocity(array, trap_pos, weights=(1e3, 1e3, 1e3))
haptics_pressure_null = levitate.cost_functions.pressure(array, haptics_pos, weight=1)
haptics_velocity_null = levitate.cost_functions.velocity(array, haptics_pos, weights=(1e3, 1e3, 1e3))
# To create a focus point we set a negative weight for the pressure.
haptics_focus_pressure = levitate.cost_functions.pressure(array, trap_pos, weight=-1e-3)

# The levitation trap is found using a minimization sequence.
# First the phases are optimized for just a trap,
# then the phases and amplitudes are optimized to include the quiet zone.
funcs = [[gorkov_laplacian, trap_pressure_null],
         [gorkov_laplacian, trap_pressure_null, haptics_pressure_null, haptics_velocity_null]]
trap_results = levitate.cost_functions.minimize(funcs, array, variable_amplitudes=[False, True])[-1]

# The haptics point can be created using a simple focusing algorithm,
# so we can optimize for the inclusion of the quiet zone straight away.
funcs = [trap_pressure_null, trap_velocity_null, haptics_focus_pressure]
array.phases = array.focus_phases(haptics_pos)
haptics_result = levitate.cost_functions.minimize(funcs, array, variable_amplitudes=True)

# Visualize the individual fields, as well as the compound field.
array.complex_amplitudes = trap_results
plot([array.visualize.pressure(), array.visualize.transducers()],
     filename='trap_results.html')
array.complex_amplitudes = haptics_result
plot([array.visualize.pressure(), array.visualize.transducers()],
     filename='haptics_results.html')
array.complex_amplitudes = haptics_result * 0.3 + trap_results * 0.7
plot([array.visualize.pressure(), array.visualize.transducers()],
     filename='combined_results.html')
