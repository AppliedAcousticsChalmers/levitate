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
array.phases = array.focus_phases(trap_pos) + array.twin_signature(trap_pos) + 0.2 * np.random.uniform(-np.pi, np.pi, array.num_transducers)


# The fields are superposed using mutual quiet zones, created by minimizing the
# pressure and velocity at the secondary point in each field.
trap_point = levitate.optimization.CostFunctionPoint(
    trap_pos, array,
    levitate.algorithms.second_order_stiffness(array, weights=(1, 1, 1)),
    levitate.algorithms.pressure_squared_magnitude(array, weights=1)
)
haptics_quiet_zone = levitate.optimization.CostFunctionPoint(
    haptics_pos, array,
    levitate.algorithms.pressure_squared_magnitude(array, weights=1),
    levitate.algorithms.velocity_squared_magnitude(array, weights=(1e3, 1e3, 1e3)),
)

# The levitation trap is found using a minimization sequence.
# First the phases are optimized for just a trap,
# then the phases and amplitudes are optimized to include the quiet zone.
trap_result = levitate.optimization.minimize(
    [[trap_point], [trap_point, haptics_quiet_zone]],
    array, variable_amplitudes=[False, True])[-1]

# The haptics point can be created using a simple focusing algorithm,
# so we can optimize for the inclusion of the quiet zone straight away.
# To retain the focus point we set a negative weight for the pressure,
# i.e. maximizing the pressure.
haptics_point = levitate.optimization.CostFunctionPoint(
    haptics_pos, array,
    levitate.algorithms.pressure_squared_magnitude(array, weights=-1e-3))
trap_quiet_zone = levitate.optimization.CostFunctionPoint(
    trap_pos, array,
    levitate.algorithms.pressure_squared_magnitude(array, weights=1),
    levitate.algorithms.velocity_squared_magnitude(array, weights=(1e3, 1e3, 1e3)),
)
array.phases = array.focus_phases(haptics_pos)
haptics_result = levitate.optimization.minimize(
    [haptics_point, trap_quiet_zone],
    array, variable_amplitudes=True)

# Visualize the individual fields, as well as the compound field.
array.complex_amplitudes = trap_result
trap_trace = array.visualize.pressure()
array.complex_amplitudes = haptics_result
haptics_trace = array.visualize.pressure()
array.complex_amplitudes = haptics_result * 0.3 + trap_result * 0.7
combined_trace = array.visualize.pressure()
fig = levitate.visualize.selection_figure(
    (trap_trace, 'Trap'),
    (haptics_trace, 'Haptics'),
    (combined_trace, 'Combined'),
    additional_traces=[array.visualize.transducers(signature_pos=trap_pos)]
)

plot(fig, filename='two_fields.html', auto_open=False)
