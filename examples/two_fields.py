"""Superposition of two fields.

A more advanced usage, designed to create a field with a levitation trap
and a haptics focus point.
"""

import numpy as np
import levitate

array = levitate.arrays.RectangularArray((21, 12))
trap_pos = np.array([-20e-3, 0, 60e-3])
haptics_pos = np.array([40e-3, 0, 90e-3])
phases = array.focus_phases(trap_pos) + array.signature(trap_pos, stype='twin') + 0.2 * np.random.uniform(-np.pi, np.pi, array.num_transducers)
start = levitate.utils.complex(phases)

# The fields are superposed using mutual quiet zones, created by minimizing the
# pressure and velocity at the secondary point in each field.
# We will need three fields, calculating the pressure magnitude,
# the velocity magnitude, and the stiffenss of the trap.
p = abs(levitate.fields.Pressure(array))
v = abs(levitate.fields.Velocity(array))
s = levitate.fields.RadiationForceStiffness(array)

# The levitation trap is found using a minimization sequence.
# First the phases are optimized for just a trap,
# then the phases and amplitudes are optimized to include the quiet zone.
trap_result = levitate.optimization.minimize(
    [
        (s * (1, 1, 1) + p * 1)@trap_pos,
        (s * (1, 1, 1) + p * 1)@trap_pos + (v * (1e3, 1e3, 1e3) + p * 1)@haptics_pos
    ],
    array, start_values=start, variable_amplitudes=[False, True])[-1]

# The haptics point can be created using a simple focusing algorithm,
# so we can optimize for the inclusion of the quiet zone straight away.
# To retain the focus point we set a negative weight for the pressure,
# i.e. maximizing the pressure.
start = levitate.utils.complex(array.focus_phases(haptics_pos))
haptics_result = levitate.optimization.minimize(
    p * (-1)@haptics_pos + (p * 1 + v * (1e3, 1e3, 1e3))@trap_pos,
    array, start_values=start, variable_amplitudes=True)

# Visualize the individual fields, as well as the compound field.
array.visualize.append('pressure')
array.visualize(
    trap_result, haptics_result, haptics_result * 0.3 + trap_result * 0.7,
    labels=['Trap', 'Haptics', 'Combined']
).write_html(file='two_fields.html', include_mathjax='cdn')
