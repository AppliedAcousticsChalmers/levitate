import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import Air
Air.c = 343
Air.rho = 1.2

pos = np.array([5, -2, 80]) * 1e-3
array = levitate.arrays.RectangularArray(shape=2)
array.phases = array.focus_phases(pos) + array.signature(stype='twin')


def test_minimize_phases_amplitudes():
    trap = abs(levitate.algorithms.Pressure(array)) * 1 @ pos + levitate.algorithms.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array)
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=True, start_values=0.5 * array.complex_amplitudes, basinhopping=3)
    result = levitate.optimization.minimize(trap, array, constrain_transducers=[0, 3])


def test_minimize_sequence():
    trap = abs(levitate.algorithms.Pressure(array)) * 1 @ pos + levitate.algorithms.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array, variable_amplitudes='phases first', start_values=0.5 * array.complex_amplitudes)
    quiet_zone = (abs(levitate.algorithms.Pressure(array)) * 1 + abs(levitate.algorithms.Velocity(array)) * (1, 1, 1)) @ (np.array([-5, -2, 60]) * 1e-3)
    result = levitate.optimization.minimize([trap, trap + quiet_zone], array)
    result, status = levitate.optimization.minimize([trap, trap + quiet_zone], array, basinhopping=True, minimize_kwargs={'tol': 1e-6}, callback=lambda **kwargs: False, return_optim_status=True)


large_array = levitate.arrays.RectangularArray(shape=9)
large_array.phases = np.random.uniform(-np.pi, np.pi, large_array.num_transducers)
large_array.amplitudes = np.random.uniform(1e-3, 1, large_array.num_transducers)
operating_point = large_array.complex_amplitudes


@pytest.mark.parametrize("func, weight", [
    (levitate.algorithms.GorkovPotential, 1),
    (levitate.algorithms.GorkovPotential, np.random.uniform(-10, 10)),
    (levitate.algorithms.GorkovGradient, (1, 0, 0)),
    (levitate.algorithms.GorkovGradient, (0, 1, 0)),
    (levitate.algorithms.GorkovGradient, (0, 0, 1)),
    (levitate.algorithms.GorkovGradient, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.GorkovLaplacian, (1, 0, 0)),
    (levitate.algorithms.GorkovLaplacian, (0, 1, 0)),
    (levitate.algorithms.GorkovLaplacian, (0, 0, 1)),
    (levitate.algorithms.GorkovLaplacian, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.RadiationForce, (1, 0, 0)),
    (levitate.algorithms.RadiationForce, (0, 1, 0)),
    (levitate.algorithms.RadiationForce, (0, 0, 1)),
    (levitate.algorithms.RadiationForce, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.RadiationForceStiffness, (1, 0, 0)),
    (levitate.algorithms.RadiationForceStiffness, (0, 1, 0)),
    (levitate.algorithms.RadiationForceStiffness, (0, 0, 1)),
    (levitate.algorithms.RadiationForceStiffness, np.random.uniform(-10, 10, 3)),
    (lambda arr, weight, position: abs(levitate.algorithms.Pressure(arr, weight=weight, position=position)), 1),
    (lambda arr, weight, position: abs(levitate.algorithms.Pressure(arr, weight=weight, position=position)), np.random.uniform(-10, 10)),
    (lambda arr, weight, position: abs(levitate.algorithms.Velocity(arr, weight=weight, position=position)), (1, 0, 0)),
    (lambda arr, weight, position: abs(levitate.algorithms.Velocity(arr, weight=weight, position=position)), (0, 1, 0)),
    (lambda arr, weight, position: abs(levitate.algorithms.Velocity(arr, weight=weight, position=position)), (0, 0, 1)),
    (lambda arr, weight, position: abs(levitate.algorithms.Velocity(arr, weight=weight, position=position)), np.random.uniform(-10, 10, 3)),
])
def test_jacobian_accuracy(func, weight):
    point = func(large_array, weight=weight, position=pos)
    values_at_operating_point = point(operating_point)

    phase_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        large_array.phases[idx] += 1e-6
        upper_val = point(large_array.complex_amplitudes)[0]
        large_array.phases[idx] -= 2e-6
        lower_val = point(large_array.complex_amplitudes)[0]
        large_array.phases[idx] += 1e-6
        phase_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(phase_jacobians, -values_at_operating_point[1].imag, 1e-5, 1e-8)

    amplitude_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        large_array.amplitudes[idx] += 1e-6
        upper_val = point(large_array.complex_amplitudes)[0]
        large_array.amplitudes[idx] -= 2e-6
        lower_val = point(large_array.complex_amplitudes)[0]
        large_array.amplitudes[idx] += 1e-6
        amplitude_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(amplitude_jacobians, values_at_operating_point[1].real / large_array.amplitudes, 1e-5, 1e-8)

    real_jacobians = np.zeros(large_array.num_transducers)
    moving_point = operating_point.copy()
    for idx in range(large_array.num_transducers):
        moving_point[idx] += 1e-6
        upper_val = point(moving_point)[0]
        moving_point[idx] -= 2e-6
        lower_val = point(moving_point)[0]
        moving_point[idx] += 1e-6
        real_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(real_jacobians, np.real(values_at_operating_point[1] / operating_point), 1e-5, 1e-8)

    imag_jacobians = np.zeros(large_array.num_transducers)
    moving_point = operating_point.copy()
    for idx in range(large_array.num_transducers):
        moving_point[idx] += 1j * 1e-6
        upper_val = point(moving_point)[0]
        moving_point[idx] -= 1j * 2e-6
        lower_val = point(moving_point)[0]
        moving_point[idx] += 1j * 1e-6
        imag_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(imag_jacobians, -np.imag(values_at_operating_point[1] / operating_point), 1e-5, 1e-8)
