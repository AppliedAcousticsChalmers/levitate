import pytest
import numpy as np
import levitate

pos = np.array([5, -2, 80]) * 1e-3
array = levitate.arrays.RectangularArray(shape=2)
array.phases = array.focus_phases(pos) + array.twin_signature()


def test_cost_function_point():
    point = levitate.optimization.CostFunctionPoint(pos, array, levitate.algorithms.pressure_squared_magnitude(weights=3))
    values, jacobians = point(array.complex_amplitudes)
    np.testing.assert_allclose(values, np.abs(array.calculate.pressure(pos))**2 * 3)
    point.append(levitate.algorithms.second_order_stiffness(array, weights=(-1.2, 4, -1.6)))
    values, jacobians = point(array.complex_amplitudes)
    np.testing.assert_allclose(values, np.abs(array.calculate.pressure(pos))**2 * 3 + np.sum(array.calculate.stiffness(pos) * np.array([-1.2, 4, -1.6])))


def test_minimize_phases_amplitudes():
    trap = levitate.optimization.CostFunctionPoint(pos, array,
        levitate.algorithms.pressure_squared_magnitude(weights=1),
        levitate.algorithms.second_order_stiffness(array, weights=(1, 1, 1)))
    result = levitate.optimization.minimize(trap, array)
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=True, start_values=0.5 * array.complex_amplitudes, basinhopping=3)
    result = levitate.optimization.minimize(trap, array, constrain_transducers=[0, 3])


def test_minimize_sequence():
    trap = levitate.optimization.CostFunctionPoint(pos, array,
        levitate.algorithms.pressure_squared_magnitude(weights=1),
        levitate.algorithms.second_order_stiffness(array, weights=(1, 1, 1)))
    result = levitate.optimization.minimize(trap, array, variable_amplitudes='phases first', start_values=0.5 * array.complex_amplitudes)
    quiet_zone = levitate.optimization.CostFunctionPoint(np.array([-5, -2, 60]) * 1e-3, array,
        levitate.algorithms.pressure_squared_magnitude(weights=1),
        levitate.algorithms.velocity_squared_magnitude(array, weights=(1, 1, 1)))
    result = levitate.optimization.minimize([[trap], [trap, quiet_zone]], array)
    result, status = levitate.optimization.minimize([[trap], [trap, quiet_zone]], array, basinhopping=True, minimize_kwargs={'tol': 1e-6}, callback=lambda **kwargs: False, return_optim_status=True)


large_array = levitate.arrays.RectangularArray(shape=9)
large_array.phases = np.random.uniform(-np.pi, np.pi, large_array.num_transducers)
large_array.amplitudes = np.random.uniform(1e-3, 1, large_array.num_transducers)
operating_point = large_array.complex_amplitudes


@pytest.mark.parametrize("func, weights", [
    (levitate.algorithms.gorkov_divergence, (1, 0, 0)),
    (levitate.algorithms.gorkov_divergence, (0, 1, 0)),
    (levitate.algorithms.gorkov_divergence, (0, 0, 1)),
    (levitate.algorithms.gorkov_divergence, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.gorkov_laplacian, (1, 0, 0)),
    (levitate.algorithms.gorkov_laplacian, (0, 1, 0)),
    (levitate.algorithms.gorkov_laplacian, (0, 0, 1)),
    (levitate.algorithms.gorkov_laplacian, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.second_order_force, (1, 0, 0)),
    (levitate.algorithms.second_order_force, (0, 1, 0)),
    (levitate.algorithms.second_order_force, (0, 0, 1)),
    (levitate.algorithms.second_order_force, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.second_order_stiffness, (1, 0, 0)),
    (levitate.algorithms.second_order_stiffness, (0, 1, 0)),
    (levitate.algorithms.second_order_stiffness, (0, 0, 1)),
    (levitate.algorithms.second_order_stiffness, np.random.uniform(-10, 10, 3)),
    (levitate.algorithms.pressure_squared_magnitude, 1),
    (levitate.algorithms.pressure_squared_magnitude, np.random.uniform(-10, 10, 1)),
    (levitate.algorithms.velocity_squared_magnitude, (1, 0, 0)),
    (levitate.algorithms.velocity_squared_magnitude, (0, 1, 0)),
    (levitate.algorithms.velocity_squared_magnitude, (0, 0, 1)),
    (levitate.algorithms.velocity_squared_magnitude, np.random.uniform(-10, 10, 3)),
])
def test_jacobian_accuracy(func, weights):
    point = levitate.optimization.CostFunctionPoint(pos, large_array, func(large_array, weights=weights))
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
