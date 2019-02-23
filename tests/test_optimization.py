import numpy as np
import levitate
import levitate.optimization
import levitate.algorithms

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
    np.testing.assert_allclose(result, array.complex_amplitudes)
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
