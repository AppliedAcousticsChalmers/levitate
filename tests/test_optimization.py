import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

pos = np.array([5, -2, 80]) * 1e-3
array = levitate.arrays.RectangularArray(shape=2)
array.phases = array.focus_phases(pos) + array.signature(stype='twin')


def test_minimize_phases_amplitudes():
    trap = abs(levitate.fields.Pressure(array)) * 1 @ pos + levitate.fields.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array)
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=True, start_values=0.5 * array.complex_amplitudes, basinhopping=3)
    result = levitate.optimization.minimize(trap, array, constrain_transducers=[0, 3])


def test_minimize_sequence():
    trap = abs(levitate.fields.Pressure(array)) * 1 @ pos + levitate.fields.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=[False, True], start_values=0.5 * array.complex_amplitudes)
    quiet_zone = (abs(levitate.fields.Pressure(array)) * 1 + abs(levitate.fields.Velocity(array)) * (1, 1, 1)) @ (np.array([-5, -2, 60]) * 1e-3)
    result = levitate.optimization.minimize([trap, trap + quiet_zone], array)
    result, status = levitate.optimization.minimize([trap, trap + quiet_zone], array, basinhopping=True, minimize_kwargs={'tol': 1e-6}, callback=lambda **kwargs: False, return_optim_status=True)


large_array = levitate.arrays.RectangularArray(shape=9)
large_array.phases = np.random.uniform(-np.pi, np.pi, large_array.num_transducers)
large_array.amplitudes = np.random.uniform(1e-3, 1, large_array.num_transducers)
operating_point = large_array.complex_amplitudes


@pytest.mark.parametrize("func, kwargs, take_abs, weight", [
    (levitate.fields.GorkovPotential, {}, False, 1),
    (levitate.fields.GorkovPotential, {}, False, np.random.uniform(-10, 10)),
    (levitate.fields.GorkovGradient, {}, False, (1, 0, 0)),
    (levitate.fields.GorkovGradient, {}, False, (0, 1, 0)),
    (levitate.fields.GorkovGradient, {}, False, (0, 0, 1)),
    (levitate.fields.GorkovGradient, {}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.GorkovLaplacian, {}, False, (1, 0, 0)),
    (levitate.fields.GorkovLaplacian, {}, False, (0, 1, 0)),
    (levitate.fields.GorkovLaplacian, {}, False, (0, 0, 1)),
    (levitate.fields.GorkovLaplacian, {}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.RadiationForce, {}, False, (1, 0, 0)),
    (levitate.fields.RadiationForce, {}, False, (0, 1, 0)),
    (levitate.fields.RadiationForce, {}, False, (0, 0, 1)),
    (levitate.fields.RadiationForce, {}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.RadiationForceStiffness, {}, False, (1, 0, 0)),
    (levitate.fields.RadiationForceStiffness, {}, False, (0, 1, 0)),
    (levitate.fields.RadiationForceStiffness, {}, False, (0, 0, 1)),
    (levitate.fields.RadiationForceStiffness, {}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.RadiationForceGradient, {}, False, [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    (levitate.fields.RadiationForceGradient, {}, False, [[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    (levitate.fields.RadiationForceGradient, {}, False, np.random.uniform(-10, 10, (3, 3))),
    (levitate.fields.Pressure, {}, True, 1),
    (levitate.fields.Pressure, {}, True, np.random.uniform(-10, 10)),
    (levitate.fields.Velocity, {}, True, (1, 0, 0)),
    (levitate.fields.Velocity, {}, True, (0, 1, 0)),
    (levitate.fields.Velocity, {}, True, (0, 0, 1)),
    (levitate.fields.Velocity, {}, True, np.random.uniform(-10, 10, 3)),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 1}, False, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 5}, False, np.random.uniform(-10, 10, (3, 36))),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'orders': 12}, False, np.random.uniform(-10, 10, (3, 169))),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [1, 0, 0]),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [0, 1, 0]),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [0, 0, 1]),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 7, 'radius_sphere': large_array.k * 8}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 16, 'radius_sphere': large_array.k * 19}, False, np.random.uniform(-10, 10, 3)),
    (levitate.fields.SphericalHarmonicsForceGradientDecomposition, {'orders': 1, 'radius_sphere': large_array.k * 2}, False, np.random.uniform(-10, 10, (3, 3, 4))),
    (levitate.fields.SphericalHarmonicsForceGradientDecomposition, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, np.random.uniform(-10, 10, (3, 3, 25))),
    (levitate.fields.SphericalHarmonicsForceGradientDecomposition, {'orders': 12, 'radius_sphere': large_array.k * 18}, False, np.random.uniform(-10, 10, (3, 3, 169))),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 4, 'radius_sphere': large_array.k * 6}, False, [[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 7, 'radius_sphere': large_array.k * 9}, False, np.random.uniform(-10, 10, (3, 3))),
    (levitate.fields.SphericalHarmonicsForceGradient, {'orders': 12, 'radius_sphere': large_array.k * 16}, False, np.random.uniform(-10, 10, (3, 3))),
    (levitate.fields.SphericalHarmonicsExpansion, {'orders': 2}, True, np.random.uniform(-10, 10, 9)),
    (levitate.fields.SphericalHarmonicsExpansion, {'orders': 5}, True, np.random.uniform(-10, 10, 36)),
    (levitate.fields.SphericalHarmonicsExpansionGradient, {'orders': 2}, True, np.random.uniform(-10, 10, (3, 9))),
    (levitate.fields.SphericalHarmonicsExpansionGradient, {'orders': 5}, True, np.random.uniform(-10, 10, (3, 36))),
])
def test_jacobian_accuracy(func, kwargs, take_abs, weight):
    point = func(large_array, weight=weight, position=pos, **kwargs)
    if take_abs:
        point = abs(point)

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
