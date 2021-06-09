import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

pos = np.array([5, -2, 80]) * 1e-3
array = levitate.arrays.RectangularArray(shape=2)
phases = array.focus_phases(pos) + array.signature(stype='twin')
amps = levitate.utils.complex(phases)


def test_minimize_phases_amplitudes():
    trap = abs(levitate.fields.Pressure(array)) * 1 @ pos + levitate.fields.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array, start_values=amps)
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=True, start_values=0.5 * amps, basinhopping=3)
    result = levitate.optimization.minimize(trap, array, constrain_transducers=[0, 3])


def test_minimize_sequence():
    trap = abs(levitate.fields.Pressure(array)) * 1 @ pos + levitate.fields.RadiationForceStiffness(array) * (1, 1, 1) @ pos
    result = levitate.optimization.minimize(trap, array, variable_amplitudes=[False, True], start_values=0.5 * amps)
    quiet_zone = (abs(levitate.fields.Pressure(array)) * 1 + abs(levitate.fields.Velocity(array)) * (1, 1, 1)) @ (np.array([-5, -2, 60]) * 1e-3)
    result = levitate.optimization.minimize([trap, trap + quiet_zone], array)
    result, status = levitate.optimization.minimize([trap, trap + quiet_zone], array, basinhopping=True, minimize_kwargs={'tol': 1e-6}, callback=lambda **kwargs: False, return_optim_status=True)


large_array = levitate.arrays.RectangularArray(shape=8)
phases = np.random.uniform(-np.pi, np.pi, large_array.num_transducers)
magnitudes = np.random.uniform(0.5, 1, large_array.num_transducers)
cplx_amps = levitate.utils.complex(phases, magnitudes)


@pytest.mark.parametrize("point", [
    levitate.fields.GorkovPotential(large_array) @ pos,   
    levitate.fields.GorkovGradient(large_array).sum() @ pos,
    levitate.fields.GorkovLaplacian(large_array).sum() @ pos,
    levitate.fields.RadiationForce(large_array).sum() @ pos,
    levitate.fields.RadiationForceStiffness(large_array).sum() @ pos,
    levitate.fields.RadiationForceGradient(large_array).sum() @ pos,
    abs(levitate.fields.Pressure(large_array)) @ pos,
    abs(levitate.fields.Velocity(large_array)).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceDecomposition(large_array, radius=1e-3, orders=1).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceDecomposition(large_array, radius=1e-3, orders=5).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceDecomposition(large_array, radius=1e-3, orders=12).sum() @ pos,
    levitate.fields.SphericalHarmonicsForce(large_array, orders=4, radius=large_array.k * 6).sum() @ pos,
    levitate.fields.SphericalHarmonicsForce(large_array, orders=7, radius=large_array.k * 8).sum() @ pos,
    levitate.fields.SphericalHarmonicsForce(large_array, orders=16, radius=large_array.k * 19).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradientDecomposition(large_array, orders=1, radius=large_array.k * 2).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradientDecomposition(large_array, orders=4, radius=large_array.k * 6).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradientDecomposition(large_array, orders=12, radius=large_array.k * 18).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradient(large_array, orders=4, radius=large_array.k * 6).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradient(large_array, orders=7, radius=large_array.k * 9).sum() @ pos,
    levitate.fields.SphericalHarmonicsForceGradient(large_array, orders=12, radius=large_array.k * 16).sum() @ pos,
    abs(levitate.fields.SphericalHarmonicsExpansion(large_array, orders=2).sum()) @ pos,
    abs(levitate.fields.SphericalHarmonicsExpansion(large_array, orders=5).sum()) @ pos,
    abs(levitate.fields.SphericalHarmonicsExpansionGradient(large_array, orders=2).sum()) @ pos,
    abs(levitate.fields.SphericalHarmonicsExpansionGradient(large_array, orders=5).sum()) @ pos,
    (levitate.fields.GorkovLaplacian(large_array).sum() * 1e9 + abs(levitate.fields.Pressure(large_array))) @ pos,
    abs(levitate.fields.Pressure(large_array)) @ [12e-3, -25e-3, 60e-3] + abs(levitate.fields.Pressure(large_array)) @ [-4e-3, 6.45e-3, 80e-3],
    abs(levitate.fields.Velocity(large_array).sum() @ pos),
    abs(levitate.fields.Pressure(large_array) * levitate.fields.RadiationForce(large_array).sum()) @ pos,
    levitate.fields.sum_of_eigenvalues(levitate.fields.RadiationForceGradient(large_array) @ pos),
])
def test_jacobian_accuracy(point):
    values_at_operating_point, jacobians_at_operating_point = point.cost_function(cplx_amps)

    phase_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        phases[idx] += 1e-6
        upper_val = point(levitate.utils.complex(phases, magnitudes))
        phases[idx] -= 2e-6
        lower_val = point(levitate.utils.complex(phases, magnitudes))
        phases[idx] += 1e-6
        phase_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(phase_jacobians, -jacobians_at_operating_point.imag, 1e-5, 1e-8)

    amplitude_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        magnitudes[idx] += 1e-6
        upper_val = point(levitate.utils.complex(phases, magnitudes))
        magnitudes[idx] -= 2e-6
        lower_val = point(levitate.utils.complex(phases, magnitudes))
        magnitudes[idx] += 1e-6
        amplitude_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(amplitude_jacobians, jacobians_at_operating_point.real / magnitudes, 1e-5, 1e-8)

    real_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        cplx_amps[idx] += 1e-6
        upper_val = point(cplx_amps)
        cplx_amps[idx] -= 2e-6
        lower_val = point(cplx_amps)
        cplx_amps[idx] += 1e-6
        real_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(real_jacobians, np.real(jacobians_at_operating_point / cplx_amps), 1e-5, 1e-8)

    imag_jacobians = np.zeros(large_array.num_transducers)
    for idx in range(large_array.num_transducers):
        cplx_amps[idx] += 1j * 1e-6
        upper_val = point(cplx_amps)
        cplx_amps[idx] -= 1j * 2e-6
        lower_val = point(cplx_amps)
        cplx_amps[idx] += 1j * 1e-6
        imag_jacobians[idx] = (upper_val - lower_val) / 2e-6
    np.testing.assert_allclose(imag_jacobians, -np.imag(jacobians_at_operating_point / cplx_amps), 1e-5, 1e-8)
