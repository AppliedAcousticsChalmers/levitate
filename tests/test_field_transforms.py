import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

array = levitate.arrays.RectangularArray(shape=(4, 5))
pos = np.array([0.1, 0.2, 0.3])
# pos_0 = np.array([0.1, 0.2, 0.3])
# pos_1 = np.array([-0.15, 1.27, 0.001])
# pos_both = np.stack((pos_0, pos_1), axis=1)
phases = array.focus_phases(pos + [1e-3, 1e-3, 1e-3]) + array.signature(stype='twin')
state = levitate.utils.complex(phases)

# Get the requirements manually, so that they dont have to be recomputed for
# every test case.
pressure_derivs_0 = array.request({'pressure_derivs': 0}, pos)
pressure_derivs_1 = array.request({'pressure_derivs': 1}, pos)
pressure_derivs_2 = array.request({'pressure_derivs': 2}, pos)
pressure_derivs_3 = array.request({'pressure_derivs': 3}, pos)
spherical_harmonics_5 = array.request({'spherical_harmonics': 5}, pos)

# requests_0 = array.request({'pressure_derivs': 3, 'spherical_harmonics': 8}, pos_0)
# requests_1 = array.request({'pressure_derivs': 3, 'spherical_harmonics': 8}, pos_1)
# requests_both = array.request({'pressure_derivs': 3, 'spherical_harmonics': 8}, pos_both)
# requirements_0 = levitate.fields._wrappers.FieldBase.evaluate_requirements(state, requests_0)
# requirements_1 = levitate.fields._wrappers.FieldBase.evaluate_requirements(state, requests_1)
# requirements_both = levitate.fields._wrappers.FieldBase.evaluate_requirements(state, requests_both)


scalar_operations = [
    lambda x, s: -x,
    lambda x, s: abs(x),
    lambda x, s: x.sum() if np.asarray(x).dtype == object else np.sum(x),
    lambda x, s: x + 0,
    lambda x, s: x + s,
    lambda x, s: x + ((1 + 1j) * s),
    lambda x, s: 3 * s + x,
    lambda x, s: x - 0,
    lambda x, s: x - s,
    lambda x, s: x - ((1 + 1j) * s),
    lambda x, s: 0.15 * s - x,
    lambda x, s: x * 0,
    lambda x, s: x * 1,
    lambda x, s: x * 3,
    lambda x, s: 3 * x,
    lambda x, s: x / 1,
    lambda x, s: x / 3,
    lambda x, s: 3 / x,
    lambda x, s: x ** 1,
    lambda x, s: x ** 2,
    lambda x, s: x ** -2,
    lambda x, s: abs(x)**3.2,
    lambda x, s: 1 ** (x / s),
    lambda x, s: 2 ** (x / s),
    lambda x, s: 3.2 ** (x / s),
    lambda x, s: levitate.fields.real(x) if np.asarray(x).dtype == object else np.real(x),
    lambda x, s: levitate.fields.imag(x) if np.asarray(x).dtype == object else np.imag(x),
    lambda x, s: levitate.fields.conjugate(x) if np.asarray(x).dtype == object else np.conjugate(x),
]


@pytest.mark.parametrize("field,scale,requests", [
    (levitate.fields.Pressure(array), 1, pressure_derivs_0),
    (levitate.fields.Velocity(array), 1e-3, pressure_derivs_1),
    (levitate.fields.GorkovPotential(array), 1e-13, pressure_derivs_1),
    (levitate.fields.GorkovGradient(array), 1e-11, pressure_derivs_2),
    (levitate.fields.RadiationForceGradient(array), 1e-8, pressure_derivs_3),
    (levitate.fields.SphericalHarmonicsExpansion(array, orders=5), 10, spherical_harmonics_5),
])
@pytest.mark.parametrize("operation", scalar_operations)
def test_single_field_transform(field, scale, requests, operation):
    transformed_field = operation(field, scale)
    as_string = str(transformed_field)  # noqa: F841
    requirements = field.evaluate_requirements(state, requests)

    manual_values = operation(field.values(requirements), scale)
    implemented_values = transformed_field.values(requirements)

    np.testing.assert_allclose(manual_values, implemented_values)

    jacobians = transformed_field.jacobians(requirements)
    real_jacobians = np.real(jacobians / state)
    imag_jacobians = -np.imag(jacobians / state)
    phase_jacobians = -np.imag(jacobians)
    amplitude_jacobians = np.real(jacobians) / np.abs(state)

    numer_real_jacobians = np.zeros(jacobians.shape)
    numer_imag_jacobians = np.zeros(jacobians.shape)
    numer_phase_jacobians = np.zeros(jacobians.shape)
    numer_amplitude_jacobians = np.zeros(jacobians.shape)

    reals = np.real(state).copy()
    imags = np.imag(state).copy()
    phases = np.angle(state).copy()
    amps = np.abs(state).copy()

    delta = 1e-6
    for idx in range(array.num_transducers):
        # Real
        state[idx] = (reals[idx] + delta) + 1j * imags[idx]
        requrements = field.evaluate_requirements(state, requests)
        upper_value = operation(field.values(requrements), scale)

        state[idx] = (reals[idx] - delta) + 1j * imags[idx]
        requrements = field.evaluate_requirements(state, requests)
        lower_value = operation(field.values(requrements), scale)

        state[idx] = reals[idx] + 1j * imags[idx]
        numer_real_jacobians[..., idx] = np.real(upper_value - lower_value) / (2 * delta)

        # Imag
        state[idx] = reals[idx] + 1j * (imags[idx] + delta)
        requrements = field.evaluate_requirements(state, requests)
        upper_value = operation(field.values(requrements), scale)

        state[idx] = reals[idx] + 1j * (imags[idx] - delta)
        requrements = field.evaluate_requirements(state, requests)
        lower_value = operation(field.values(requrements), scale)

        state[idx] = reals[idx] + 1j * imags[idx]
        numer_imag_jacobians[..., idx] = np.real(upper_value - lower_value) / (2 * delta)

        # Phase
        state[idx] = amps[idx] * np.exp(1j * (phases[idx] + delta))
        requrements = field.evaluate_requirements(state, requests)
        upper_value = operation(field.values(requrements), scale)

        state[idx] = amps[idx] * np.exp(1j * (phases[idx] - delta))
        requrements = field.evaluate_requirements(state, requests)
        lower_value = operation(field.values(requrements), scale)

        state[idx] = reals[idx] + 1j * imags[idx]
        numer_phase_jacobians[..., idx] = np.real(upper_value - lower_value) / (2 * delta)

        # amplitude
        state[idx] = (amps[idx] + delta) * np.exp(1j * phases[idx])
        requrements = field.evaluate_requirements(state, requests)
        upper_value = operation(field.values(requrements), scale)

        state[idx] = (amps[idx] - delta) * np.exp(1j * phases[idx])
        requrements = field.evaluate_requirements(state, requests)
        lower_value = operation(field.values(requrements), scale)

        state[idx] = reals[idx] + 1j * imags[idx]
        numer_amplitude_jacobians[..., idx] = np.real(upper_value - lower_value) / (2 * delta)

    atol = np.mean(np.abs(np.stack([
        real_jacobians, imag_jacobians, amplitude_jacobians, phase_jacobians,
        numer_real_jacobians, numer_imag_jacobians, numer_amplitude_jacobians, numer_phase_jacobians,
    ]))) * 1e-6
    np.testing.assert_allclose(real_jacobians, numer_real_jacobians, atol=atol)
    np.testing.assert_allclose(imag_jacobians, numer_imag_jacobians, atol=atol)
    np.testing.assert_allclose(amplitude_jacobians, numer_amplitude_jacobians, atol=atol)
    np.testing.assert_allclose(phase_jacobians, numer_phase_jacobians, atol=atol)
