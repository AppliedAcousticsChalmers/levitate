import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

array = levitate.arrays.RectangularArray(shape=(4, 5))
pos_0 = np.array([0.1, 0.2, 0.3])
pos_1 = np.array([-0.15, 1.27, 0.001])
pos_both = np.stack((pos_0, pos_1), axis=1)
phases = array.focus_phases((pos_0 + pos_1) / 2) + array.signature(stype='twin')
amps = levitate.complex(phases)

spat_ders = array.pressure_derivs(pos_both, orders=3)
ind_ders = np.einsum('i, ji...->ji...', amps, spat_ders)
sum_ders = np.sum(ind_ders, axis=1)
sph_harm = array.spherical_harmonics(pos_both, orders=6)
sph_harm_ind = np.einsum('i, ji...->ji...', amps, sph_harm)
sph_harm_sum = np.sum(sph_harm_ind, axis=1)

requirements_both = {
    'pressure_derivs_summed': sum_ders, 'pressure_derivs_individual': ind_ders,
    'spherical_harmonics_summed': sph_harm_sum, 'spherical_harmonics_individual': sph_harm_ind
}
requirements_0 = {
    'pressure_derivs_summed': sum_ders[..., 0], 'pressure_derivs_individual': ind_ders[..., 0],
    'spherical_harmonics_summed': sph_harm_sum[..., 0], 'spherical_harmonics_individual': sph_harm_ind[..., 0]
}
requirements_1 = {
    'pressure_derivs_summed': sum_ders[..., 1], 'pressure_derivs_individual': ind_ders[..., 1],
    'spherical_harmonics_summed': sph_harm_sum[..., 1], 'spherical_harmonics_individual': sph_harm_ind[..., 1]
}


# Defines the fields to use for testing.
# Note that the field implementations themselves are tested elsewhere.
fields_to_test = [
    levitate.fields.GorkovPotential,
    levitate.fields.GorkovGradient,
    levitate.fields.GorkovLaplacian,
    levitate.fields.RadiationForceGradient,
    lambda arr: levitate.fields.SphericalHarmonicsForce(arr, orders=5, radius=1e-3),
]


@pytest.mark.parametrize("func", fields_to_test)
def test_Field(func):
    field = func(array)
    calc_values = field.values

    val_0 = field(amps, pos_0)
    val_1 = field(amps, pos_1)
    val_both = field(amps, pos_both)

    np.testing.assert_allclose(val_0, calc_values(requirements_0))
    np.testing.assert_allclose(val_1, calc_values(requirements_1))
    np.testing.assert_allclose(val_both, np.stack([val_0, val_1], axis=field.ndim))


@pytest.mark.parametrize("pos", [pos_0, pos_1, pos_both])
@pytest.mark.parametrize("func", fields_to_test)
def test_FieldPoint(func, pos):
    field = func(array)
    np.testing.assert_allclose((field @ pos)(amps), field(amps, pos))


@pytest.mark.parametrize("func0", fields_to_test)
@pytest.mark.parametrize("func1", fields_to_test)
def test_MultiField(func0, func1):
    field_0 = func0(array)
    field_1 = func1(array)

    val0 = field_0(amps, pos_both)
    val1 = field_1(amps, pos_both)
    val_both = levitate.fields.stack(field_0, field_1)(amps, pos_both)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])


@pytest.mark.parametrize("func0", fields_to_test)
@pytest.mark.parametrize("func1", fields_to_test)
def test_MultiFieldPoint(func0, func1):
    field = levitate.fields.stack(func0(array), func1(array))

    val_field = field(amps, pos_both)
    val_bound = (field @ pos_both)(amps)
    np.testing.assert_allclose(val_field[0], val_bound[0])
    np.testing.assert_allclose(val_field[1], val_bound[1])


@pytest.mark.parametrize("func0", fields_to_test)
@pytest.mark.parametrize("func1", fields_to_test)
@pytest.mark.parametrize("pos0", [pos_0, pos_1])
@pytest.mark.parametrize("pos1", [pos_0, pos_1])
def test_MultiFieldMultiPoint(func0, func1, pos0, pos1):
    field_0 = func0(array) @ pos0
    field_1 = func1(array) @ pos1
    field_both = levitate.fields.stack(field_0, field_1)

    val0 = field_0(amps)
    val1 = field_1(amps)
    val_both = field_both(amps)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])
