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
array.phases = array.focus_phases((pos_0 + pos_1) / 2) + array.signature(stype='twin')

spat_ders = array.pressure_derivs(pos_both, orders=3)
ind_ders = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), spat_ders)
sum_ders = np.sum(ind_ders, axis=1)
sph_harm = array.spherical_harmonics(pos_both, orders=6)
sph_harm_ind = np.einsum('i, ji...->ji...', array.complex_amplitudes, sph_harm)
sph_harm_sum = np.sum(sph_harm_ind, axis=1)

requirements = {
    'pressure_derivs_summed': sum_ders, 'pressure_derivs_individual': ind_ders,
    'spherical_harmonics_summed': sph_harm_sum, 'spherical_harmonics_individual': sph_harm_ind
}


# Defines the fields to use for testing.
# Note that the field implementations themselves are tested elsewhere.
has_jabobians_fields = [
    levitate.fields.GorkovPotential,
    levitate.fields.GorkovGradient,
    levitate.fields.GorkovLaplacian,
]
no_jacobians_fields = [
    levitate.fields.RadiationForceGradient,
    lambda arr: levitate.fields.SphericalHarmonicsForce(arr, orders=5),
]

values_fields = has_jabobians_fields + no_jacobians_fields


@pytest.mark.parametrize("func", values_fields)
def test_Field(func):
    field = func(array)
    calc_values = field.values

    val_0 = field(array.complex_amplitudes, pos_0)
    val_1 = field(array.complex_amplitudes, pos_1)
    val_both = field(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, calc_values(**{key: requirements[key][..., 0] for key in field.values_require}))
    np.testing.assert_allclose(val_1, calc_values(**{key: requirements[key][..., 1] for key in field.values_require}))
    np.testing.assert_allclose(val_both, np.stack([val_0, val_1], axis=field.ndim))


@pytest.mark.parametrize("pos", [pos_0, pos_1, pos_both])
@pytest.mark.parametrize("func", values_fields)
def test_FieldPoint(func, pos):
    field = func(array)
    np.testing.assert_allclose((field@pos)(array.complex_amplitudes), field(array.complex_amplitudes, pos))


@pytest.mark.parametrize("weight", [1, 1e-3, 1e3])
@pytest.mark.parametrize("func", has_jabobians_fields)
def test_CostField(func, weight):
    field = func(array)
    weight = np.random.uniform(-weight, weight, (1,) * field.ndim)
    field = field * weight
    calc_values, calc_jacobians = field.values, field.jacobians

    raw_0 = calc_values(**{key: requirements[key][..., 0] for key in field.values_require})
    raw_1 = calc_values(**{key: requirements[key][..., 1] for key in field.values_require})
    raw_both = calc_values(**{key: requirements[key] for key in field.values_require})
    val_0 = np.einsum(field._sum_str, weight, raw_0)
    val_1 = np.einsum(field._sum_str, weight, raw_1)
    val_both = np.einsum(field._sum_str, weight, raw_both)

    raw_0 = calc_jacobians(**{key: requirements[key][..., 0] for key in field.jacobians_require})
    raw_1 = calc_jacobians(**{key: requirements[key][..., 1] for key in field.jacobians_require})
    raw_both = calc_jacobians(**{key: requirements[key] for key in field.jacobians_require})
    jac_0 = np.einsum(field._sum_str, weight, raw_0)
    jac_1 = np.einsum(field._sum_str, weight, raw_1)
    jac_both = np.einsum(field._sum_str, weight, raw_both)

    field_val_0, field_jac_0 = field(array.complex_amplitudes, pos_0)
    field_val_1, field_jac_1 = field(array.complex_amplitudes, pos_1)
    field_val_both, field_jac_both = field(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, field_val_0)
    np.testing.assert_allclose(val_1, field_val_1)
    np.testing.assert_allclose(val_both, field_val_both)

    np.testing.assert_allclose(jac_0, field_jac_0)
    np.testing.assert_allclose(jac_1, field_jac_1)
    np.testing.assert_allclose(jac_both, field_jac_both)


@pytest.mark.parametrize("weight", [1, 10])
@pytest.mark.parametrize("pos", [pos_0, pos_1])
@pytest.mark.parametrize("func", has_jabobians_fields)
def test_CostFieldPoint(func, weight, pos):
    field = func(array)
    weight = np.random.uniform(-weight, weight, (1,) * field.ndim)
    field = field * weight

    val, jac = (field@pos)(array.complex_amplitudes)
    val_ub, jac_ub = field(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val, val_ub)
    np.testing.assert_allclose(jac, jac_ub)


@pytest.mark.parametrize("pos", [pos_0, pos_both])
@pytest.mark.parametrize("func", values_fields)
@pytest.mark.parametrize("target_scale", [1, 1e-3, 1e3])
def test_SquaredField(func, target_scale, pos):
    field = func(array)
    target = np.random.uniform(-target_scale, target_scale, (1,) * field.ndim)
    value = np.abs(field(array.complex_amplitudes, pos) - np.asarray(target).reshape(target.shape + (pos.ndim - 1) * (1,)))**2

    np.testing.assert_allclose((field - target)(array.complex_amplitudes, pos), value)


@pytest.mark.parametrize("pos", [pos_0, pos_both])
@pytest.mark.parametrize("func", values_fields)
@pytest.mark.parametrize("target_scale", [1, 10])
def test_SquaredFieldPoint(func, target_scale, pos):
    field = func(array)
    target = np.random.uniform(-target_scale, target_scale, (1,) * field.ndim)
    field = field - target
    np.testing.assert_allclose((field@pos)(array.complex_amplitudes), field(array.complex_amplitudes, pos))


@pytest.mark.parametrize("func", has_jabobians_fields)
@pytest.mark.parametrize("weight", [1, 6])
@pytest.mark.parametrize("target_scale", [1, 1e-4])
def test_SquaredCostField(func, target_scale, weight):
    field = func(array)
    target = np.random.uniform(-target_scale, target_scale, (1,) * field.ndim)
    weight = np.random.uniform(-weight, weight, (1,) * field.ndim)
    field = (field - target) * weight
    calc_values, calc_jacobians = field.values, field.jacobians

    raw_0 = calc_values(**{key: requirements[key][..., 0] for key in field.values_require})
    raw_1 = calc_values(**{key: requirements[key][..., 1] for key in field.values_require})
    raw_both = calc_values(**{key: requirements[key] for key in field.values_require})
    val_0 = np.einsum(field._sum_str, weight, raw_0)
    val_1 = np.einsum(field._sum_str, weight, raw_1)
    val_both = np.einsum(field._sum_str, weight, raw_both)

    raw_0 = calc_jacobians(**{key: requirements[key][..., 0] for key in field.jacobians_require})
    raw_1 = calc_jacobians(**{key: requirements[key][..., 1] for key in field.jacobians_require})
    raw_both = calc_jacobians(**{key: requirements[key] for key in field.jacobians_require})
    jac_0 = np.einsum(field._sum_str, weight, raw_0)
    jac_1 = np.einsum(field._sum_str, weight, raw_1)
    jac_both = np.einsum(field._sum_str, weight, raw_both)

    field_val_0, field_jac_0 = field(array.complex_amplitudes, pos_0)
    field_val_1, field_jac_1 = field(array.complex_amplitudes, pos_1)
    field_val_both, field_jac_both = field(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, field_val_0)
    np.testing.assert_allclose(val_1, field_val_1)
    np.testing.assert_allclose(val_both, field_val_both)

    np.testing.assert_allclose(jac_0, field_jac_0)
    np.testing.assert_allclose(jac_1, field_jac_1)
    np.testing.assert_allclose(jac_both, field_jac_both)


@pytest.mark.parametrize("func", has_jabobians_fields)
@pytest.mark.parametrize("weight", [1, 200])
@pytest.mark.parametrize("target_scale", [1, 30])
@pytest.mark.parametrize("pos", [pos_0, pos_both])
def test_SquaredCostFieldPoint(func, weight, target_scale, pos):
    field = func(array)
    target = np.random.uniform(-target_scale, target_scale, (1,) * field.ndim)
    weight = np.random.uniform(-weight, weight, (1,) * field.ndim)
    field = (field - target) * weight
    # field = (func(array) - target) * weight
    val, jac = (field@pos)(array.complex_amplitudes)
    val_ub, jac_ub = field(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val, val_ub)
    np.testing.assert_allclose(jac, jac_ub)


@pytest.mark.parametrize("func0", values_fields)
@pytest.mark.parametrize("func1", values_fields)
def test_MultiField(func0, func1):
    field_0 = func0(array)
    field_1 = func1(array)

    val0 = field_0(array.complex_amplitudes, pos_both)
    val1 = field_1(array.complex_amplitudes, pos_both)
    val_both = (field_0 + field_1)(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])


@pytest.mark.parametrize("func0", values_fields)
@pytest.mark.parametrize("func1", values_fields)
def test_MultiFieldPoint(func0, func1):
    field = func0(array) + func1(array)

    val_field = field(array.complex_amplitudes, pos_both)
    val_bound = (field@pos_both)(array.complex_amplitudes)
    np.testing.assert_allclose(val_field[0], val_bound[0])
    np.testing.assert_allclose(val_field[1], val_bound[1])


@pytest.mark.parametrize("func0", has_jabobians_fields)
@pytest.mark.parametrize("func1", has_jabobians_fields)
@pytest.mark.parametrize("pos", [pos_0])
@pytest.mark.parametrize("weight0", [1, 1e-3, 1e3])
@pytest.mark.parametrize("weight1", [1, 1e-3, 1e3])
def test_MultiCostField(func0, func1, pos, weight0, weight1):
    field_0 = func0(array)
    weight0 = np.random.uniform(-weight0, weight0, (1,) * field_0.ndim)
    field_0 = field_0 * weight0

    field_1 = func1(array)
    weight1 = np.random.uniform(-weight1, weight1, (1,) * field_1.ndim)
    field_1 = field_1 * weight1

    val0, jac0 = field_0(array.complex_amplitudes, pos)
    val1, jac1 = field_1(array.complex_amplitudes, pos)
    val_both, jac_both = (field_0 + field_1)(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)


@pytest.mark.parametrize("func0", has_jabobians_fields)
@pytest.mark.parametrize("func1", has_jabobians_fields)
@pytest.mark.parametrize("pos", [pos_0, pos_1])
@pytest.mark.parametrize("weight0", [1, 8])
@pytest.mark.parametrize("weight1", [1, 1e-5])
def test_MultiCostFieldPoint(func0, func1, pos, weight0, weight1):
    field_0 = func0(array)
    weight0 = np.random.uniform(-weight0, weight0, (1,) * field_0.ndim)
    field_0 = field_0 * weight0 @ pos

    field_1 = func1(array)
    weight1 = np.random.uniform(-weight1, weight1, (1,) * field_1.ndim)
    field_1 = field_1 * weight1 @ pos

    val0, jac0 = field_0(array.complex_amplitudes)
    val1, jac1 = field_1(array.complex_amplitudes)
    val_both, jac_both = (field_0 + field_1)(array.complex_amplitudes)
    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)


@pytest.mark.parametrize("func0", values_fields)
@pytest.mark.parametrize("func1", values_fields)
@pytest.mark.parametrize("pos0", [pos_0, pos_1])
@pytest.mark.parametrize("pos1", [pos_0, pos_1])
def test_MultiFieldMultiPoint(func0, func1, pos0, pos1):
    field_0 = func0(array)@pos0
    field_1 = func1(array)@pos1
    field_both = field_0 + field_1

    val0 = field_0(array.complex_amplitudes)
    val1 = field_1(array.complex_amplitudes)
    val_both = field_both(array.complex_amplitudes)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])


@pytest.mark.parametrize("func0", has_jabobians_fields)
@pytest.mark.parametrize("func1", has_jabobians_fields)
@pytest.mark.parametrize("pos0", [pos_0, pos_1])
@pytest.mark.parametrize("pos1", [pos_0, pos_1])
@pytest.mark.parametrize("weight0", [1, 1e-3])
@pytest.mark.parametrize("weight1", [1, 1e3])
def test_MultiCostFieldMultiPoint(func0, func1, pos0, pos1, weight0, weight1):
    field_0 = func0(array)
    weight0 = np.random.uniform(-weight0, weight0, (1,) * field_0.ndim)
    field_0 = field_0 * weight0 @ pos0

    field_1 = func1(array)
    weight1 = np.random.uniform(-weight1, weight1, (1,) * field_1.ndim)
    field_1 = field_1 * weight1 @ pos1

    field_both = field_0 + field_1

    val0, jac0 = field_0(array.complex_amplitudes)
    val1, jac1 = field_1(array.complex_amplitudes)
    val_both, jac_both = field_both(array.complex_amplitudes)

    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)
