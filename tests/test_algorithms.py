import levitate
import levitate.algorithms  # Not a default import for levitate
import numpy as np

# Hardcoded values for the tests were created using the previous jacobian convention inside the cost functions.
# The new jacobian convention is conjugated compared to the previous one, and the return format is different
# for the algorithms compared to the cost functions.
from levitate.materials import Air
Air.c = 343
Air.rho = 1.2

array = levitate.arrays.RectangularArray(shape=(2, 1))
pos_1 = np.array([0.1, 0.2, 0.3])
pos_2 = np.array([-0.15, 1.27, 0.001])
both_pos = np.stack((pos_1, pos_2), axis=1)
array.phases = array.focus_phases((pos_1 + pos_2) / 2)

spat_ders = array.spatial_derivatives(both_pos, orders=3)
ind_ders = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), spat_ders)
sum_ders = np.sum(ind_ders, axis=1)


def test_gorkov_divergence():
    calc_values, calc_jacobians = levitate.algorithms.gorkov_divergence(array)
    val_1 = calc_values(sum_ders[..., 0])
    val_2 = calc_values(sum_ders[..., 1])
    val_12 = calc_values(sum_ders)
    np.testing.assert_allclose(val_1, np.array([2.30070037e-11, -1.62961537e-12, -2.44442306e-12]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    jac_1 = calc_jacobians(sum_ders[..., 0], ind_ders[..., 0])
    jac_2 = calc_jacobians(sum_ders[..., 1], ind_ders[..., 1])
    jac_12 = calc_jacobians(sum_ders, ind_ders)
    np.testing.assert_allclose(-jac_1.imag, np.array([[-1.79047948e-11, 1.79047948e-11], [-9.84604578e-13, 9.84604578e-13], [-1.47690687e-12, 1.47690687e-12]]))
    np.testing.assert_allclose(jac_1.real, np.array([[2.30839871e-11, 2.29300203e-11], [-1.69118632e-12, -1.56804442e-12], [-2.53677948e-12, -2.35206663e-12]]))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_gorkov_laplacian():
    calc_values, calc_jacobians = levitate.algorithms.gorkov_laplacian(array)
    val_1 = calc_values(sum_ders[..., 0])
    val_2 = calc_values(sum_ders[..., 1])
    val_12 = calc_values(sum_ders)
    np.testing.assert_allclose(val_1, np.array([-3.98121194e-10, 8.74737783e-12, 2.98666962e-11]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    jac_1 = calc_jacobians(sum_ders[..., 0], ind_ders[..., 0])
    jac_2 = calc_jacobians(sum_ders[..., 1], ind_ders[..., 1])
    jac_12 = calc_jacobians(sum_ders, ind_ders)
    np.testing.assert_allclose(-jac_1.imag, np.array([[-3.33886801e-10, 3.33886801e-10], [-1.94724287e-11, 1.94724287e-11], [-3.76591861e-11, 3.76591861e-11]]))
    np.testing.assert_allclose(jac_1.real, np.array([[-3.98912624e-10, -3.97329763e-10], [8.96724049e-12, 8.52751518e-12], [3.07462056e-11, 2.89871868e-11]]))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_second_order_force():
    calc_values, calc_jacobians = levitate.algorithms.second_order_force(array)
    val_1 = calc_values(sum_ders[..., 0])
    val_2 = calc_values(sum_ders[..., 1])
    val_12 = calc_values(sum_ders)
    np.testing.assert_allclose(val_1, np.array([1.83399145e-10, 4.15099186e-10, 6.22648779e-10]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    jac_1 = calc_jacobians(sum_ders[..., 0], ind_ders[..., 0])
    jac_2 = calc_jacobians(sum_ders[..., 1], ind_ders[..., 1])
    jac_12 = calc_jacobians(sum_ders, ind_ders)
    np.testing.assert_allclose(-jac_1.imag, np.array([[-3.89064704e-10, 3.89064704e-10], [-8.13263002e-10, 8.13263002e-10], [-1.21989450e-09, 1.21989450e-09]]))
    np.testing.assert_allclose(jac_1.real, np.array([[2.03139282e-10, 1.63659008e-10], [4.04354167e-10, 4.25844205e-10], [6.06531251e-10, 6.38766308e-10]]))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))
