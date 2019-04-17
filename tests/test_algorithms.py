import levitate
import numpy as np
import pytest

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

spat_ders = array.pressure_derivs(both_pos, orders=3)
ind_ders = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), spat_ders)
sum_ders = np.sum(ind_ders, axis=1)


def test_gorkov_differentiations():
    amps = array.complex_amplitudes
    potential = levitate.algorithms.gorkov_potential(array)
    gradient = levitate.algorithms.gorkov_gradient(array)
    delta = 1e-6
    implemented_gradient = gradient(amps, pos_1)

    x_plus = pos_1 + np.array([delta, 0, 0])
    x_minus = pos_1 - np.array([delta, 0, 0])
    y_plus = pos_1 + np.array([0, delta, 0])
    y_minus = pos_1 - np.array([0, delta, 0])
    z_plus = pos_1 + np.array([0, 0, delta])
    z_minus = pos_1 - np.array([0, 0, delta])

    dUdx = (potential(amps, x_plus) - potential(amps, x_minus)) / (2 * delta)
    dUdy = (potential(amps, y_plus) - potential(amps, y_minus)) / (2 * delta)
    dUdz = (potential(amps, z_plus) - potential(amps, z_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_gradient[0], dUdx)
    np.testing.assert_allclose(implemented_gradient[1], dUdy)
    np.testing.assert_allclose(implemented_gradient[2], dUdz)

    implemented_laplacian = levitate.algorithms.gorkov_laplacian(array)(amps, pos_1)
    d2Udx2 = (gradient(amps, x_plus)[0] - gradient(amps, x_minus)[0]) / (2 * delta)
    d2Udy2 = (gradient(amps, y_plus)[1] - gradient(amps, y_minus)[1]) / (2 * delta)
    d2Udz2 = (gradient(amps, z_plus)[2] - gradient(amps, z_minus)[2]) / (2 * delta)
    np.testing.assert_allclose(implemented_laplacian[0], d2Udx2)
    np.testing.assert_allclose(implemented_laplacian[1], d2Udy2)
    np.testing.assert_allclose(implemented_laplacian[2], d2Udz2)


@pytest.mark.parametrize("algorithm, value_at_pos_1, real_jacobian_at_pos_1, imag_jacobian_at_pos_1", [
    (levitate.algorithms.gorkov_potential,
        -6.19402404e-13,
        [-6.08626619e-13, -6.30178190e-13],
        [-1.21656276e-12, 1.21656276e-12]
     ),
    (levitate.algorithms.gorkov_gradient,
        [2.30070037e-11, -1.62961537e-12, -2.44442306e-12],
        [[2.30839871e-11, 2.29300203e-11], [-1.69118632e-12, -1.56804442e-12], [-2.53677948e-12, -2.35206663e-12]],
        [[1.79047948e-11, -1.79047948e-11], [9.84604578e-13, -9.84604578e-13], [1.47690687e-12, -1.47690687e-12]]
     ),
    (levitate.algorithms.gorkov_laplacian,
        [-3.98121194e-10, 8.74737783e-12, 2.98666962e-11],
        [[-3.98912624e-10, -3.97329763e-10], [8.96724049e-12, 8.52751518e-12], [3.07462056e-11, 2.89871868e-11]],
        [[3.33886801e-10, -3.33886801e-10], [1.94724287e-11, -1.94724287e-11], [3.76591861e-11, -3.76591861e-11]]
     ),
    (levitate.algorithms.second_order_force,
        [1.83399145e-10, 4.15099186e-10, 6.22648779e-10],
        [[2.03139282e-10, 1.63659008e-10], [4.04354167e-10, 4.25844205e-10], [6.06531251e-10, 6.38766308e-10]],
        [[3.89064704e-10, -3.89064704e-10], [8.13263002e-10, -8.13263002e-10], [1.21989450e-09, -1.21989450e-09]]
     ),
    (levitate.algorithms.pressure_squared_magnitude,
        2.10706889e+02,
        [2.07034544e+02, 2.14379234e+02],
        [4.15076576e+02, -4.15076576e+02],
     ),
    (levitate.algorithms.velocity_squared_magnitude,
        [8.93991803e-05, 3.55387889e-04, 7.99622751e-04],
        [[1.07974283e-04, 7.08240775e-05], [3.43002548e-04, 3.67773230e-04], [7.71755733e-04, 8.27489769e-04]],
        [[0.000174546016, -0.000174546016], [0.000699933899, -0.000699933899], [0.001574851272, -0.001574851272]]
     ),
])
def test_algorithm(algorithm, value_at_pos_1, real_jacobian_at_pos_1, imag_jacobian_at_pos_1):
    algorithm = algorithm(array)
    calc_values, calc_jacobians = algorithm.calc_values, algorithm.calc_jacobians

    val_1 = calc_values(sum_ders[..., 0])
    val_2 = calc_values(sum_ders[..., 1])
    val_12 = calc_values(sum_ders)
    np.testing.assert_allclose(val_1, np.array(value_at_pos_1))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    jac_1 = calc_jacobians(sum_ders[..., 0], ind_ders[..., 0])
    jac_2 = calc_jacobians(sum_ders[..., 1], ind_ders[..., 1])
    jac_12 = calc_jacobians(sum_ders, ind_ders)
    np.testing.assert_allclose(jac_1.real, np.array(real_jacobian_at_pos_1))
    np.testing.assert_allclose(jac_1.imag, np.array(imag_jacobian_at_pos_1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))
