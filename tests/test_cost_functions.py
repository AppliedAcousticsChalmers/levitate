import levitate
import numpy as np

# Tests created with these air properties
from levitate.materials import Air
Air.c = 343
Air.rho = 1.2

array = levitate.arrays.TransducerArray(shape=(2, 1))
pos_1 = np.array([0.1, 0.2, 0.3])
pos_2 = np.array([-0.15, 1.27, 0.001])
both_pos = np.stack((pos_1, pos_2), axis=0)
array.phases = array.focus_phases((pos_1 + pos_2) / 2)


def test_gorkov_divergence():
    func_1 = levitate.cost_functions.gorkov_divergence(array, pos_1, weights=None)
    func_2 = levitate.cost_functions.gorkov_divergence(array, pos_2, weights=None)
    func_12 = levitate.cost_functions.gorkov_divergence(array, both_pos, weights=None)
    val_1 = func_1(array.phases_amplitudes)
    val_2 = func_2(array.phases_amplitudes)
    val_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([2.30070037e-11, -1.62961537e-12, -2.44442306e-12]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    func_1 = levitate.cost_functions.gorkov_divergence(array, pos_1, weights=False)
    func_2 = levitate.cost_functions.gorkov_divergence(array, pos_2, weights=False)
    func_12 = levitate.cost_functions.gorkov_divergence(array, both_pos, weights=False)
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([2.30070037e-11, -1.62961537e-12, -2.44442306e-12]))
    np.testing.assert_allclose(jac_1, np.array([[-1.79047948e-11, 1.79047948e-11, 2.30839871e-11, 2.29300203e-11], [-9.84604578e-13, 9.84604578e-13, -1.69118632e-12, -1.56804442e-12], [-1.47690687e-12, 1.47690687e-12, -2.53677948e-12, -2.35206663e-12]]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))

    func_1 = levitate.cost_functions.gorkov_divergence(array, pos_1, weights=(1, 1, 1))
    func_2 = levitate.cost_functions.gorkov_divergence(array, pos_2, weights=(1, 1, 1))
    func_12 = levitate.cost_functions.gorkov_divergence(array, both_pos, weights=(1, 1, 1))
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, 1.893296525220701e-11)
    np.testing.assert_allclose(jac_1, np.array([-2.03663063e-11, 2.03663063e-11, 1.88560212e-11, 1.90099093e-11]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_gorkov_laplacian():
    func_1 = levitate.cost_functions.gorkov_laplacian(array, pos_1, weights=None)
    func_2 = levitate.cost_functions.gorkov_laplacian(array, pos_2, weights=None)
    func_12 = levitate.cost_functions.gorkov_laplacian(array, both_pos, weights=None)
    val_1 = func_1(array.phases_amplitudes)
    val_2 = func_2(array.phases_amplitudes)
    val_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([-3.98121194e-10, 8.74737783e-12, 2.98666962e-11]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    func_1 = levitate.cost_functions.gorkov_laplacian(array, pos_1, weights=False)
    func_2 = levitate.cost_functions.gorkov_laplacian(array, pos_2, weights=False)
    func_12 = levitate.cost_functions.gorkov_laplacian(array, both_pos, weights=False)
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([-3.98121194e-10, 8.74737783e-12, 2.98666962e-11]))
    np.testing.assert_allclose(jac_1, np.array([[-3.33886801e-10, 3.33886801e-10, -3.98912624e-10, -3.97329763e-10], [-1.94724287e-11, 1.94724287e-11, 8.96724049e-12, 8.52751518e-12], [-3.76591861e-11, 3.76591861e-11, 3.07462056e-11, 2.89871868e-11]]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))

    func_1 = levitate.cost_functions.gorkov_laplacian(array, pos_1, weights=(1, 1, 1))
    func_2 = levitate.cost_functions.gorkov_laplacian(array, pos_2, weights=(1, 1, 1))
    func_12 = levitate.cost_functions.gorkov_laplacian(array, both_pos, weights=(1, 1, 1))
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, -3.595071196216751e-10)
    np.testing.assert_allclose(jac_1, np.array([-3.91018416e-10, 3.91018416e-10, -3.59199178e-10, -3.59815061e-10]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_second_order_force():
    func_1 = levitate.cost_functions.second_order_force(array, pos_1, weights=None)
    func_2 = levitate.cost_functions.second_order_force(array, pos_2, weights=None)
    func_12 = levitate.cost_functions.second_order_force(array, both_pos, weights=None)
    val_1 = func_1(array.phases_amplitudes)
    val_2 = func_2(array.phases_amplitudes)
    val_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([1.83399145e-10, 4.15099186e-10, 6.22648779e-10]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    func_1 = levitate.cost_functions.second_order_force(array, pos_1, weights=False)
    func_2 = levitate.cost_functions.second_order_force(array, pos_2, weights=False)
    func_12 = levitate.cost_functions.second_order_force(array, both_pos, weights=False)
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([1.83399145e-10, 4.15099186e-10, 6.22648779e-10]))
    np.testing.assert_allclose(jac_1, np.array([[-3.89064704e-10, 3.89064704e-10, 2.03139282e-10, 1.63659008e-10], [-8.13263002e-10, 8.13263002e-10, 4.04354167e-10, 4.25844205e-10], [-1.21989450e-09, 1.21989450e-09, 6.06531251e-10, 6.38766308e-10]]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))

    func_1 = levitate.cost_functions.second_order_force(array, pos_1, weights=(1, 1, 1))
    func_2 = levitate.cost_functions.second_order_force(array, pos_2, weights=(1, 1, 1))
    func_12 = levitate.cost_functions.second_order_force(array, both_pos, weights=(1, 1, 1))
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, 1.221147110725865e-09)
    np.testing.assert_allclose(jac_1, np.array([-2.42222221e-09, 2.42222221e-09, 1.21402470e-09, 1.22826952e-09]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_second_order_stiffness():
    func_1 = levitate.cost_functions.second_order_stiffness(array, pos_1, weights=None)
    func_2 = levitate.cost_functions.second_order_stiffness(array, pos_2, weights=None)
    func_12 = levitate.cost_functions.second_order_stiffness(array, both_pos, weights=None)
    val_1 = func_1(array.phases_amplitudes)
    val_2 = func_2(array.phases_amplitudes)
    val_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([-5.37791668e-09, 2.56362884e-09, 3.17379497e-09]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    func_1 = levitate.cost_functions.second_order_stiffness(array, pos_1, weights=False)
    func_2 = levitate.cost_functions.second_order_stiffness(array, pos_2, weights=False)
    func_12 = levitate.cost_functions.second_order_stiffness(array, both_pos, weights=False)
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([-5.37791668e-09, 2.56362884e-09, 3.17379497e-09]))
    np.testing.assert_allclose(jac_1, np.array([[2.54837811e-09, -2.54837811e-09, -5.51989676e-09, -5.23593661e-09], [-2.22777029e-09, 2.22777029e-09, 2.58655676e-09, 2.54070091e-09], [7.04106016e-11, -7.04106016e-11, 3.29253917e-09, 3.05505076e-09]]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))

    func_1 = levitate.cost_functions.second_order_stiffness(array, pos_1, weights=(1, 1, 1))
    func_2 = levitate.cost_functions.second_order_stiffness(array, pos_2, weights=(1, 1, 1))
    func_12 = levitate.cost_functions.second_order_stiffness(array, both_pos, weights=(1, 1, 1))
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, 3.595071196216047e-10)
    np.testing.assert_allclose(jac_1, np.array([3.91018416e-10, -3.91018416e-10, 3.59199178e-10, 3.59815061e-10]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_pressure_null():
    func_1 = levitate.cost_functions.pressure_null(array, pos_1, weights=None)
    func_2 = levitate.cost_functions.pressure_null(array, pos_2, weights=None)
    func_12 = levitate.cost_functions.pressure_null(array, both_pos, weights=None)
    val_1 = func_1(array.phases_amplitudes)
    val_2 = func_2(array.phases_amplitudes)
    val_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([12.06891691 + 8.0652423j, -1802.0307781 + 2210.04077511j, -3142.98101855 + 4737.84121587j, -4714.47152782 + 7106.76182381j]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    func_1 = levitate.cost_functions.pressure_null(array, pos_1, weights=False)
    func_2 = levitate.cost_functions.pressure_null(array, pos_2, weights=False)
    func_12 = levitate.cost_functions.pressure_null(array, both_pos, weights=False)
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, np.array([2.10706889e+02, 8.13159515e+06, 3.23254691e+07, 7.27323054e+07]))
    np.testing.assert_allclose(jac_1, np.array([[-4.15076576e+02, 4.15076576e+02, 2.07034544e+02, 2.14379234e+02], [-1.58764044e+07, 1.58764044e+07, 9.82115445e+06, 6.44203586e+06], [-6.36647795e+07, 6.36647795e+07, 3.11989198e+07, 3.34520183e+07], [-1.43245754e+08, 1.43245754e+08, 7.01975696e+07, 7.52670412e+07]]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))

    func_1 = levitate.cost_functions.pressure_null(array, pos_1, weights=(1, 1, 1))
    func_2 = levitate.cost_functions.pressure_null(array, pos_2, weights=(1, 1, 1))
    func_12 = levitate.cost_functions.pressure_null(array, both_pos, weights=(1, 1, 1))
    val_1, jac_1 = func_1(array.phases_amplitudes)
    val_2, jac_2 = func_2(array.phases_amplitudes)
    val_12, jac_12 = func_12(array.phases_amplitudes)
    np.testing.assert_allclose(val_1, 1.1318937e+08)
    np.testing.assert_allclose(jac_1, np.array([-2.22786938e+08, 2.22786938e+08, 1.11217644e+08, 1.15161095e+08]))
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))
    np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))


def test_amplitude_limiting():
    func = levitate.cost_functions.amplitude_limiting(array)
    val, jac = func(array.phases_amplitudes)
    np.testing.assert_allclose(val, 2.000000000000007e-08)
    np.testing.assert_allclose(jac, np.array([0.e+00, 0.e+00, 4.e-06, 4.e-06]))


def test_minimize():
    pos = np.array([5, -2, 80]) * 1e-3
    array = levitate.arrays.TransducerArray(shape=2)
    stiffness = levitate.cost_functions.second_order_stiffness(array, pos, weights=(-1, -1, -1))
    zero_pressure = levitate.cost_functions.pressure_null(array, pos, weights=1e-3)
    quiet_zone = levitate.cost_functions.pressure_null(array, np.array([-5, -2, 60]) * 1e-3, weights=(1, 1 / array.k**2, 1 / array.k**2, 1 / array.k**2))
    array.phases = array.focus_phases(pos) + array.twin_signature()
    result = levitate.cost_functions.minimize([[stiffness, zero_pressure], [stiffness, zero_pressure, quiet_zone]], array, variable_amplitudes=[False, True])
 