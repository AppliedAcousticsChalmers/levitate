import levitate
import numpy as np
import pytest

# Hardcoded values for the tests were created using the previous jacobian convention inside the cost functions.
# The new jacobian convention is conjugated compared to the previous one, and the return format is different
# for the fields compared to the cost functions.
from levitate.materials import air
air.c = 343
air.rho = 1.2

large_array = levitate.arrays.RectangularArray(shape=(9, 8))
pos = np.array([-23, 12, 34.1]) * 1e-3
large_array.phases = large_array.focus_phases(pos) + large_array.signature(stype='vortex')


def test_gorkov_differentiations():
    amps = large_array.complex_amplitudes
    potential = levitate.fields.GorkovPotential(large_array)
    gradient = levitate.fields.GorkovGradient(large_array)
    delta = 1e-9
    implemented_gradient = gradient(amps, pos)

    x_plus = pos + np.array([delta, 0, 0])
    x_minus = pos - np.array([delta, 0, 0])
    y_plus = pos + np.array([0, delta, 0])
    y_minus = pos - np.array([0, delta, 0])
    z_plus = pos + np.array([0, 0, delta])
    z_minus = pos - np.array([0, 0, delta])

    dUdx = (potential(amps, x_plus) - potential(amps, x_minus)) / (2 * delta)
    dUdy = (potential(amps, y_plus) - potential(amps, y_minus)) / (2 * delta)
    dUdz = (potential(amps, z_plus) - potential(amps, z_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_gradient[0], dUdx)
    np.testing.assert_allclose(implemented_gradient[1], dUdy)
    np.testing.assert_allclose(implemented_gradient[2], dUdz)

    implemented_laplacian = levitate.fields.GorkovLaplacian(large_array)(amps, pos)
    d2Udx2 = (gradient(amps, x_plus)[0] - gradient(amps, x_minus)[0]) / (2 * delta)
    d2Udy2 = (gradient(amps, y_plus)[1] - gradient(amps, y_minus)[1]) / (2 * delta)
    d2Udz2 = (gradient(amps, z_plus)[2] - gradient(amps, z_minus)[2]) / (2 * delta)
    np.testing.assert_allclose(implemented_laplacian[0], d2Udx2)
    np.testing.assert_allclose(implemented_laplacian[1], d2Udy2)
    np.testing.assert_allclose(implemented_laplacian[2], d2Udz2)


def test_RadiationForce_implementations():
    amps = large_array.complex_amplitudes
    force = levitate.fields.RadiationForce(large_array)
    stiffness = levitate.fields.RadiationForceStiffness(large_array)
    gradient = levitate.fields.RadiationForceGradient(large_array)
    curl = levitate.fields.RadiationForceCurl(large_array)

    delta = 1e-9
    x_plus = pos + np.array([delta, 0, 0])
    x_minus = pos - np.array([delta, 0, 0])
    y_plus = pos + np.array([0, delta, 0])
    y_minus = pos - np.array([0, delta, 0])
    z_plus = pos + np.array([0, 0, delta])
    z_minus = pos - np.array([0, 0, delta])

    dFdx = (force(amps, x_plus) - force(amps, x_minus)) / (2 * delta)
    dFdy = (force(amps, y_plus) - force(amps, y_minus)) / (2 * delta)
    dFdz = (force(amps, z_plus) - force(amps, z_minus)) / (2 * delta)

    implemented_stiffness = stiffness(amps, pos)
    np.testing.assert_allclose(implemented_stiffness, [dFdx[0], dFdy[1], dFdz[2]])

    implemented_curl = curl(amps, pos)
    np.testing.assert_allclose(implemented_curl, [dFdy[2] - dFdz[1], dFdz[0] - dFdx[2], dFdx[1] - dFdy[0]])

    implemented_gradient = gradient(amps, pos)
    np.testing.assert_allclose(implemented_gradient, np.stack([dFdx, dFdy, dFdz], axis=1))


array = levitate.arrays.RectangularArray(shape=(2, 1))
pos_1 = np.array([0.1, 0.2, 0.3])
pos_2 = np.array([-0.15, 1.27, 0.001])
both_pos = np.stack((pos_1, pos_2), axis=1)
array.phases = array.focus_phases((pos_1 + pos_2) / 2)

spat_ders = array.pressure_derivs(both_pos, orders=3)
ind_ders = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), spat_ders)
sum_ders = np.sum(ind_ders, axis=1)
sph_harm = array.spherical_harmonics(both_pos, orders=15)
ind_harms = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), sph_harm)
sum_harms = np.sum(ind_harms, axis=1)

requirements = dict(
    pressure_derivs_summed=sum_ders,
    pressure_derivs_individual=ind_ders,
    spherical_harmonics_summed=sum_harms,
    spherical_harmonics_individual=ind_harms,
)


@pytest.mark.parametrize("field, kwargs, value_at_pos_1, jacobian_at_pos_1", [
    (levitate.fields.Pressure, {},
        12.068916910969428 + 8.065242302836108j,
        [-2.014671808191e+00 + 1.584976293557e+01j, +1.408358871916e+01 - 7.784520632737e+00j]
     ),
    (levitate.fields.Velocity, {},
        [+7.327894037353e-03 + 5.975043873706e-03j, +1.570939268938e-02 + 1.042127010721e-02j, +2.356408903408e-02 + 1.563190516081e-02j],
        [[-1.407708646094e-03 + 1.076187601421e-02j, +8.735602683448e-03 - 4.786832140508e-03j], [-2.681349802084e-03 + 2.049881145565e-02j, +1.839074249147e-02 - 1.007754134844e-02j], [-4.022024703126e-03 + 3.074821718347e-02j, +2.758611373720e-02 - 1.511631202266e-02j]]
     ),
    (levitate.fields.GorkovPotential, {},
        -6.19402404e-13,
        [-6.08626619e-13 - 1.21656276e-12j, -6.30178190e-13 + 1.21656276e-12j],
     ),
    (levitate.fields.GorkovGradient, {},
        [2.30070037e-11, -1.62961537e-12, -2.44442306e-12],
        [[2.30839871e-11 + 1.79047948e-11j, 2.29300203e-11 - 1.79047948e-11j], [-1.69118632e-12 + 9.84604578e-13j, -1.56804442e-12 - 9.84604578e-13j], [-2.53677948e-12 + 1.47690687e-12j, -2.35206663e-12 - 1.47690687e-12j]],
     ),
    (levitate.fields.GorkovLaplacian, {},
        [-3.98121194e-10, 8.74737783e-12, 2.98666962e-11],
        [[-3.98912624e-10 + 3.33886801e-10j, -3.97329763e-10 - 3.33886801e-10j], [8.96724049e-12 + 1.94724287e-11j, 8.52751518e-12 - 1.94724287e-11j], [3.07462056e-11 + 3.76591861e-11j, 2.89871868e-11 - 3.76591861e-11j]],
     ),
    (levitate.fields.RadiationForce, {},
        [1.83399145e-10, 4.15099186e-10, 6.22648779e-10],
        [[2.03139282e-10 + 3.89064704e-10j, 1.63659008e-10 - 3.89064704e-10j], [4.04354167e-10 + 8.13263002e-10j, 4.25844205e-10 - 8.13263002e-10j], [6.06531251e-10 + 1.21989450e-09j, 6.38766308e-10 - 1.21989450e-09j]],
     ),
    (levitate.fields.RadiationForceStiffness, {},
        [-5.377916683452e-09, +2.563628836166e-09, +3.173794966908e-09],
        [[-5.519896755958e-09 - 2.548378109047e-09j, -5.235936610946e-09 + 2.548378109047e-09j], [+2.586556763000e-09 + 2.227770294366e-09j, +2.540700909331e-09 - 2.227770294367e-09j], [+3.292539171218e-09 - 7.041060162908e-11j, +3.055050762598e-09 + 7.041060162897e-11j]]
     ),
    (levitate.fields.RadiationForceCurl, {},
        [+2.615958687199e-23, +2.391674937388e-08, -1.594449958258e-08],
        [[+1.242838870361e-22 - 2.905471901593e-23j, -9.150685526368e-24 + 7.961613395823e-24j], [+2.391674937388e-08 + 1.748702221918e-08j, +2.391674937388e-08 - 1.748702221918e-08j], [-1.594449958258e-08 - 1.165801481279e-08j, -1.594449958258e-08 + 1.165801481279e-08j]]
     ),
    (levitate.fields.RadiationForceGradient, {},
        [[-5.377916683452e-09, +2.991379534396e-10, +4.487069301596e-10], [-1.564536162914e-08, +2.563628836166e-09, +7.321993568902e-10], [-2.346804244372e-08, +7.321993568902e-10, +3.173794966908e-09]],
        None
     ),
    (levitate.fields.SphericalHarmonicsForce, {'orders': 12},
        [+9.950696205718e-11, +2.697812005596e-10, +4.046718008394e-10],
        None,
     ),
])
def test_field(field, kwargs, value_at_pos_1, jacobian_at_pos_1):
    field = field(array, **kwargs).field

    val_1 = field.values(**{key: requirements[key][..., 0] for key in field.values_require})
    val_2 = field.values(**{key: requirements[key][..., 1] for key in field.values_require})
    val_12 = field.values(**{key: requirements[key] for key in field.values_require})
    np.testing.assert_allclose(val_1, np.array(value_at_pos_1), atol=1e-20)
    np.testing.assert_allclose(val_12, np.stack([val_1, val_2], -1))

    if jacobian_at_pos_1 is not None:
        jac_1 = field.jacobians(**{key: requirements[key][..., 0] for key in field.jacobians_require})
        jac_2 = field.jacobians(**{key: requirements[key][..., 1] for key in field.jacobians_require})
        jac_12 = field.jacobians(**{key: requirements[key] for key in field.jacobians_require})
        np.testing.assert_allclose(jac_1, jacobian_at_pos_1, atol=1e-20)
        np.testing.assert_allclose(jac_12, np.stack([jac_1, jac_2], -1))
