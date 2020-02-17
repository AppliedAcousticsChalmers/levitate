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
phases = large_array.focus_phases(pos) + large_array.signature(stype='vortex')
amps_large = levitate.utils.complex(phases)


def test_gorkov_differentiations():
    potential = levitate.fields.GorkovPotential(large_array)
    gradient = levitate.fields.GorkovGradient(large_array)
    delta = 1e-9
    implemented_gradient = gradient(amps_large, pos)

    x_plus = pos + np.array([delta, 0, 0])
    x_minus = pos - np.array([delta, 0, 0])
    y_plus = pos + np.array([0, delta, 0])
    y_minus = pos - np.array([0, delta, 0])
    z_plus = pos + np.array([0, 0, delta])
    z_minus = pos - np.array([0, 0, delta])

    dUdx = (potential(amps_large, x_plus) - potential(amps_large, x_minus)) / (2 * delta)
    dUdy = (potential(amps_large, y_plus) - potential(amps_large, y_minus)) / (2 * delta)
    dUdz = (potential(amps_large, z_plus) - potential(amps_large, z_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_gradient[0], dUdx)
    np.testing.assert_allclose(implemented_gradient[1], dUdy)
    np.testing.assert_allclose(implemented_gradient[2], dUdz)

    implemented_laplacian = levitate.fields.GorkovLaplacian(large_array)(amps_large, pos)
    d2Udx2 = (gradient(amps_large, x_plus)[0] - gradient(amps_large, x_minus)[0]) / (2 * delta)
    d2Udy2 = (gradient(amps_large, y_plus)[1] - gradient(amps_large, y_minus)[1]) / (2 * delta)
    d2Udz2 = (gradient(amps_large, z_plus)[2] - gradient(amps_large, z_minus)[2]) / (2 * delta)
    np.testing.assert_allclose(implemented_laplacian[0], d2Udx2)
    np.testing.assert_allclose(implemented_laplacian[1], d2Udy2)
    np.testing.assert_allclose(implemented_laplacian[2], d2Udz2)


def test_RadiationForce_implementations():
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

    dFdx = (force(amps_large, x_plus) - force(amps_large, x_minus)) / (2 * delta)
    dFdy = (force(amps_large, y_plus) - force(amps_large, y_minus)) / (2 * delta)
    dFdz = (force(amps_large, z_plus) - force(amps_large, z_minus)) / (2 * delta)

    implemented_stiffness = stiffness(amps_large, pos)
    np.testing.assert_allclose(implemented_stiffness, [dFdx[0], dFdy[1], dFdz[2]])

    implemented_curl = curl(amps_large, pos)
    np.testing.assert_allclose(implemented_curl, [dFdy[2] - dFdz[1], dFdz[0] - dFdx[2], dFdx[1] - dFdy[0]])

    implemented_gradient = gradient(amps_large, pos)
    np.testing.assert_allclose(implemented_gradient, np.stack([dFdx, dFdy, dFdz], axis=1))


def test_SphericalHarmonicsExpansions():
    orders = 8
    S = levitate.fields.SphericalHarmonicsExpansion(large_array, orders=orders)
    dS = levitate.fields.SphericalHarmonicsExpansionGradient(large_array, orders=orders)

    delta = 1e-8
    xp = pos + [delta, 0, 0]
    xm = pos - [delta, 0, 0]
    yp = pos + [0, delta, 0]
    ym = pos - [0, delta, 0]
    zp = pos + [0, 0, delta]
    zm = pos - [0, 0, delta]

    dSdx = (S(amps_large, xp) - S(amps_large, xm)) / (2 * delta)
    dSdy = (S(amps_large, yp) - S(amps_large, ym)) / (2 * delta)
    dSdz = (S(amps_large, zp) - S(amps_large, zm)) / (2 * delta)

    np.testing.assert_allclose(dS(amps_large, pos), [dSdx, dSdy, dSdz])


def test_SphericalHarmonicsForces():
    orders = 9
    radius = 12 * large_array.k
    F = levitate.fields.SphericalHarmonicsForce(large_array, orders=orders, radius=radius)
    dF = levitate.fields.SphericalHarmonicsForceGradient(large_array, orders=orders, radius=radius)
    F_sep = levitate.fields.SphericalHarmonicsForceDecomposition(large_array, orders=orders, radius=radius)
    dF_sep = levitate.fields.SphericalHarmonicsForceGradientDecomposition(large_array, orders=orders, radius=radius)

    delta = 1e-7

    dFdx = (F(amps_large, pos + [delta, 0, 0]) - F(amps_large, pos - [delta, 0, 0])) / (2 * delta)
    dFdy = (F(amps_large, pos + [0, delta, 0]) - F(amps_large, pos - [0, delta, 0])) / (2 * delta)
    dFdz = (F(amps_large, pos + [0, 0, delta]) - F(amps_large, pos - [0, 0, delta])) / (2 * delta)

    dFdx_sep = (F_sep(amps_large, pos + [delta, 0, 0]) - F_sep(amps_large, pos - [delta, 0, 0])) / (2 * delta)
    dFdy_sep = (F_sep(amps_large, pos + [0, delta, 0]) - F_sep(amps_large, pos - [0, delta, 0])) / (2 * delta)
    dFdz_sep = (F_sep(amps_large, pos + [0, 0, delta]) - F_sep(amps_large, pos - [0, 0, delta])) / (2 * delta)

    np.testing.assert_allclose(dF(amps_large, pos), np.stack([dFdx, dFdy, dFdz], axis=1), rtol=1e-6)
    np.testing.assert_allclose(dF_sep(amps_large, pos), np.stack([dFdx_sep, dFdy_sep, dFdz_sep], axis=1), rtol=1e-6)
    np.testing.assert_allclose(dF(amps_large, pos), np.sum(dF_sep(amps_large, pos), axis=2), rtol=1e-6)
    np.testing.assert_allclose(F(amps_large, pos), np.sum(F_sep(amps_large, pos), axis=1), rtol=1e-6)


array = levitate.arrays.RectangularArray(shape=(2, 1))
pos_1 = np.array([0.1, 0.2, 0.3])
pos_2 = np.array([-0.15, 1.27, 0.001])
both_pos = np.stack((pos_1, pos_2), axis=1)
phases = array.focus_phases((pos_1 + pos_2) / 2)
amps = levitate.utils.complex(phases)

spat_ders = array.pressure_derivs(both_pos, orders=3)
ind_ders = np.einsum('i, ji...->ji...', amps, spat_ders)
sum_ders = np.sum(ind_ders, axis=1)
sph_harm = array.spherical_harmonics(both_pos, orders=15)
ind_harms = np.einsum('i, ji...->ji...', amps, sph_harm)
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
    (levitate.fields.SphericalHarmonicsForce, {'radius': 1e-3, 'orders': 12},
        [+9.950696205718e-11, +2.697812005596e-10, +4.046718008394e-10],
        None,
     ),
    (levitate.fields.SphericalHarmonicsForceDecomposition, {'radius': 1e-3, 'orders': 2},
        [[+1.766918922489e-10, -1.796622157995e-11, -2.191928296745e-11, -2.599333684854e-11, -1.001026447213e-12, -7.856639712671e-13, -2.159467674377e-12, -5.722295540776e-12, -1.280839850720e-12], [+2.144581380772e-10, -1.437955598814e-11, +4.592891425167e-11, +2.274814569305e-11, +2.623003571298e-13, -1.045489575308e-12, -3.966663817153e-13, +1.700953892274e-12, +4.791025771738e-13], [+3.216872071158e-10, +2.052902419245e-11, +4.313866796441e-11, +1.777856377802e-11, -1.925567596154e-14, +2.272961554343e-13, +1.044756168385e-12, +2.505146189833e-13, -3.009962508997e-15]],
        None
     ),
    (levitate.fields.SphericalHarmonicsExpansion, {'radius': 1e-3, 'orders': 15}, sum_harms[..., 0], None),
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
