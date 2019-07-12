import levitate.transducers
import numpy as np
import pytest

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

source_pos = np.array([0.01, 0.12, -0.025])
source_normal = np.array([2., 3., 4.])
source_normal /= np.sum(source_normal**2)**0.5
receiver_pos = np.stack(([0.1, 0.2, 0.3], [-0.15, 1.27, 0.001]), axis=1)


@pytest.mark.parametrize("t_model, args, atol, rtol", [
    (levitate.transducers.PointSource, {}, 0, 1e-7),
    (levitate.transducers.PlaneWaveTransducer, {}, 0, 1e-7),
    (levitate.transducers.CircularPiston, {'effective_radius': 3e-3}, 0, 1e-4),
    (levitate.transducers.CircularRing, {'effective_radius': 3e-3}, 0, 1e-7),
])
def test_pressure_derivs(t_model, args, atol, rtol):
    idx = levitate.utils.pressure_derivs_order.index
    T = t_model(**args)
    spos = np.array([0, 0, 0])
    n = np.array([0, 0, 1])
    rpos = np.array([10, -20, 60]) * 1e-3
    delta = 1e-6
    x_plus = rpos + np.array([delta, 0, 0])
    x_minus = rpos - np.array([delta, 0, 0])
    y_plus = rpos + np.array([0, delta, 0])
    y_minus = rpos - np.array([0, delta, 0])
    z_plus = rpos + np.array([0, 0, delta])
    z_minus = rpos - np.array([0, 0, delta])

    implemented_results = T.pressure_derivs(spos, n, rpos)
    np.testing.assert_allclose(implemented_results[idx('')], T.greens_function(spos, n, rpos), rtol=rtol, atol=atol)

    dx = (T.greens_function(spos, n, x_plus) - T.greens_function(spos, n, x_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('x')], dx, rtol=rtol, atol=atol)
    dy = (T.greens_function(spos, n, y_plus) - T.greens_function(spos, n, y_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('y')], dy, rtol=rtol, atol=atol)
    dz = (T.greens_function(spos, n, z_plus) - T.greens_function(spos, n, z_minus)) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('z')], dz, rtol=rtol, atol=atol)

    dx2 = (T.pressure_derivs(spos, n, x_plus, orders=1)[idx('x')] - T.pressure_derivs(spos, n, x_minus, orders=1)[idx('x')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xx')], dx2, rtol=rtol, atol=atol)
    dy2 = (T.pressure_derivs(spos, n, y_plus, orders=1)[idx('y')] - T.pressure_derivs(spos, n, y_minus, orders=1)[idx('y')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('yy')], dy2, rtol=rtol, atol=atol)
    dz2 = (T.pressure_derivs(spos, n, z_plus, orders=1)[idx('z')] - T.pressure_derivs(spos, n, z_minus, orders=1)[idx('z')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('zz')], dz2, rtol=rtol, atol=atol)

    dxdy = (T.pressure_derivs(spos, n, y_plus, orders=1)[idx('x')] - T.pressure_derivs(spos, n, y_minus, orders=1)[idx('x')]) / (2 * delta)
    dydx = (T.pressure_derivs(spos, n, x_plus, orders=1)[idx('y')] - T.pressure_derivs(spos, n, x_minus, orders=1)[idx('y')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xy')], dxdy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xy')], dydx, rtol=rtol, atol=atol)
    dxdz = (T.pressure_derivs(spos, n, z_plus, orders=1)[idx('x')] - T.pressure_derivs(spos, n, z_minus, orders=1)[idx('x')]) / (2 * delta)
    dzdx = (T.pressure_derivs(spos, n, x_plus, orders=1)[idx('z')] - T.pressure_derivs(spos, n, x_minus, orders=1)[idx('z')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xz')], dxdz, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xz')], dzdx, rtol=rtol, atol=atol)
    dydz = (T.pressure_derivs(spos, n, z_plus, orders=1)[idx('y')] - T.pressure_derivs(spos, n, z_minus, orders=1)[idx('y')]) / (2 * delta)
    dzdy = (T.pressure_derivs(spos, n, y_plus, orders=1)[idx('z')] - T.pressure_derivs(spos, n, y_minus, orders=1)[idx('z')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('yz')], dydz, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('yz')], dzdy, rtol=rtol, atol=atol)

    dx3 = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('xx')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('xx')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xxx')], dx3, rtol=rtol, atol=atol)
    dy3 = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('yy')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('yy')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('yyy')], dy3, rtol=rtol, atol=atol)
    dz3 = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('zz')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('zz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('zzz')], dz3, rtol=rtol, atol=atol)

    dx2dy = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('xx')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('xx')]) / (2 * delta)
    dxdydx = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('xy')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('xy')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xxy')], dx2dy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xxy')], dxdydx, rtol=rtol, atol=atol)
    dx2dz = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('xx')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('xx')]) / (2 * delta)
    dxdzdx = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('xz')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('xz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xxz')], dx2dz, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xxz')], dxdzdx, rtol=rtol, atol=atol)
    dy2dx = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('yy')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('yy')]) / (2 * delta)
    dxdydy = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('xy')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('xy')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('yyx')], dy2dx, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('yyx')], dxdydy, rtol=rtol, atol=atol)
    dy2dz = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('yy')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('yy')]) / (2 * delta)
    dydzdy = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('yz')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('yz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('yyz')], dy2dz, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('yyz')], dydzdy, rtol=rtol, atol=atol)
    dz2dx = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('zz')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('zz')]) / (2 * delta)
    dxdzdz = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('xz')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('xz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('zzx')], dz2dx, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('zzx')], dxdzdz, rtol=rtol, atol=atol)
    dz2dy = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('zz')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('zz')]) / (2 * delta)
    dydzdz = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('yz')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('yz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('zzy')], dz2dy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('zzy')], dydzdz, rtol=rtol, atol=atol)
    dxdydz = (T.pressure_derivs(spos, n, z_plus, orders=2)[idx('xy')] - T.pressure_derivs(spos, n, z_minus, orders=2)[idx('xy')]) / (2 * delta)
    dxdzdy = (T.pressure_derivs(spos, n, y_plus, orders=2)[idx('xz')] - T.pressure_derivs(spos, n, y_minus, orders=2)[idx('xz')]) / (2 * delta)
    dydzdx = (T.pressure_derivs(spos, n, x_plus, orders=2)[idx('yz')] - T.pressure_derivs(spos, n, x_minus, orders=2)[idx('yz')]) / (2 * delta)
    np.testing.assert_allclose(implemented_results[idx('xyz')], dxdydz, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xyz')], dxdzdy, rtol=rtol, atol=atol)
    np.testing.assert_allclose(implemented_results[idx('xyz')], dydzdx, rtol=rtol, atol=atol)


def test_PointSource():
    transducer = levitate.transducers.PointSource()
    expected_result = np.array([-15.10269228 + 8.46147216j, -4.76079297 + 2.00641887j])
    np.testing.assert_allclose(transducer.greens_function(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [-1.51026923e+01+8.46147216e+00j, -4.76079297e+00+2.00641887e+00j],
        [-1.59865364e+03-2.87993690e+03j,  2.01978326e+02+4.80828427e+02j],
        [-1.42102546e+03-2.55994391e+03j, -1.45171922e+03-3.45595432e+03j],
        [-5.77291591e+03-1.03997721e+04j, -3.28214780e+01-7.81346194e+01j],
        [ 5.32591336e+05-3.31855829e+05j,  4.73239427e+04-2.32802902e+04j],
        [ 4.17084909e+05-2.68922977e+05j,  2.50871386e+06-1.05042066e+06j],
        [ 7.15892330e+06-3.94216546e+06j,  2.06176347e+01-3.54056736e+03j],
        [ 4.89203693e+05-2.66539137e+05j, -3.49214083e+05+1.45727371e+05j],
        [ 1.98739000e+06-1.08281525e+06j, -7.89527492e+03+3.29470579e+03j],
        [ 1.76656889e+06-9.62502441e+05j,  5.67472885e+04-2.36806979e+04j],
        [ 7.41593207e+07+9.53900787e+07j, -2.94041700e+06-4.53170575e+06j],
        [ 5.55068738e+07+6.51308240e+07j,  7.60087762e+08+1.82107897e+09j],
        [ 2.69449904e+09+4.92642965e+09j,  1.56744694e+05-4.06988479e+04j],
        [ 5.50482030e+07+9.07142730e+07j,  1.67690712e+07+3.43932273e+07j],
        [ 2.23633325e+08+3.68526734e+08j,  3.79126826e+05+7.77586007e+05j],
        [ 5.02151407e+07+7.99356555e+07j, -1.05144012e+08-2.53620948e+08j],
        [ 1.81332453e+08+2.88656534e+08j,  1.70859019e+07+4.12134041e+07j],
        [ 7.33938873e+08+1.37090554e+09j, -3.57253891e+05-2.98445839e+03j],
        [ 6.52390109e+08+1.21858270e+09j,  2.56776234e+06+2.14507947e+04j],
        [ 1.79156634e+08+3.38273791e+08j, -2.37030351e+06-5.73690379e+06j]])
    np.testing.assert_allclose(transducer.pressure_derivs(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [-5.353765017637e+01 + 2.999513783003e+01j, -1.687657166657e+01 + 7.112569704547e+00j],
        [+5.695860852980e+00 - 2.548416823971e+01j, +2.167407273009e+01 - 5.752746078262e+00j],
        [-4.837433985040e+01 - 8.714523473163e+01j, -2.750286609559e-01 - 6.547316278457e-01j],
        [+2.464062477253e+01 + 8.644378088995e+00j, +1.928053880784e+01 - 1.145077898789e+01j],
        [+5.863822153231e+00 + 8.276759896288e+00j, -2.488833213001e+01 + 2.971870739412e+00j],
        [+5.333966279800e+01 + 1.236356565275e+01j, +2.854304528881e-01 + 1.085673389825e+00j],
        [+9.868256531775e+01 - 5.376649075788e+01j, -1.886812414453e+01 + 7.873686280016e+00j],
        [-1.853191532112e+01 + 5.152228156424e+01j, +5.709536101225e-01 + 9.665243027971e-01j],
        [-3.775246378829e+00 - 9.414705486879e+00j, -1.961827635764e+01 + 1.560077961708e+01j],
        [-3.778260442830e+00 - 4.635117406916e-01j, +2.705960198748e+01 + 6.203930665125e-01j],
        [-2.070501703273e+01 + 1.430452227680e+01j, -1.708304991530e-01 - 1.474786761261e+00j],
        [-1.968756018004e+01 + 8.059689952413e+01j, +2.025469122568e+01 - 5.248816716441e+00j],
        [+4.975648181395e+01 + 9.394747674264e+01j, -6.208276193295e-01 - 1.502604332121e+00j],
        [-7.773286212700e+01 - 2.900107557128e+01j, +1.805262554628e+01 - 1.057853035428e+01j],
        [+2.346684539051e+01 - 9.089788518495e+00j, -9.199565969320e-01 - 1.165272032786e+00j],
        [-1.739556019461e+00 + 3.385858822038e+00j, +1.873027112797e+01 - 1.953929094799e+01j],
        [+1.067167256341e+00 - 9.104799473031e-01j, -2.830756018738e+01 - 4.738810592805e+00j],
        [+1.471204148735e+00 - 1.060716489148e+01j, -5.020922105269e-02 + 1.817170541889e+00j],
        [-2.515148410436e+01 - 3.766028611012e+01j, -2.149878266779e+01 + 2.387934380959e+00j],
        [-1.023636845838e+02 - 2.671986827657e+01j, +5.073534692683e-01 + 1.996427449022e+00j],
        [-7.549405607371e+01 + 3.847298817205e+01j, -1.898342421571e+01 + 7.739075518117e+00j],
        [+3.853685289483e+01 - 9.852505392664e+01j, +1.033062840690e+00 + 1.782108658071e+00j],
        [+1.569025610876e+01 + 4.248188044083e+01j, -1.704057465374e+01 + 1.332361445986e+01j],
        [-9.446854895005e+00 - 5.043145970021e+00j, +1.306377128024e+00 + 1.264123626693e+00j],
        [+1.363811663471e+00 - 3.283861051063e-01j, -1.692598551605e+01 + 2.317941552126e+01j]])
    np.testing.assert_allclose(transducer.spherical_harmonics(source_pos, source_normal, receiver_pos, orders=4), expected_result)


def test_ReflectingTransducer():
    transducer = levitate.transducers.TransducerReflector(levitate.transducers.PointSource, plane_distance=0.5, plane_normal=(3, -1, 9), reflection_coefficient=np.exp(1j))
    expected_result = np.array([-22.81413464 + 10.64739914j, -3.69589556 + 5.43218304j])
    np.testing.assert_allclose(transducer.greens_function(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [-2.28141346e+01+1.06473991e+01j, -3.69589556e+00+5.43218304e+00j],
        [-1.07726257e+03-1.02772709e+03j,  9.45472742e+02+2.50378634e+02j],
        [-1.82867333e+03-4.00808802e+03j, -3.34590674e+03-2.86884147e+03j],
        [-4.32566814e+03-5.25851314e+03j,  1.43803609e+03-5.34034241e+02j],
        [ 9.75572453e+05-4.62954398e+05j, -3.91587711e+03-1.84216321e+05j],
        [ 6.87045908e+05-3.51997121e+05j,  2.18437515e+06-2.09756062e+06j],
        [ 1.05862362e+07-4.90161175e+06j, -1.96141743e+05-6.34749208e+05j],
        [ 1.41198233e+05-1.69944810e+05j, -2.22495978e+05+5.56926718e+05j],
        [ 3.22289288e+06-1.42574828e+06j, -1.06293286e+05-3.16006136e+05j],
        [ 8.00595061e+05-6.94381380e+05j,  3.07434154e+05+7.89796387e+05j],
        [ 3.96401718e+07-1.01147813e+07j, -3.76855650e+07+7.15226960e+06j],
        [ 7.36593410e+07+1.15095530e+08j,  1.33885518e+09+1.64154678e+09j],
        [ 2.05758654e+09+2.64193412e+09j, -2.70592708e+08+8.47864254e+07j],
        [ 7.91989744e+07+1.73990605e+08j,  1.05800397e+08+6.28672780e+06j],
        [ 1.37892292e+08+7.28757772e+07j, -6.87546825e+07+2.26025938e+07j],
        [ 3.06274689e+07+1.50221459e+07j, -3.32518971e+08-1.83803909e+08j],
        [ 1.26962103e+08+1.08473293e+08j, -4.32730764e+08+1.79332694e+08j],
        [ 5.08112071e+08+5.46876866e+08j, -1.37417530e+08+4.22239466e+07j],
        [ 8.28952046e+08+1.86284679e+09j,  3.51753794e+08-1.07559352e+08j],
        [2.42254321e+08+5.70520748e+08j, 1.74266853e+08-5.96975457e+07j]])
    np.testing.assert_allclose(transducer.pressure_derivs(source_pos, source_normal, receiver_pos), expected_result)

def test_PlaneWaveTransducer():
    transducer = levitate.transducers.PlaneWaveTransducer()
    expected_result = np.array([0.10009402 + 5.99916504j, 5.8662305 + 1.2598967j])
    np.testing.assert_allclose(transducer.greens_function(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [ 1.00094019e-01+5.99916504e+00j,  5.86623050e+00+1.25989670e+00j],
        [-1.63255397e+03+2.72386050e+01j, -3.42855939e+02+1.59637846e+03j],
        [-2.44883095e+03+4.08579075e+01j, -5.14283908e+02+2.39456769e+03j],
        [-3.26510793e+03+5.44772100e+01j, -6.85711877e+02+3.19275693e+03j],
        [-1.48248939e+04-8.88534465e+05j, -8.68845572e+05-1.86602908e+05j],
        [-1.66780056e+04-9.99601273e+05j, -9.77451268e+05-2.09928272e+05j],
        [-2.96497878e+04-1.77706893e+06j, -1.73769114e+06-3.73205817e+05j],
        [-1.11186704e+04-6.66400849e+05j, -6.51634179e+05-1.39952181e+05j],
        [-1.48248939e+04-8.88534465e+05j, -8.68845572e+05-1.86602908e+05j],
        [-2.22373409e+04-1.33280170e+06j, -1.30326836e+06-2.79904363e+05j],
        [ 2.41797059e+08-4.03430130e+06j,  5.07802863e+07-2.36439117e+08j],
        [ 4.08032537e+08-6.80788345e+06j,  8.56917331e+07-3.98991010e+08j],
        [ 9.67188237e+08-1.61372052e+07j,  2.03121145e+08-9.45756467e+08j],
        [ 3.62695589e+08-6.05145195e+06j,  7.61704294e+07-3.54658675e+08j],
        [ 4.83594118e+08-8.06860260e+06j,  1.01560573e+08-4.72878234e+08j],
        [ 2.72021692e+08-4.53858896e+06j,  5.71278220e+07-2.65994006e+08j],
        [ 5.44043383e+08-9.07717793e+06j,  1.14255644e+08-5.31988013e+08j],
        [ 4.83594118e+08-8.06860260e+06j,  1.01560573e+08-4.72878234e+08j],
        [ 7.25391178e+08-1.21029039e+07j,  1.52340859e+08-7.09317350e+08j],
        [3.62695589e+08-6.05145195e+06j, 7.61704294e+07-3.54658675e+08j]])
    np.testing.assert_allclose(transducer.pressure_derivs(source_pos, source_normal, receiver_pos), expected_result)


def test_CircularPiston():
    transducer = levitate.transducers.CircularPiston(effective_radius=3e-3)
    expected_result = np.array([-13.76846639 + 7.71395542j, -2.94292484 + 1.24028496j])
    np.testing.assert_allclose(transducer.greens_function(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [-1.37684664e+01+7.71395542e+00j, -2.94292484e+00+1.24028496e+00j],
        [-1.46345035e+03-2.62213599e+03j,  1.24020677e+02+2.97579665e+02j],
        [-1.31120087e+03-2.32498540e+03j, -8.97477518e+02-2.13629191e+03j],
        [-5.25737863e+03-9.48412135e+03j, -2.16665257e+01-4.77189774e+01j],
        [ 4.84382213e+05-3.04903352e+05j,  2.93243277e+04-1.42224002e+04j],
        [ 3.77388146e+05-2.50552438e+05j,  1.55073129e+06-6.49449802e+05j],
        [ 6.53069266e+06-3.58626397e+06j, -7.42767086e+00-2.23335601e+03j],
        [ 4.43752707e+05-2.47008880e+05j, -2.16119628e+05+8.94854997e+04j],
        [ 1.81008550e+06-9.90242571e+05j, -4.82923089e+03+2.16269160e+03j],
        [ 1.60504101e+06-8.87366311e+05j,  3.46591766e+04-1.56403155e+04j],
        [ 6.82828158e+07+8.66331965e+07j, -1.79275397e+06-2.81350794e+06j],
        [ 5.19358860e+07+5.85916458e+07j,  4.69989068e+08+1.12565967e+09j],
        [ 2.44856151e+09+4.49550825e+09j,  9.68920897e+04-2.82922526e+04j],
        [ 5.11404315e+07+8.21611475e+07j,  1.02443418e+07+2.13108604e+07j],
        [ 2.05309319e+08+3.35303612e+08j,  2.45407452e+05+4.75368579e+05j],
        [ 4.69742337e+07+7.22314736e+07j, -6.45682263e+07-1.56955705e+08j],
        [ 1.68880510e+08+2.61333135e+08j,  1.12904036e+07+2.51731080e+07j],
        [ 6.70487884e+08+1.24899315e+09j, -2.25374325e+05-4.85350052e+02j],
        [ 6.00924515e+08+1.10756082e+09j,  1.61978412e+06-1.32990931e+03j],
        [ 1.65922945e+08+3.06976755e+08j, -1.55693915e+06-3.50934304e+06j]])
    np.testing.assert_allclose(transducer.pressure_derivs(source_pos, source_normal, receiver_pos), expected_result)


def test_CircularRing():
    transducer = levitate.transducers.CircularRing(effective_radius=3e-3)
    expected_result = np.array([-12.47473964 + 6.98912884j, -1.39288579 + 0.58702665j])
    np.testing.assert_allclose(transducer.greens_function(source_pos, source_normal, receiver_pos), expected_result)
    expected_result = np.array([
        [-1.24747396e+01+6.98912884e+00j, -1.39288579e+00+5.87026646e-01j],
        [-1.33216229e+03-2.37226612e+03j,  5.76923908e+01+1.41268617e+02j],
        [-1.20421839e+03-2.09743421e+03j, -4.24878396e+02-1.01106339e+03j],
        [-4.75766332e+03-8.59616715e+03j, -1.19175750e+01-2.18845970e+01j],
        [ 4.37670943e+05-2.78693954e+05j,  1.39642907e+04-6.52800321e+03j],
        [ 3.38973535e+05-2.32564318e+05j,  7.33898537e+05-3.07533292e+05j],
        [ 5.92139795e+06-3.24140373e+06j, -2.82896142e+01-1.11084914e+03j],
        [ 3.99746624e+05-2.27943198e+05j, -1.02591095e+05+4.16327578e+04j],
        [ 1.63821875e+06-9.00384072e+05j, -2.22401225e+03+1.17584088e+03j],
        [ 1.44859124e+06-8.14203747e+05j,  1.58976478e+04-8.61182144e+03j],
        [ 6.25630965e+07+7.81513868e+07j, -8.18448334e+05-1.34630517e+06j],
        [ 4.84284909e+07+5.22692998e+07j,  2.22607867e+08+5.32707102e+08j],
        [ 2.21033466e+09+4.07752947e+09j,  4.58481144e+04-1.71939165e+04j],
        [ 4.73203334e+07+7.38824174e+07j,  4.70182541e+06+1.01471404e+07j],
        [ 1.87496291e+08+3.03109238e+08j,  1.29505217e+05+2.18641076e+05j],
        [ 4.37922126e+07+6.47773905e+07j, -3.00442186e+07-7.45016718e+07j],
        [ 1.56691171e+08+2.34889313e+08j,  6.22315409e+06+1.15482964e+07j],
        [ 6.08920494e+08+1.13080836e+09j, -1.12129531e+05+1.44617054e+03j],
        [ 5.50831237e+08+1.00001981e+09j,  8.05741086e+05-1.85496241e+04j],
        [ 1.53008644e+08+2.76671835e+08j, -8.47676537e+05-1.61653014e+06j]])
    np.testing.assert_allclose(transducer.pressure_derivs(source_pos, source_normal, receiver_pos), expected_result)
