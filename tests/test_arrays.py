import levitate.arrays
import levitate.hardware
import numpy as np
import pytest

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

# Tests were mostly written before the layout was transposed,
# so a lot of the "expected" positions etc are transposed in the test hardcoding.


def test_rectangular_grid():
    # positions, normals = levitate.arrays.rectangular_grid(shape=(5, 3), spread=10e-3)
    array = levitate.arrays.RectangularArray(shape=(5, 3), transducer_size=10e-3)
    expected_positions = np.array([
        [-0.02, -0.01, 0.],
        [-0.01, -0.01, 0.],
        [0.   , -0.01, 0.],
        [ 0.01, -0.01, 0.],
        [ 0.02, -0.01, 0.],
        [-0.02,  0.  , 0.],
        [-0.01,  0.  , 0.],
        [ 0.  ,  0.  , 0.],
        [ 0.01,  0.  , 0.],
        [ 0.02,  0.  , 0.],
        [-0.02,  0.01, 0.],
        [-0.01,  0.01, 0.],
        [ 0.  ,  0.01, 0.],
        [ 0.01,  0.01, 0.],
        [ 0.02,  0.01, 0.]]).T
    expected_normals = np.array([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]]).T
    np.testing.assert_allclose(array.positions, expected_positions)
    np.testing.assert_allclose(array.normals, expected_normals)


@pytest.mark.parametrize('rectangular_class, shape, spread', [
    (levitate.hardware.DragonflyArray, (16, 16), 10.47e-3),
    (levitate.hardware.AcoustophoreticBoard, (16, 16), 10.47e-3),
])
def test_rectangular_variants(rectangular_class, shape, spread):
    # Check that we do place all the indices.
    all_indices = np.sort(rectangular_class.grid_indices.flatten())
    np.testing.assert_equal(all_indices, list(range(np.prod(shape))))
    array = rectangular_class()
    # Test that every element is where it should be
    for idx, (x, y, _) in enumerate(zip(*array.positions)):
        y_idx, x_idx = np.argwhere(array.grid_indices == idx)[0]
        np.testing.assert_allclose(x / array.spread + 7.5, x_idx)
        np.testing.assert_allclose(7.5 - y / array.spread, y_idx)
    # Test that this is indeed a 16x16 rectangular array with spread 10.47 mm
    rectangular = levitate.arrays.RectangularArray(shape=shape, spread=spread)
    differences = rectangular.positions[:, :, None] - array.positions[:, None, :]
    distances = np.sum(differences**2, axis=0)
    assert all([(np.min(distances, axis=axis) == 0).all() for axis in range(distances.ndim)])


def test_array_offset():
    array = levitate.arrays.RectangularArray(shape=(4, 2), offset=(0.1, -0.2, 1.4))
    expected_positions = np.array([
        [0.085, -0.205, 1.4],
        [0.095, -0.205, 1.4],
        [0.105, -0.205, 1.4],
        [0.115, -0.205, 1.4],
        [0.085, -0.195, 1.4],
        [0.095, -0.195, 1.4],
        [0.105, -0.195, 1.4],
        [0.115, -0.195, 1.4]]).T
    expected_normals = np.array([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]]).T
    np.testing.assert_allclose(array.positions, expected_positions)
    np.testing.assert_allclose(array.normals, expected_normals)


@pytest.mark.parametrize('offset', [(0., 0., 0.), (1.85194e-3, 28.2839e-3, 35.2830e-3)])
@pytest.mark.parametrize('radius', [0.4, 0.05])
@pytest.mark.parametrize('rings', [4, 15])
@pytest.mark.parametrize('normal', [(0., 0., 1.), (-2., 5., 4.37)])
@pytest.mark.parametrize('packing', ['distance', 'count'])
@pytest.mark.parametrize('spread', [10e-3, 5e-3])
def test_spherical_cap(radius, rings, normal, offset, packing, spread):
    normal, offset = np.asarray(normal), np.asarray(offset)
    array = levitate.arrays.SphericalCapArray(radius=radius, rings=rings, spread=spread, packing=packing, normal=normal, offset=offset)
    focus = normal / np.sum(normal**2)**0.5 * radius + offset
    diff = focus[:, None] - array.positions
    dist = np.sum(diff**2, 0)**0.5
    np.testing.assert_allclose(dist, radius)
    np.testing.assert_allclose(np.sum(diff * array.normals, 0) / dist, 1.)

    array = levitate.arrays.DoublesidedArray(
        levitate.arrays.SphericalCapArray(
            radius=radius, rings=rings, spread=spread, packing=packing, normal=normal
        ),
        separation=radius * 2, normal=normal, offset=offset
    )
    diff = offset[:, None] - array.positions
    dist = np.sum(diff**2, 0)**0.5
    np.testing.assert_allclose(dist, radius)
    np.testing.assert_allclose(np.sum(diff * array.normals, 0) / dist, 1.)


def test_array_normal():
    array = levitate.arrays.RectangularArray(shape=(4, 2), normal=(2, 3, 4))
    expected_positions = np.array([
        [-1.321925551875e-02, -2.328883278124e-03, +8.356290217967e-03],
        [-4.010697510416e-03, -3.516046265624e-03, +4.642383454426e-03],
        [+5.197860497917e-03, -4.703209253125e-03, +9.284766908853e-04],
        [+1.440641850625e-02, -5.890372240625e-03, -2.785430072656e-03],
        [-1.440641850625e-02, +5.890372240625e-03, +2.785430072656e-03],
        [-5.197860497917e-03, +4.703209253125e-03, -9.284766908853e-04],
        [+4.010697510416e-03, +3.516046265624e-03, -4.642383454426e-03],
        [+1.321925551875e-02, +2.328883278124e-03, -8.356290217967e-03]]).T
    expected_normals = np.array([
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01],
        [+3.713906763541e-01, +5.570860145312e-01, +7.427813527082e-01]]).T
    np.testing.assert_allclose(array.positions, expected_positions)
    np.testing.assert_allclose(array.normals, expected_normals)


def test_array_rotation():
    array = levitate.arrays.RectangularArray(shape=(4, 2), rotation=1, normal=(-1, 4, -2))
    expected_positions = np.array([
        [-8.747019947780e-03, +4.075784833732e-03, +1.252507964135e-02],
        [-9.564871486238e-04, +2.940455719910e-03, +6.359155014132e-03],
        [+6.834045650533e-03, +1.805126606089e-03, +1.932303869110e-04],
        [+1.462457844969e-02, +6.697974922671e-04, -5.972694240310e-03],
        [-1.462457844969e-02, -6.697974922671e-04, +5.972694240310e-03],
        [-6.834045650533e-03, -1.805126606089e-03, -1.932303869110e-04],
        [+9.564871486238e-04, -2.940455719910e-03, -6.359155014132e-03],
        [+8.747019947780e-03, -4.075784833732e-03, -1.252507964135e-02]]).T
    expected_normals = np.array([
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01],
        [-2.182178902360e-01, +8.728715609440e-01, -4.364357804720e-01]]).T
    np.testing.assert_allclose(array.positions, expected_positions)
    np.testing.assert_allclose(array.normals, expected_normals)


def test_double_sided_grid():
    array = levitate.arrays.DoublesidedArray(levitate.arrays.RectangularArray, separation=0.5, shape=(2, 2), spread=5e-3, offset=(0.1, -0.2, 1.4), normal=(0.2, -1.4, 2), rotation=1)

    expected_positions = np.array([
        [+8.02489979e-02, -5.99269663e-02, +1.19384001e+00],
        [+8.30486836e-02, -5.64068310e-02, +1.19602413e+00],
        [+7.61264873e-02, -5.78193656e-02, +1.19572758e+00],
        [+7.89261731e-02, -5.42992303e-02, +1.19791170e+00],
        [+1.21073827e-01, -3.45700770e-01, +1.60208830e+00],
        [+1.23873513e-01, -3.42180634e-01, +1.60427242e+00],
        [+1.16951316e-01, -3.43593169e-01, +1.60397587e+00],
        [+1.19751002e-01, -3.40073034e-01, +1.60615999e+00]]).T
    expected_normals = np.array([
        [+8.16496581e-02, -5.71547607e-01, +8.16496581e-01],
        [+8.16496581e-02, -5.71547607e-01, +8.16496581e-01],
        [+8.16496581e-02, -5.71547607e-01, +8.16496581e-01],
        [+8.16496581e-02, -5.71547607e-01, +8.16496581e-01],
        [-8.16496581e-02, +5.71547607e-01, -8.16496581e-01],
        [-8.16496581e-02, +5.71547607e-01, -8.16496581e-01],
        [-8.16496581e-02, +5.71547607e-01, -8.16496581e-01],
        [-8.16496581e-02, +5.71547607e-01, -8.16496581e-01]]).T
    np.testing.assert_allclose(array.positions, expected_positions)
    np.testing.assert_allclose(array.normals, expected_normals)
    expected_signature = np.array([0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi])
    np.testing.assert_allclose(array.signature(stype='doublesided'), expected_signature)


def test_Array_basics():
    pos, norm = np.random.normal(size=(2, 3, 8))
    array = levitate.arrays.TransducerArray(pos, norm)
    array.omega = 200000
    np.testing.assert_allclose(2 * np.pi * array.freq, array.omega)
    array.k = 730
    np.testing.assert_allclose(2 * np.pi / array.k, array.wavelength)
    array.wavelength = 8.5e-3
    np.testing.assert_allclose(air.c, array.wavelength * array.freq)
    array.freq = 41e3
    np.testing.assert_allclose(array.transducer.freq, 41e3)

    from levitate.transducers import PlaneWaveTransducer
    array = levitate.arrays.RectangularArray(shape=(4, 4), transducer=PlaneWaveTransducer)
    pos = np.array([7e-3, -3e-3, 70e-3])
    np.testing.assert_allclose(array.focus_phases(pos), np.array([+2.069588782645e+00, -2.511641902621e+00, -1.794667262208e+00, -2.103144086214e+00, +2.763870843565e+00, -1.794667262208e+00, -1.067679552580e+00, -1.380498791671e+00, +2.465230031099e+00, -2.103144086214e+00, -1.380498791671e+00, -1.691434660688e+00, +1.189733359830e+00, +2.863786956082e+00, -2.714709695304e+00, -3.017860322253e+00]))
    np.testing.assert_allclose(array.signature(pos, stype='twin'), np.array([-1.570796326795e+00, -1.570796326795e+00, +1.570796326795e+00, +1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, +1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, +1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, -1.570796326795e+00, +1.570796326795e+00]))
    np.testing.assert_allclose(array.signature(pos, stype='vortex'), np.array([-2.642245931910e+00, -2.356194490192e+00, -1.735945004210e+00, -9.827937232473e-01, -3.050932766389e+00, -2.976443976175e+00, -2.356194490192e+00, -2.449786631269e-01, +2.792821650006e+00, +2.553590050042e+00, +1.815774989922e+00, +7.853981633974e-01, +2.455863142684e+00, +2.158798930342e+00, +1.681453547969e+00, +1.152571997216e+00]))
    np.testing.assert_allclose(array.signature(pos, stype='bottle'), np.array([+3.141592653590e+00, +3.141592653590e+00, +0.000000000000e+00, +0.000000000000e+00, +3.141592653590e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +3.141592653590e+00, +0.000000000000e+00, +0.000000000000e+00, +0.000000000000e+00, +3.141592653590e+00, +3.141592653590e+00, +3.141592653590e+00, +3.141592653590e+00]))
    pos = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(array.focus_phases(pos), np.array([-1.4782875, 0.70451472, 2.70433793, -1.76613547, 1.07535199, -3.05482743, -1.08272084, 0.70451472, -2.79668059, -0.67375749, 1.27038374, 3.03192953, -0.52217161, 1.57048339, -2.79668059, -1.0609514]))
    np.testing.assert_allclose(array.signature(pos, stype='twin'), np.array([-1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633]))
    np.testing.assert_allclose(array.signature(stype='vortex'), np.array([-2.35619449, -1.89254688, -1.24904577, -0.78539816, -2.8198421, -2.35619449, -0.78539816, -0.32175055, 2.8198421, 2.35619449, 0.78539816, 0.32175055, 2.35619449, 1.89254688, 1.24904577, 0.78539816]))
    np.testing.assert_allclose(array.signature(stype='bottle'), np.array([3.14159265, 0., 0., 3.14159265, 0., 0., 0., 0., 0., 0., 0., 0., 3.14159265, 0., 0., 3.14159265]))


def test_Array_visualizer():
    array = levitate.arrays.RectangularArray(shape=2)
    array.visualize()
    pos = np.array([0, 0, 0.05])
    signature = array.signature(angle=np.pi, stype='twin')
    phase = array.focus_phases(pos) + signature
    amps = levitate.complex(phase)
    from levitate.analysis import find_trap
    trap_pos = find_trap(array, amps, pos)
    np.testing.assert_allclose(pos, trap_pos, atol=0.1e-3)
    np.testing.assert_allclose(signature, array.signature(pos, phase))


def test_Array_calculations():
    array = levitate.arrays.RectangularArray(shape=2)
    pos = np.array([0.1, -0.2, 0.3])
    expected_result = np.array([
        [-1.60298320e+01+1.39433272e+00j,  7.60005836e+00+1.43149212e+01j, 1.23972894e+01+9.89785204e+00j,  4.72749101e+00-1.52603890e+01j],
        [-2.75579656e+02-3.30839037e+03j, -2.69691942e+03+1.41912500e+03j, -2.02250900e+03+2.51457673e+03j,  2.82526527e+03+8.86498882e+02j],
        [ 5.11790789e+02+6.14415354e+03j,  5.53578197e+03-2.91294080e+03j, 3.94870805e+03-4.90941170e+03j, -6.09662505e+03-1.91297127e+03j],
        [-7.87370445e+02-9.45254391e+03j, -8.51658764e+03+4.48144738e+03j, -5.77859714e+03+7.18450493e+03j,  8.92189032e+03+2.79947015e+03j],
        [ 6.80382495e+05-8.33698641e+04j, -2.91485806e+05-4.94127262e+05j, -5.27798435e+05-3.91154134e+05j, -1.38372830e+05+5.31781607e+05j],
        [ 2.35305285e+06-2.10377336e+05j, -1.13689503e+06-2.12990534e+06j, -1.95769683e+06-1.55833572e+06j, -7.53077687e+05+2.44212524e+06j],
        [ 5.57294329e+06-4.54866720e+05j, -2.65207242e+06-5.06161450e+06j, -4.17057990e+06-3.36464336e+06j, -1.64672816e+06+5.21935967e+06j],
        [-1.26844169e+06+9.63139996e+04j,  5.40041580e+05+1.04492376e+06j, 9.92856867e+05+8.10438183e+05j,  3.62769003e+05-1.12739219e+06j],
        [ 1.95144875e+06-1.48175384e+05j, -8.30833200e+05-1.60757501e+06j, -1.45296127e+06-1.18600710e+06j, -5.30881467e+05+1.64984223e+06j],
        [-3.62411911e+06+2.75182856e+05j,  1.70539446e+06+3.29975923e+06j, 2.83673391e+06+2.31553766e+06j,  1.14558632e+06-3.56018587e+06j],
        [ 2.86670148e+07+1.39550752e+08j,  8.79550534e+07-6.44855683e+07j, 7.10265677e+07-1.14387829e+08j, -1.01799225e+08-1.57149418e+07j],
        [-9.48653965e+07-9.00596323e+08j, -8.15465047e+08+4.51662413e+08j, -6.08347649e+08+7.86174017e+08j,  9.81016375e+08+2.88086328e+08j],
        [ 2.69225703e+08+3.28515432e+09j,  3.00523878e+09-1.57526192e+09j, 1.95396625e+09-2.42520390e+09j, -3.05538963e+09-9.62345944e+08j],
        [-2.90779477e+07-2.61000235e+08j, -1.91908617e+08+1.10366719e+08j, -1.57582477e+08+2.07891701e+08j,  2.12034769e+08+5.76457627e+07j],
        [ 4.47353042e+07+4.01538824e+08j,  2.95244026e+08-1.69794952e+08j, 2.30608503e+08-3.04231758e+08j, -3.10294784e+08-8.43596528e+07j],
        [ 3.80717090e+07+4.85924318e+08j,  4.02816731e+08-2.09323496e+08j, 3.21279107e+08-3.94767783e+08j, -4.51078135e+08-1.44502368e+08j],
        [ 1.08776311e+08+1.38835519e+09j,  1.27205284e+09-6.61021567e+08j, 9.17940305e+08-1.12790795e+09j, -1.42445727e+09-4.56323268e+08j],
        [ 8.12193376e+07+1.15079185e+09j,  9.57197835e+08-4.88115774e+08j, 6.93574595e+08-8.40914649e+08j, -9.64000841e+08-3.15741830e+08j],
        [-1.50835913e+08-2.13718486e+09j, -1.96477450e+09+1.00192185e+09j, -1.35412183e+09+1.64178574e+09j,  2.08021234e+09+6.81337634e+08j],
        [-4.85644305e+07-7.48335747e+08j, -6.23978731e+08+3.13792174e+08j, -4.77252163e+08+5.71923550e+08j,  6.57524678e+08+2.19514891e+08j]])
    np.testing.assert_allclose(array.pressure_derivs(pos), expected_result)

    expected_result = np.array([
        [-5.682427477075e+01 + 4.942780801222e+00j, +2.694150540810e+01 + 5.074507426259e+01j, +4.394724655282e+01 + 3.508697193457e+01j, +1.675851928357e+01 - 5.409667034377e+01j],
        [-3.803836147518e+01 - 1.657048080374e+01j, +1.279975203258e+00 + 4.120938631452e+01j, +1.710554711045e+01 + 3.829641433944e+01j, +2.807511601090e+01 - 3.087116808379e+01j],
        [-6.597796686300e+00 - 7.920790438223e+01j, -7.136502785396e+01 + 3.755243658313e+01j, -4.842194588566e+01 + 6.020279674447e+01j, +7.476127507154e+01 + 2.345825276365e+01j],
        [-3.477261873074e+01 + 2.263543161484e+01j, +3.323968212216e+01 + 2.439211736111e+01j, +4.107318751625e+01 + 8.497550262354e+00j, -5.405546885483e+00 - 4.137657816630e+01j],
        [-1.686407000850e+01 - 2.178756862198e+01j, -1.125090492261e+01 + 2.455396689049e+01j, -1.726622397846e+00 + 2.850517035519e+01j, +2.594291002246e+01 - 1.071176900518e+01j],
        [+3.031007376630e+01 - 6.821025513107e+01j, -7.468896294954e+01 + 1.768761675083e+00j, -6.814162421776e+01 + 2.984827855687e+01j, +5.477558195724e+01 + 5.054666208556e+01j],
        [+5.988534535282e+01 - 4.547151978544e+00j, -2.861052213724e+01 - 5.535835638677e+01j, -4.322596009999e+01 - 3.528400690252e+01j, -1.773137889725e+01 + 5.510453754651e+01j],
        [-4.026173202706e+01 - 6.285166991374e+01j, -4.464287578055e+01 + 5.990478567180e+01j, -1.559702090250e+01 + 7.273880410627e+01j, +7.397427410032e+01 - 9.117905355749e+00j],
        [-1.338098961723e+01 + 2.408410514370e+01j, +2.653751067525e+01 + 5.024010230645e+00j, +2.758190339425e+01 - 7.400308576910e+00j, -1.482787772910e+01 - 2.383087530295e+01j],
        [-2.707354746999e+00 - 1.746661744378e+01j, -1.332639233207e+01 + 1.070479817804e+01j, -9.733075343865e+00 + 1.606625396807e+01j, +1.821285983077e+01 + 9.712939338454e-01j],
        [+4.676829722253e+01 - 3.538544878575e+01j, -5.237549857426e+01 - 2.470351811875e+01j, -5.977893107220e+01 - 4.270891236968e+00j, +2.203974234614e+01 + 5.507262673066e+01j],
        [+7.892432023664e+01 + 3.611381986030e+01j, -1.111183761150e+00 - 8.806141599513e+01j, -3.294243372319e+01 - 7.747677728413e+01j, -5.866451648153e+01 + 6.219821126638e+01j],
        [+9.287877976849e-01 + 1.431181429618e+01j, +1.559873321909e+01 - 7.844434688683e+00j, +5.510582917790e+00 - 6.603704269743e+00j, -1.080601629861e+01 - 3.607593105422e+00j],
        [+7.359464973840e+01 - 4.601177935703e+01j, -7.002975115181e+01 - 5.340301180676e+01j, -8.212235075402e+01 - 1.854115596653e+01j, +9.533619685308e+00 + 8.496618783049e+01j],
        [-5.094954128769e+01 - 2.904389528692e+01j, -1.140045085313e+01 + 5.677575520142e+01j, +1.490296577659e+01 + 5.804879605259e+01j, +5.070404281442e+01 - 3.078708333006e+01j],
        [-4.271062938279e-01 + 1.767003324465e+01j, +1.653992786475e+01 - 4.314652112229e+00j, +1.406441459177e+01 - 1.245188798563e+01j, -1.514361128043e+01 - 1.016477803538e+01j]])
    np.testing.assert_allclose(array.spherical_harmonics(pos, orders=3), expected_result)
