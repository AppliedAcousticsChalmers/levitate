import pytest
import numpy as np
import levitate

array = levitate.arrays.RectangularArray(shape=(4, 5))
pos_0 = np.array([0.1, 0.2, 0.3])
pos_1 = np.array([-0.15, 1.27, 0.001])
pos_both = np.stack((pos_0, pos_1), axis=1)
array.phases = array.focus_phases((pos_0 + pos_1) / 2) + array.signature(stype='twin')

spat_ders = array.pressure_derivs(pos_both, orders=3)
ind_ders = np.einsum('i, ji...->ji...', array.amplitudes * np.exp(1j * array.phases), spat_ders)
sum_ders = np.sum(ind_ders, axis=1)


# Defines the algorithms to use for testing.
# Note that the algorithm implementations themselves are tested elsewhere.
pressure_derivs_algorithms = [
    levitate.algorithms.GorkovGradient,
    levitate.algorithms.GorkovLaplacian,
]


@pytest.mark.parametrize("func", pressure_derivs_algorithms)
def test_Algorithm(func):
    algorithm = func(array)
    calc_values = algorithm.values

    val_0 = algorithm(array.complex_amplitudes, pos_0)
    val_1 = algorithm(array.complex_amplitudes, pos_1)
    val_both = algorithm(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, calc_values(pressure_derivs_summed=sum_ders[..., 0]))
    np.testing.assert_allclose(val_1, calc_values(pressure_derivs_summed=sum_ders[..., 1]))
    np.testing.assert_allclose(val_both, np.stack([val_0, val_1], axis=1))


@pytest.mark.parametrize("pos", [pos_0, pos_1, pos_both])
@pytest.mark.parametrize("func", pressure_derivs_algorithms)
def test_BoundAlgorithm(func, pos):
    algorithm = func(array)
    np.testing.assert_allclose((algorithm@pos)(array.complex_amplitudes), algorithm(array.complex_amplitudes, pos))


@pytest.mark.parametrize("weight", [np.random.uniform(-10, 10, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("func", pressure_derivs_algorithms)
def test_UnboundCostFunction(func, weight):
    algorithm = func(array) * weight
    calc_values, calc_jacobians = algorithm.values, algorithm.jacobians

    val_0 = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders[..., 0]), np.atleast_1d(weight))
    val_1 = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders[..., 1]), np.atleast_1d(weight))
    val_both = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders), np.atleast_1d(weight))

    jac_0 = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders[..., 0], pressure_derivs_individual=ind_ders[..., 0]), np.atleast_1d(weight))
    jac_1 = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders[..., 1], pressure_derivs_individual=ind_ders[..., 1]), np.atleast_1d(weight))
    jac_both = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders, pressure_derivs_individual=ind_ders), np.atleast_1d(weight))

    alg_val_0, alg_jac_0 = algorithm(array.complex_amplitudes, pos_0)
    alg_val_1, alg_jac_1 = algorithm(array.complex_amplitudes, pos_1)
    alg_val_both, alg_jac_both = algorithm(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, alg_val_0)
    np.testing.assert_allclose(val_1, alg_val_1)
    np.testing.assert_allclose(val_both, alg_val_both)

    np.testing.assert_allclose(jac_0, alg_jac_0)
    np.testing.assert_allclose(jac_1, alg_jac_1)
    np.testing.assert_allclose(jac_both, alg_jac_both)


@pytest.mark.parametrize("weight", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("pos", [pos_0, pos_1])
@pytest.mark.parametrize("func", pressure_derivs_algorithms)
def test_CostFunction(func, weight, pos):
    algorithm = func(array) * weight

    val, jac = (algorithm@pos)(array.complex_amplitudes)
    val_ub, jac_ub = algorithm(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val, val_ub)
    np.testing.assert_allclose(jac, jac_ub)


@pytest.mark.parametrize("pos", [pos_0, pos_both])
@pytest.mark.parametrize("func", pressure_derivs_algorithms)
@pytest.mark.parametrize("target", [(1, 0, 0), (0, 1, 0), (0, 0, 1), np.random.uniform(-10, 10, 3)])
def test_MagnitudeSquaredAlgorithm(func, target, pos):
    algorithm = func(array)
    np.testing.assert_allclose((algorithm - target)(array.complex_amplitudes, pos), np.abs(algorithm(array.complex_amplitudes, pos) - np.asarray(target).reshape([-1] + (pos.ndim - 1) * [1]))**2)


@pytest.mark.parametrize("pos", [pos_0, pos_both])
@pytest.mark.parametrize("func", pressure_derivs_algorithms)
@pytest.mark.parametrize("target", [np.random.uniform(-10, 10, 3)])
def test_MagnitudeSquaredBoundAlgorithm(func, target, pos):
    algorithm = func(array) - target
    np.testing.assert_allclose((algorithm@pos)(array.complex_amplitudes), algorithm(array.complex_amplitudes, pos))


@pytest.mark.parametrize("func", pressure_derivs_algorithms)
@pytest.mark.parametrize("weight", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("target", [(1, 0, 0), (0, 1, 0), (0, 0, 1), np.random.uniform(-10, 10, 3)])
def test_MagnitudeSquaredUnboundCostFunction(func, target, weight):
    algorithm = (func(array) - target) * weight
    calc_values, calc_jacobians = algorithm.values, algorithm.jacobians

    val_0 = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders[..., 0]), np.atleast_1d(weight))
    val_1 = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders[..., 1]), np.atleast_1d(weight))
    val_both = np.einsum('i..., i', calc_values(pressure_derivs_summed=sum_ders), np.atleast_1d(weight))

    jac_0 = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders[..., 0], pressure_derivs_individual=ind_ders[..., 0]), np.atleast_1d(weight))
    jac_1 = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders[..., 1], pressure_derivs_individual=ind_ders[..., 1]), np.atleast_1d(weight))
    jac_both = np.einsum('i..., i', calc_jacobians(pressure_derivs_summed=sum_ders, pressure_derivs_individual=ind_ders), np.atleast_1d(weight))

    alg_val_0, alg_jac_0 = algorithm(array.complex_amplitudes, pos_0)
    alg_val_1, alg_jac_1 = algorithm(array.complex_amplitudes, pos_1)
    alg_val_both, alg_jac_both = algorithm(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val_0, alg_val_0)
    np.testing.assert_allclose(val_1, alg_val_1)
    np.testing.assert_allclose(val_both, alg_val_both)

    np.testing.assert_allclose(jac_0, alg_jac_0)
    np.testing.assert_allclose(jac_1, alg_jac_1)
    np.testing.assert_allclose(jac_both, alg_jac_both)


@pytest.mark.parametrize("func", pressure_derivs_algorithms)
@pytest.mark.parametrize("weight", [np.random.uniform(-10, 10, 1), np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("target", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("pos", [pos_0, pos_both])
def test_MagnitudeSquaredCostFunction(func, weight, target, pos):
    algorithm = (func(array) - target) * weight
    val, jac = (algorithm@pos)(array.complex_amplitudes)
    val_ub, jac_ub = algorithm(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val, val_ub)
    np.testing.assert_allclose(jac, jac_ub)


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
def test_AlgorithmPoint(func0, func1):
    alg0 = func0(array)
    alg1 = func1(array)

    val0 = alg0(array.complex_amplitudes, pos_both)
    val1 = alg1(array.complex_amplitudes, pos_both)
    val_both = (alg0 + alg1)(array.complex_amplitudes, pos_both)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
def test_BoundAlgorithmPoint(func0, func1):
    alg = func0(array) + func1(array)

    val_alg = alg(array.complex_amplitudes, pos_both)
    val_bound = (alg@pos_both)(array.complex_amplitudes)
    np.testing.assert_allclose(val_alg[0], val_bound[0])
    np.testing.assert_allclose(val_alg[1], val_bound[1])


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
@pytest.mark.parametrize("pos", [pos_0])
@pytest.mark.parametrize("weight0", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("weight1", [np.random.uniform(-10, 10, 3)])
def test_UnboundCostFunctionPoint(func0, func1, pos, weight0, weight1):
    alg0 = func0(array) * weight0
    alg1 = func1(array) * weight1

    val0, jac0 = alg0(array.complex_amplitudes, pos)
    val1, jac1 = alg1(array.complex_amplitudes, pos)
    val_both, jac_both = (alg0 + alg1)(array.complex_amplitudes, pos)
    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
@pytest.mark.parametrize("pos", [pos_0, pos_1])
@pytest.mark.parametrize("weight0", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("weight1", [np.random.uniform(-10, 10, 3)])
def test_CostFunctionPoint(func0, func1, pos, weight0, weight1):
    alg0 = func0(array) * weight0 @ pos
    alg1 = func1(array) * weight1 @ pos

    val0, jac0 = alg0(array.complex_amplitudes)
    val1, jac1 = alg1(array.complex_amplitudes)
    val_both, jac_both = (alg0 + alg1)(array.complex_amplitudes)
    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
@pytest.mark.parametrize("pos0", [pos_0, pos_1])
@pytest.mark.parametrize("pos1", [pos_0, pos_1])
def test_AlgorithmCollection(func0, func1, pos0, pos1):
    alg0 = func0(array)@pos0
    alg1 = func1(array)@pos1
    alg_both = alg0 + alg1

    val0 = alg0(array.complex_amplitudes)
    val1 = alg1(array.complex_amplitudes)
    val_both = alg_both(array.complex_amplitudes)

    np.testing.assert_allclose(val0, val_both[0])
    np.testing.assert_allclose(val1, val_both[1])


@pytest.mark.parametrize("func0", pressure_derivs_algorithms)
@pytest.mark.parametrize("func1", pressure_derivs_algorithms)
@pytest.mark.parametrize("pos0", [pos_0, pos_1])
@pytest.mark.parametrize("pos1", [pos_0, pos_1])
@pytest.mark.parametrize("weight0", [np.random.uniform(-10, 10, 3)])
@pytest.mark.parametrize("weight1", [np.random.uniform(-10, 10, 3)])
def test_CostFunctionCollection(func0, func1, pos0, pos1, weight0, weight1):
    alg0 = func0(array)@pos0*weight0
    alg1 = func1(array)@pos1*weight1
    alg_both = alg0 + alg1

    val0, jac0 = alg0(array.complex_amplitudes)
    val1, jac1 = alg1(array.complex_amplitudes)
    val_both, jac_both = alg_both(array.complex_amplitudes)

    np.testing.assert_allclose(val0 + val1, val_both)
    np.testing.assert_allclose(jac0 + jac1, jac_both)
