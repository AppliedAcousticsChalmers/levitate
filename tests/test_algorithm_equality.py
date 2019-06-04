import numpy as np
import levitate
import pickle

pos = np.array([0.1, 0.2, 0.3])
pos_b = np.array([-0.15, 1.27, 0.001])
array = levitate.arrays.RectangularArray(shape=(4, 5))
array_b = levitate.arrays.RectangularArray(shape=(5, 4))


def test_gorkov_parameters():

    assert levitate.algorithms.GorkovPotential(array) == levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array)))
    assert levitate.algorithms.GorkovPotential(array, radius_sphere=1e-3) != levitate.algorithms.GorkovPotential(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array, sphere_material=levitate.materials.Air)

    assert levitate.algorithms.GorkovGradient(array) == levitate.algorithms.GorkovGradient(array)
    assert levitate.algorithms.GorkovGradient(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovGradient(array)))
    assert levitate.algorithms.GorkovGradient(array, radius_sphere=1e-3) != levitate.algorithms.GorkovGradient(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovGradient(array) != levitate.algorithms.GorkovGradient(array, sphere_material=levitate.materials.Air)

    assert levitate.algorithms.GorkovLaplacian(array) == levitate.algorithms.GorkovLaplacian(array)
    assert levitate.algorithms.GorkovLaplacian(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovLaplacian(array)))
    assert levitate.algorithms.GorkovLaplacian(array, radius_sphere=1e-3) != levitate.algorithms.GorkovLaplacian(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovLaplacian(array) != levitate.algorithms.GorkovLaplacian(array, sphere_material=levitate.materials.Air)


def test_second_order_parameters():

    assert levitate.algorithms.SecondOrderForce(array) == levitate.algorithms.SecondOrderForce(array)
    assert levitate.algorithms.SecondOrderForce(array) == pickle.loads(pickle.dumps(levitate.algorithms.SecondOrderForce(array)))
    assert levitate.algorithms.SecondOrderForce(array, radius_sphere=1e-3) != levitate.algorithms.SecondOrderForce(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.SecondOrderForce(array) != levitate.algorithms.SecondOrderForce(array, sphere_material=levitate.materials.Air)

    assert levitate.algorithms.SecondOrderStiffness(array) == levitate.algorithms.SecondOrderStiffness(array)
    assert levitate.algorithms.SecondOrderStiffness(array) == pickle.loads(pickle.dumps(levitate.algorithms.SecondOrderStiffness(array)))
    assert levitate.algorithms.SecondOrderStiffness(array, radius_sphere=1e-3) != levitate.algorithms.SecondOrderStiffness(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.SecondOrderStiffness(array) != levitate.algorithms.SecondOrderStiffness(array, sphere_material=levitate.materials.Air)

    assert levitate.algorithms.SecondOrderCurl(array) == levitate.algorithms.SecondOrderCurl(array)
    assert levitate.algorithms.SecondOrderCurl(array) == pickle.loads(pickle.dumps(levitate.algorithms.SecondOrderCurl(array)))
    assert levitate.algorithms.SecondOrderCurl(array, radius_sphere=1e-3) != levitate.algorithms.SecondOrderCurl(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.SecondOrderCurl(array) != levitate.algorithms.SecondOrderCurl(array, sphere_material=levitate.materials.Air)

    assert levitate.algorithms.SecondOrderForceGradient(array) == levitate.algorithms.SecondOrderForceGradient(array)
    assert levitate.algorithms.SecondOrderForceGradient(array) == pickle.loads(pickle.dumps(levitate.algorithms.SecondOrderForceGradient(array)))
    assert levitate.algorithms.SecondOrderForceGradient(array, radius_sphere=1e-3) != levitate.algorithms.SecondOrderForceGradient(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.SecondOrderForceGradient(array) != levitate.algorithms.SecondOrderForceGradient(array, sphere_material=levitate.materials.Air)


def test_spherical_harmonics_parameters():
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) == levitate.algorithms.SphericalHarmonicsForce(array, orders=2)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) == pickle.loads(pickle.dumps(levitate.algorithms.SphericalHarmonicsForce(array, orders=2)))
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=3)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, radius_sphere=1.1e-3)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible')
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible') != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, sphere_material=levitate.materials.Air, scattering_model='compressible')


def test_direct_params():
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array_b)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array, weight=1)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array, position=pos)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array, weight=1, position=pos)


def test_simple_types():
    # Algorithm, should diff if array, algorithm, or type is different.
    assert levitate.algorithms.GorkovPotential(array) == levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array)))
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array_b)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovGradient(array)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array) * 1
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array) @ pos
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array) * 1 @ pos

    # UnboundCostFunction, should also diff if weight is different
    assert levitate.algorithms.GorkovPotential(array) * 1 == levitate.algorithms.GorkovPotential(array) * 1
    assert levitate.algorithms.GorkovPotential(array) * 1 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) * 1))
    assert levitate.algorithms.GorkovPotential(array) * 1 != levitate.algorithms.GorkovPotential(array) * 2
    assert levitate.algorithms.GorkovPotential(array) * 1 != levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) * 1 != levitate.algorithms.GorkovPotential(array) @ pos
    assert levitate.algorithms.GorkovPotential(array) * 1 != levitate.algorithms.GorkovPotential(array) * 1 @ pos

    # BoundAlgorithm, should also diff if position is different
    assert levitate.algorithms.GorkovPotential(array) @ pos == levitate.algorithms.GorkovPotential(array) @ pos
    assert levitate.algorithms.GorkovPotential(array) @ pos == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) @ pos))
    assert levitate.algorithms.GorkovPotential(array) @ pos != levitate.algorithms.GorkovPotential(array) * 4
    assert levitate.algorithms.GorkovPotential(array) @ pos != levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) @ pos != levitate.algorithms.GorkovPotential(array) * 1
    assert levitate.algorithms.GorkovPotential(array) @ pos != levitate.algorithms.GorkovPotential(array) @ pos * 1

    # CostFunction, should diff if position or weight is different
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos == levitate.algorithms.GorkovPotential(array) * 1 @ pos
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) * 1 @ pos))
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array) * 1 @ pos_b
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array) * 2 @ pos
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array) * 2 @ pos_b
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array) * 1
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos != levitate.algorithms.GorkovPotential(array) @ pos


def test_vector_types():
    # These should diff if the algorithm is different, or if the target "vector" is different.
    # VectorAlgorithm
    assert levitate.algorithms.GorkovPotential(array) - 0 == levitate.algorithms.GorkovPotential(array) - 0
    assert levitate.algorithms.GorkovPotential(array) - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) - 0))
    assert levitate.algorithms.GorkovPotential(array) - 0 != levitate.algorithms.GorkovGradient(array) - 0
    assert levitate.algorithms.GorkovPotential(array) - 0 != levitate.algorithms.GorkovPotential(array) - 1

    # VectorUnboundCostFunction
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 == levitate.algorithms.GorkovPotential(array) * 1 - 0
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) * 1 - 0))
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 != levitate.algorithms.GorkovPotential(array) * 1 - 1

    # VectorBoundAlgorithm
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 == levitate.algorithms.GorkovPotential(array) @ pos - 0
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) @ pos - 0))
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 != levitate.algorithms.GorkovPotential(array) @ pos - 1

    # VectorCostFunction
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos - 0 == levitate.algorithms.GorkovPotential(array) * 1 @ pos - 0
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) * 1 @ pos - 0))
    assert levitate.algorithms.GorkovPotential(array) * 1 @ pos - 0 != levitate.algorithms.GorkovPotential(array) * 1 @ pos - 1


def test_points_collections():
    # Should diff if any one algorithm is different, or if they have a different order, or if the type is different.
    alg_a = levitate.algorithms.GorkovPotential(array)
    alg_b = levitate.algorithms.GorkovGradient(array)

    # AlgorithmPoint
    assert alg_a + alg_a == alg_a + alg_a
    assert alg_a + alg_a == pickle.loads(pickle.dumps(alg_a + alg_a))
    assert alg_a + alg_b == alg_a + alg_b
    assert alg_a + alg_a != alg_a + alg_b
    assert alg_a + alg_b != alg_b + alg_a

    # BoundAlgorithmPoint/Collection
    assert alg_a @ pos + alg_a @ pos == (alg_a + alg_a) @ pos
    assert alg_a @ pos + alg_a @ pos_b == alg_a @ pos + alg_a @ pos_b
    assert alg_a @ pos + alg_a @ pos != alg_a @ pos_b + alg_a @ pos_b
    assert alg_a @ pos + alg_a @ pos != alg_a @ pos_b + alg_a @ pos
    assert alg_a @ pos + alg_a @ pos != alg_a @ pos + alg_b @ pos

    # UnboundCostFunctionPoint
    assert alg_a * 2 + alg_a * 2 == (alg_a + alg_a) * 2
    assert alg_a * 2 + alg_a * 2 != alg_a * 4 + alg_a * 4
    assert alg_a * 2 + alg_a * 2 != alg_a * 2 + alg_b * 2

    # CostFunctionPoint
    assert (alg_a * 2 + alg_a * 2) @ pos == alg_a * 2 @ pos + alg_a * 2 @ pos
    assert (alg_a @ pos + alg_a @ pos_b) * 2 == alg_a * 2 @ pos + alg_a * 2 @ pos_b
    assert (alg_a @ pos + alg_a @ pos_b) * 2 == pickle.loads(pickle.dumps(alg_a * 2 @ pos + alg_a * 2 @ pos_b))
    assert (alg_a @ pos + alg_a @ pos_b) * 2 != alg_b * 2 @ pos + alg_a * 2 @ pos
    assert (alg_a * 2 + alg_a * 2) @ pos != (alg_a * 2 + alg_a * 2) @ pos_b
    assert (alg_a * 2 + alg_a * 2) @ pos != (alg_a * 2 + alg_b * 2) @ pos_b
    assert (alg_a * 2 + alg_a * 2) @ pos != (alg_a * 2 + alg_a * 4) @ pos
