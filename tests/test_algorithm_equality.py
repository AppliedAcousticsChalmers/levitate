import numpy as np
import levitate
import pickle

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

pos = np.array([0.1, 0.2, 0.3])
pos_b = np.array([-0.15, 1.27, 0.001])
array = levitate.arrays.RectangularArray(shape=(4, 5))
array_b = levitate.arrays.RectangularArray(shape=(5, 4))


def test_gorkov_parameters():

    assert levitate.algorithms.GorkovPotential(array) == levitate.algorithms.GorkovPotential(array)
    assert levitate.algorithms.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array)))
    assert levitate.algorithms.GorkovPotential(array, radius_sphere=1e-3) != levitate.algorithms.GorkovPotential(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovPotential(array) != levitate.algorithms.GorkovPotential(array, sphere_material=levitate.materials.air)

    assert levitate.algorithms.GorkovGradient(array) == levitate.algorithms.GorkovGradient(array)
    assert levitate.algorithms.GorkovGradient(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovGradient(array)))
    assert levitate.algorithms.GorkovGradient(array, radius_sphere=1e-3) != levitate.algorithms.GorkovGradient(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovGradient(array) != levitate.algorithms.GorkovGradient(array, sphere_material=levitate.materials.air)

    assert levitate.algorithms.GorkovLaplacian(array) == levitate.algorithms.GorkovLaplacian(array)
    assert levitate.algorithms.GorkovLaplacian(array) == pickle.loads(pickle.dumps(levitate.algorithms.GorkovLaplacian(array)))
    assert levitate.algorithms.GorkovLaplacian(array, radius_sphere=1e-3) != levitate.algorithms.GorkovLaplacian(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.GorkovLaplacian(array) != levitate.algorithms.GorkovLaplacian(array, sphere_material=levitate.materials.air)


def test_radiation_force_parameters():

    assert levitate.algorithms.RadiationForce(array) == levitate.algorithms.RadiationForce(array)
    assert levitate.algorithms.RadiationForce(array) == pickle.loads(pickle.dumps(levitate.algorithms.RadiationForce(array)))
    assert levitate.algorithms.RadiationForce(array, radius_sphere=1e-3) != levitate.algorithms.RadiationForce(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.RadiationForce(array) != levitate.algorithms.RadiationForce(array, sphere_material=levitate.materials.air)

    assert levitate.algorithms.RadiationForceStiffness(array) == levitate.algorithms.RadiationForceStiffness(array)
    assert levitate.algorithms.RadiationForceStiffness(array) == pickle.loads(pickle.dumps(levitate.algorithms.RadiationForceStiffness(array)))
    assert levitate.algorithms.RadiationForceStiffness(array, radius_sphere=1e-3) != levitate.algorithms.RadiationForceStiffness(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.RadiationForceStiffness(array) != levitate.algorithms.RadiationForceStiffness(array, sphere_material=levitate.materials.air)

    assert levitate.algorithms.RadiationForceCurl(array) == levitate.algorithms.RadiationForceCurl(array)
    assert levitate.algorithms.RadiationForceCurl(array) == pickle.loads(pickle.dumps(levitate.algorithms.RadiationForceCurl(array)))
    assert levitate.algorithms.RadiationForceCurl(array, radius_sphere=1e-3) != levitate.algorithms.RadiationForceCurl(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.RadiationForceCurl(array) != levitate.algorithms.RadiationForceCurl(array, sphere_material=levitate.materials.air)

    assert levitate.algorithms.RadiationForceGradient(array) == levitate.algorithms.RadiationForceGradient(array)
    assert levitate.algorithms.RadiationForceGradient(array) == pickle.loads(pickle.dumps(levitate.algorithms.RadiationForceGradient(array)))
    assert levitate.algorithms.RadiationForceGradient(array, radius_sphere=1e-3) != levitate.algorithms.RadiationForceGradient(array, radius_sphere=1.1e-3)
    assert levitate.algorithms.RadiationForceGradient(array) != levitate.algorithms.RadiationForceGradient(array, sphere_material=levitate.materials.air)


def test_spherical_harmonics_parameters():
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) == levitate.algorithms.SphericalHarmonicsForce(array, orders=2)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) == pickle.loads(pickle.dumps(levitate.algorithms.SphericalHarmonicsForce(array, orders=2)))
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=3)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, radius_sphere=1.1e-3)
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2) != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible')
    assert levitate.algorithms.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible') != levitate.algorithms.SphericalHarmonicsForce(array, orders=2, sphere_material=levitate.materials.air, scattering_model='compressible')


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


def test_magnitude_squared_types():
    # These should diff if the algorithm is different, or if the target "vector" is different.
    # MagnitudeSquaredAlgorithm
    assert levitate.algorithms.GorkovPotential(array) - 0 == levitate.algorithms.GorkovPotential(array) - 0
    assert levitate.algorithms.GorkovPotential(array) - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) - 0))
    assert levitate.algorithms.GorkovPotential(array) - 0 != levitate.algorithms.GorkovGradient(array) - 0
    assert levitate.algorithms.GorkovPotential(array) - 0 != levitate.algorithms.GorkovPotential(array) - 1

    # MagnitudeSquaredUnboundCostFunction
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 == levitate.algorithms.GorkovPotential(array) * 1 - 0
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) * 1 - 0))
    assert levitate.algorithms.GorkovPotential(array) * 1 - 0 != levitate.algorithms.GorkovPotential(array) * 1 - 1

    # MagnitudeSquaredBoundAlgorithm
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 == levitate.algorithms.GorkovPotential(array) @ pos - 0
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 == pickle.loads(pickle.dumps(levitate.algorithms.GorkovPotential(array) @ pos - 0))
    assert levitate.algorithms.GorkovPotential(array) @ pos - 0 != levitate.algorithms.GorkovPotential(array) @ pos - 1

    # MagnitudeSquaredCostFunction
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
