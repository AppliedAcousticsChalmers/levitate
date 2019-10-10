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

    assert levitate.fields.GorkovPotential(array) == levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array)))
    assert levitate.fields.GorkovPotential(array, radius_sphere=1e-3) != levitate.fields.GorkovPotential(array, radius_sphere=1.1e-3)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array, sphere_material=levitate.materials.air)

    assert levitate.fields.GorkovGradient(array) == levitate.fields.GorkovGradient(array)
    assert levitate.fields.GorkovGradient(array) == pickle.loads(pickle.dumps(levitate.fields.GorkovGradient(array)))
    assert levitate.fields.GorkovGradient(array, radius_sphere=1e-3) != levitate.fields.GorkovGradient(array, radius_sphere=1.1e-3)
    assert levitate.fields.GorkovGradient(array) != levitate.fields.GorkovGradient(array, sphere_material=levitate.materials.air)

    assert levitate.fields.GorkovLaplacian(array) == levitate.fields.GorkovLaplacian(array)
    assert levitate.fields.GorkovLaplacian(array) == pickle.loads(pickle.dumps(levitate.fields.GorkovLaplacian(array)))
    assert levitate.fields.GorkovLaplacian(array, radius_sphere=1e-3) != levitate.fields.GorkovLaplacian(array, radius_sphere=1.1e-3)
    assert levitate.fields.GorkovLaplacian(array) != levitate.fields.GorkovLaplacian(array, sphere_material=levitate.materials.air)


def test_radiation_force_parameters():

    assert levitate.fields.RadiationForce(array) == levitate.fields.RadiationForce(array)
    assert levitate.fields.RadiationForce(array) == pickle.loads(pickle.dumps(levitate.fields.RadiationForce(array)))
    assert levitate.fields.RadiationForce(array, radius_sphere=1e-3) != levitate.fields.RadiationForce(array, radius_sphere=1.1e-3)
    assert levitate.fields.RadiationForce(array) != levitate.fields.RadiationForce(array, sphere_material=levitate.materials.air)

    assert levitate.fields.RadiationForceStiffness(array) == levitate.fields.RadiationForceStiffness(array)
    assert levitate.fields.RadiationForceStiffness(array) == pickle.loads(pickle.dumps(levitate.fields.RadiationForceStiffness(array)))
    assert levitate.fields.RadiationForceStiffness(array, radius_sphere=1e-3) != levitate.fields.RadiationForceStiffness(array, radius_sphere=1.1e-3)
    assert levitate.fields.RadiationForceStiffness(array) != levitate.fields.RadiationForceStiffness(array, sphere_material=levitate.materials.air)

    assert levitate.fields.RadiationForceCurl(array) == levitate.fields.RadiationForceCurl(array)
    assert levitate.fields.RadiationForceCurl(array) == pickle.loads(pickle.dumps(levitate.fields.RadiationForceCurl(array)))
    assert levitate.fields.RadiationForceCurl(array, radius_sphere=1e-3) != levitate.fields.RadiationForceCurl(array, radius_sphere=1.1e-3)
    assert levitate.fields.RadiationForceCurl(array) != levitate.fields.RadiationForceCurl(array, sphere_material=levitate.materials.air)

    assert levitate.fields.RadiationForceGradient(array) == levitate.fields.RadiationForceGradient(array)
    assert levitate.fields.RadiationForceGradient(array) == pickle.loads(pickle.dumps(levitate.fields.RadiationForceGradient(array)))
    assert levitate.fields.RadiationForceGradient(array, radius_sphere=1e-3) != levitate.fields.RadiationForceGradient(array, radius_sphere=1.1e-3)
    assert levitate.fields.RadiationForceGradient(array) != levitate.fields.RadiationForceGradient(array, sphere_material=levitate.materials.air)


def test_spherical_harmonics_parameters():
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2) == levitate.fields.SphericalHarmonicsForce(array, orders=2)
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2) == pickle.loads(pickle.dumps(levitate.fields.SphericalHarmonicsForce(array, orders=2)))
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2) != levitate.fields.SphericalHarmonicsForce(array, orders=3)
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2) != levitate.fields.SphericalHarmonicsForce(array, orders=2, radius_sphere=1.1e-3)
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2) != levitate.fields.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible')
    assert levitate.fields.SphericalHarmonicsForce(array, orders=2, scattering_model='compressible') != levitate.fields.SphericalHarmonicsForce(array, orders=2, sphere_material=levitate.materials.air, scattering_model='compressible')


def test_direct_params():
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array_b)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array, weight=1)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array, position=pos)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array, weight=1, position=pos)


def test_simple_types():
    # Algorithm, should diff if array, algorithm, or type is different.
    assert levitate.fields.GorkovPotential(array) == levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array)))
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array_b)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovGradient(array)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) * 1 @ pos

    # UnboundCostFunction, should also diff if weight is different
    assert levitate.fields.GorkovPotential(array) * 1 == levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) * 1 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1))
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) * 2
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) * 1 @ pos

    # BoundAlgorithm, should also diff if position is different
    assert levitate.fields.GorkovPotential(array) @ pos == levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) @ pos == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) @ pos))
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) * 4
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) @ pos * 1

    # CostFunction, should diff if position or weight is different
    assert levitate.fields.GorkovPotential(array) * 1 @ pos == levitate.fields.GorkovPotential(array) * 1 @ pos
    assert levitate.fields.GorkovPotential(array) * 1 @ pos == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 @ pos))
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 1 @ pos_b
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 2 @ pos
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 2 @ pos_b
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) @ pos


def test_magnitude_squared_types():
    # These should diff if the algorithm is different, or if the target "vector" is different.
    # MagnitudeSquaredAlgorithm
    assert levitate.fields.GorkovPotential(array) - 0 == levitate.fields.GorkovPotential(array) - 0
    assert levitate.fields.GorkovPotential(array) - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) - 0))
    assert levitate.fields.GorkovPotential(array) - 0 != levitate.fields.GorkovGradient(array) - 0
    assert levitate.fields.GorkovPotential(array) - 0 != levitate.fields.GorkovPotential(array) - 1

    # MagnitudeSquaredUnboundCostFunction
    assert levitate.fields.GorkovPotential(array) * 1 - 0 == levitate.fields.GorkovPotential(array) * 1 - 0
    assert levitate.fields.GorkovPotential(array) * 1 - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 - 0))
    assert levitate.fields.GorkovPotential(array) * 1 - 0 != levitate.fields.GorkovPotential(array) * 1 - 1

    # MagnitudeSquaredBoundAlgorithm
    assert levitate.fields.GorkovPotential(array) @ pos - 0 == levitate.fields.GorkovPotential(array) @ pos - 0
    assert levitate.fields.GorkovPotential(array) @ pos - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) @ pos - 0))
    assert levitate.fields.GorkovPotential(array) @ pos - 0 != levitate.fields.GorkovPotential(array) @ pos - 1

    # MagnitudeSquaredCostFunction
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 == levitate.fields.GorkovPotential(array) * 1 @ pos - 0
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 @ pos - 0))
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 != levitate.fields.GorkovPotential(array) * 1 @ pos - 1


def test_points_collections():
    # Should diff if any one algorithm is different, or if they have a different order, or if the type is different.
    alg_a = levitate.fields.GorkovPotential(array)
    alg_b = levitate.fields.GorkovGradient(array)

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
