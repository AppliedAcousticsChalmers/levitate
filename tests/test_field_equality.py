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


def test_spheherical_harmonics_parameters():
    assert levitate.fields.SphericalHarmonicsExpansion(array, orders=3) == levitate.fields.SphericalHarmonicsExpansion(array, orders=3)
    assert levitate.fields.SphericalHarmonicsExpansion(array, orders=3) == pickle.loads(pickle.dumps(levitate.fields.SphericalHarmonicsExpansion(array, orders=3)))
    assert levitate.fields.SphericalHarmonicsExpansion(array, orders=3) != levitate.fields.SphericalHarmonicsExpansion(array, orders=4)

    assert levitate.fields.SphericalHarmonicsExpansion(array, orders=3) != levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3)
    assert levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3) == levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3)
    assert levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3) == pickle.loads(pickle.dumps(levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3)))
    assert levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=3) != levitate.fields.SphericalHarmonicsExpansionGradient(array, orders=4)


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


def test_spherical_harmonics_force_parameters():
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
    # Field, should diff if array, field, or type is different.
    assert levitate.fields.GorkovPotential(array) == levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array)))
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array_b)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovGradient(array)
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) != levitate.fields.GorkovPotential(array) * 1 @ pos

    # CostField, should also diff if weight is different
    assert levitate.fields.GorkovPotential(array) * 1 == levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) * 1 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1))
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) * 2
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) * 1 != levitate.fields.GorkovPotential(array) * 1 @ pos

    # FieldPoint, should also diff if position is different
    assert levitate.fields.GorkovPotential(array) @ pos == levitate.fields.GorkovPotential(array) @ pos
    assert levitate.fields.GorkovPotential(array) @ pos == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) @ pos))
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) * 4
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) @ pos != levitate.fields.GorkovPotential(array) @ pos * 1

    # CostFieldPoint, should diff if position or weight is different
    assert levitate.fields.GorkovPotential(array) * 1 @ pos == levitate.fields.GorkovPotential(array) * 1 @ pos
    assert levitate.fields.GorkovPotential(array) * 1 @ pos == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 @ pos))
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 1 @ pos_b
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 2 @ pos
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 2 @ pos_b
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array)
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) * 1
    assert levitate.fields.GorkovPotential(array) * 1 @ pos != levitate.fields.GorkovPotential(array) @ pos


def test_squared_types():
    # These should diff if the field is different, or if the target "vector" is different.
    # SquaredField
    assert levitate.fields.GorkovPotential(array) - 0 == levitate.fields.GorkovPotential(array) - 0
    assert levitate.fields.GorkovPotential(array) - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) - 0))
    assert levitate.fields.GorkovPotential(array) - 0 != levitate.fields.GorkovGradient(array) - 0
    assert levitate.fields.GorkovPotential(array) - 0 != levitate.fields.GorkovPotential(array) - 1

    # SquaredCostField
    assert levitate.fields.GorkovPotential(array) * 1 - 0 == levitate.fields.GorkovPotential(array) * 1 - 0
    assert levitate.fields.GorkovPotential(array) * 1 - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 - 0))
    assert levitate.fields.GorkovPotential(array) * 1 - 0 != levitate.fields.GorkovPotential(array) * 1 - 1

    # SquaredFieldPoint
    assert levitate.fields.GorkovPotential(array) @ pos - 0 == levitate.fields.GorkovPotential(array) @ pos - 0
    assert levitate.fields.GorkovPotential(array) @ pos - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) @ pos - 0))
    assert levitate.fields.GorkovPotential(array) @ pos - 0 != levitate.fields.GorkovPotential(array) @ pos - 1

    # SquaredCostFieldPoint
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 == levitate.fields.GorkovPotential(array) * 1 @ pos - 0
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 == pickle.loads(pickle.dumps(levitate.fields.GorkovPotential(array) * 1 @ pos - 0))
    assert levitate.fields.GorkovPotential(array) * 1 @ pos - 0 != levitate.fields.GorkovPotential(array) * 1 @ pos - 1


def test_multis():
    # Should diff if any one field is different, or if they have a different order, or if the type is different.
    field_a = levitate.fields.GorkovPotential(array)
    field_b = levitate.fields.GorkovGradient(array)

    # MultiField
    assert field_a + field_a == field_a + field_a
    assert field_a + field_a == pickle.loads(pickle.dumps(field_a + field_a))
    assert field_a + field_b == field_a + field_b
    assert field_a + field_a != field_a + field_b
    assert field_a + field_b != field_b + field_a

    # MultiFieldPoint / MultiPoint
    assert field_a @ pos + field_a @ pos == (field_a + field_a) @ pos
    assert field_a @ pos + field_a @ pos_b == field_a @ pos + field_a @ pos_b
    assert field_a @ pos + field_a @ pos != field_a @ pos_b + field_a @ pos_b
    assert field_a @ pos + field_a @ pos != field_a @ pos_b + field_a @ pos
    assert field_a @ pos + field_a @ pos != field_a @ pos + field_b @ pos

    # MultiCostField
    assert field_a * 2 + field_a * 2 == (field_a + field_a) * 2
    assert field_a * 2 + field_a * 2 != field_a * 4 + field_a * 4
    assert field_a * 2 + field_a * 2 != field_a * 2 + field_b * 2

    # MultiCostFieldPoint / MultiPoint
    assert (field_a * 2 + field_a * 2) @ pos == field_a * 2 @ pos + field_a * 2 @ pos
    assert (field_a @ pos + field_a @ pos_b) * 2 == field_a * 2 @ pos + field_a * 2 @ pos_b
    assert (field_a @ pos + field_a @ pos_b) * 2 == pickle.loads(pickle.dumps(field_a * 2 @ pos + field_a * 2 @ pos_b))
    assert (field_a @ pos + field_a @ pos_b) * 2 != field_b * 2 @ pos + field_a * 2 @ pos
    assert (field_a * 2 + field_a * 2) @ pos != (field_a * 2 + field_a * 2) @ pos_b
    assert (field_a * 2 + field_a * 2) @ pos != (field_a * 2 + field_b * 2) @ pos_b
    assert (field_a * 2 + field_a * 2) @ pos != (field_a * 2 + field_a * 4) @ pos
