"""A collection of levitation related mathematical implementations.

The algorithms is one of the most important parts of the package,
containing implementations of various ways to calculate levitate-related physical properties.
To simplify the management and manipulation of the implemented algorithms they are wrapped
in an additional abstraction layer, described in the :ref:`algorithm_wrappers` section.
The short description is that the classes implemented in the `~levitate.algorithms` module
will not return objects of the called class, but typically objects of `~levitate._algorithm.Algorithm`.
These objects support algebraic operations, like `+`, `*`, and `abs`. The full description of
what the different operands do can be found at the top of :ref:`algorithm_wrappers`.

.. autosummary::
    :nosignatures:

    Pressure
    Velocity
    GorkovGradient
    GorkovLaplacian
    GorkovPotential
    RadiationForce
    RadiationForceStiffness
    RadiationForceCurl
    RadiationForceGradient
    SphericalHarmonicsForce
"""

import numpy as np
from . import materials, utils
from ._algorithm import AlgorithmImplementation
from ._algorithms_legacy import gorkov_potential, gorkov_gradient, gorkov_laplacian  # noqa: F401
from ._algorithms_legacy import second_order_force, second_order_stiffness, second_order_curl, second_order_force_gradient  # noqa: F401
from ._algorithms_legacy import pressure_squared_magnitude, velocity_squared_magnitude  # noqa: F401


class GorkovPotential(AlgorithmImplementation):
    """
    Create gorkov potential calculation algorithm.

    Creates functions which calculates the gorkov gradient and the jacobian of
    the field specified using spaial derivatives of the pressure.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 0
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=1)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=1, pressure_derivs_individual=1)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pressure_coefficient, other.pressure_coefficient, atol=0)
            and np.allclose(self.gradient_coefficient, other.gradient_coefficient, atol=0)
        )

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        V = 4 / 3 * np.pi * radius_sphere**3
        monopole_coefficient = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        dipole_coefficient = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012
        preToVel = 1 / (array.omega * array.medium.rho)  # Converting velocity to pressure gradient using equation of motion
        self.pressure_coefficient = V / 4 * array.medium.compressibility * monopole_coefficient
        self.gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * array.medium.rho

    def values(self, pressure_derivs_summed):
        values = self.pressure_coefficient * np.real(pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[0]))
        values -= self.gradient_coefficient * np.real(pressure_derivs_summed[1:4] * np.conj(pressure_derivs_summed[1:4])).sum(axis=0)
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = self.pressure_coefficient * 2 * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[0])
        jacobians -= self.gradient_coefficient * 2 * (pressure_derivs_individual[1:4] * np.conj(pressure_derivs_summed[1:4, None])).sum(axis=0)
        return jacobians


class GorkovGradient(AlgorithmImplementation):
    """
    Create gorkov gradient calculation functions.

    Creates functions which calculates the gorkov gradient and the jacobian of
    the field specified using spaial derivatives of the pressure.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        V = 4 / 3 * np.pi * radius_sphere**3
        monopole_coefficient = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        dipole_coefficient = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012
        preToVel = 1 / (array.omega * array.medium.rho)  # Converting velocity to pressure gradient using equation of motion
        self.pressure_coefficient = V / 4 * array.medium.compressibility * monopole_coefficient
        self.gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * array.medium.rho

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pressure_coefficient, other.pressure_coefficient, atol=0)
            and np.allclose(self.gradient_coefficient, other.gradient_coefficient, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = np.real(self.pressure_coefficient * np.conj(pressure_derivs_summed[0]) * pressure_derivs_summed[1:4])  # Pressure parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[1]) * pressure_derivs_summed[[4, 7, 8]])  # Vx parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[2]) * pressure_derivs_summed[[7, 5, 9]])  # Vy parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[3]) * pressure_derivs_summed[[8, 9, 6]])  # Vz parts
        return values * 2

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[1:4] + np.conj(pressure_derivs_summed[1:4, None]) * pressure_derivs_individual[0])  # Pressure parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[4, 7, 8]] + np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[1])  # Vx parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[7, 5, 9]] + np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[2])  # Vy parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[8, 9, 6]] + np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[3])  # Vz parts
        return jacobians * 2


class GorkovLaplacian(AlgorithmImplementation):
    """
    Create gorkov laplacian calculation functions.

    Creates functions which calculates the laplacian of the gorkov potential
    and the jacobian of the field specified using spaial derivatives of the pressure.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        V = 4 / 3 * np.pi * radius_sphere**3
        monopole_coefficient = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        dipole_coefficient = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012
        preToVel = 1 / (array.omega * array.medium.rho)  # Converting velocity to pressure gradient using equation of motion
        self.pressure_coefficient = V / 4 * array.medium.compressibility * monopole_coefficient
        self.gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * array.medium.rho

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pressure_coefficient, other.pressure_coefficient, atol=0)
            and np.allclose(self.gradient_coefficient, other.gradient_coefficient, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = np.real(self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_summed[[4, 5, 6]] + pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[[1, 2, 3]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_summed[[10, 15, 17]] + pressure_derivs_summed[[4, 7, 8]] * np.conj(pressure_derivs_summed[[4, 7, 8]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_summed[[13, 11, 18]] + pressure_derivs_summed[[7, 5, 9]] * np.conj(pressure_derivs_summed[[7, 5, 9]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_summed[[14, 16, 12]] + pressure_derivs_summed[[8, 9, 6]] * np.conj(pressure_derivs_summed[[8, 9, 6]])))
        return values * 2

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[4, 5, 6]] + np.conj(pressure_derivs_summed[[4, 5, 6], None]) * pressure_derivs_individual[0] + 2 * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[[1, 2, 3]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[10, 15, 17]] + np.conj(pressure_derivs_summed[[10, 15, 17], None]) * pressure_derivs_individual[1] + 2 * np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[[4, 7, 8]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[13, 11, 18]] + np.conj(pressure_derivs_summed[[13, 11, 18], None]) * pressure_derivs_individual[2] + 2 * np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[[7, 5, 9]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[14, 16, 12]] + np.conj(pressure_derivs_summed[[14, 16, 12], None]) * pressure_derivs_individual[3] + 2 * np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[[8, 9, 6]])
        return jacobians * 2


class RadiationForce(AlgorithmImplementation):
    """
    Create second order radiation force calculation functions.

    Creates functions which calculates the radiation force on a sphere generated
    by the field specified using spaial derivatives of the pressure, and the
    corresponding jacobians.

    This is more suitable than the Gor'kov formulation for use with progressive
    wave fiends, e.g. single sided arrays, see https://doi.org/10.1121/1.4773924.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

        ka = array.k * radius_sphere
        self.k_square = array.k**2
        self.psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
        self.psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
        self.force_coeff = -np.pi / array.k**5 * array.medium.compressibility

        # Including the j factors from the paper directly in the coefficients.
        self.psi_0 *= 1j
        self.psi_1 *= 1j

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.k_square, other.k_square, atol=0)
            and np.allclose(self.psi_0, other.psi_0, atol=0)
            and np.allclose(self.psi_1, other.psi_1, atol=0)
            and np.allclose(self.force_coeff, other.force_coeff, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = np.real(self.k_square * self.psi_0 * pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[[1, 2, 3]]))
        values += np.real(self.k_square * self.psi_1 * pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[0]))
        values += np.real(3 * self.psi_1 * pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[[4, 7, 8]]))
        values += np.real(3 * self.psi_1 * pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[[7, 5, 9]]))
        values += np.real(3 * self.psi_1 * pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[[8, 9, 6]]))
        return values * self.force_coeff

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = self.k_square * (self.psi_0 * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[[1, 2, 3], None]) + np.conj(self.psi_0) * np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[1, 2, 3]])
        jacobians += self.k_square * (self.psi_1 * pressure_derivs_individual[[1, 2, 3]] * np.conj(pressure_derivs_summed[0]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[0])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[1] * np.conj(pressure_derivs_summed[[4, 7, 8], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[4, 7, 8]])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[2] * np.conj(pressure_derivs_summed[[7, 5, 9], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[7, 5, 9]])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[3] * np.conj(pressure_derivs_summed[[8, 9, 6], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[8, 9, 6]])
        return jacobians * self.force_coeff


class RadiationForceStiffness(AlgorithmImplementation):
    """
    Create second order radiation stiffness calculation functions.

    Creates functions which calculates the radiation stiffness on a sphere
    generated by the field specified using spaial derivatives of
    the pressure, and the corresponding jacobians.

    This is more suitable than the Gor'kov formulation for use with progressive
    wave fiends, e.g. single sided arrays, see https://doi.org/10.1121/1.4773924.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

        ka = array.k * radius_sphere
        self.k_square = array.k**2
        self.psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
        self.psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
        self.force_coeff = -np.pi / array.k**5 * array.medium.compressibility

        # Including the j factors from the paper directly in the coefficients.
        self.psi_0 *= 1j
        self.psi_1 *= 1j

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.k_square, other.k_square, atol=0)
            and np.allclose(self.psi_0, other.psi_0, atol=0)
            and np.allclose(self.psi_1, other.psi_1, atol=0)
            and np.allclose(self.force_coeff, other.force_coeff, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = np.real(self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[[4, 5, 6]]) + pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[[1, 2, 3]])))
        values += np.real(self.k_square * self.psi_1 * (pressure_derivs_summed[[4, 5, 6]] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[[1, 2, 3]])))
        values += np.real(3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[[10, 15, 17]]) + pressure_derivs_summed[[4, 7, 8]] * np.conj(pressure_derivs_summed[[4, 7, 8]])))
        values += np.real(3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[[13, 11, 18]]) + pressure_derivs_summed[[7, 5, 9]] * np.conj(pressure_derivs_summed[[7, 5, 9]])))
        values += np.real(3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[[14, 16, 12]]) + pressure_derivs_summed[[8, 9, 6]] * np.conj(pressure_derivs_summed[[8, 9, 6]])))
        return values * self.force_coeff

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = self.k_square * (self.psi_0 * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[[4, 5, 6], None]) + np.conj(self.psi_0) * np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[4, 5, 6]] + (self.psi_0 + np.conj(self.psi_0)) * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[[1, 2, 3]])
        jacobians += self.k_square * (self.psi_1 * pressure_derivs_individual[[4, 5, 6]] * np.conj(pressure_derivs_summed[0]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[[4, 5, 6], None]) * pressure_derivs_individual[0] + (self.psi_1 + np.conj(self.psi_1)) * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[[1, 2, 3]])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[1] * np.conj(pressure_derivs_summed[[10, 15, 17], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[10, 15, 17]] + (self.psi_1 + np.conj(self.psi_1)) * np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[[4, 7, 8]])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[2] * np.conj(pressure_derivs_summed[[13, 11, 18], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[13, 11, 18]] + (self.psi_1 + np.conj(self.psi_1)) * np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[[7, 5, 9]])
        jacobians += 3 * (self.psi_1 * pressure_derivs_individual[3] * np.conj(pressure_derivs_summed[[14, 16, 12], None]) + np.conj(self.psi_1) * np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[14, 16, 12]] + (self.psi_1 + np.conj(self.psi_1)) * np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[[8, 9, 6]])
        return jacobians * self.force_coeff


class RadiationForceCurl(AlgorithmImplementation):

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

        ka = array.k * radius_sphere
        overall_coef = 2 * np.pi * array.medium.compressibility / array.k**5
        self.pressure_coefficient = -2 / 9 * ka**6 * (f_1**2 + f_1 * f_2) * array.k**2 * overall_coef
        self.velocity_coefficient = -3 * ka**6 / 18 * f_2**2 * overall_coef

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pressure_coefficient, other.pressure_coefficient, atol=0)
            and np.allclose(self.velocity_coefficient, other.velocity_coefficient, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = self.pressure_coefficient * np.imag(pressure_derivs_summed[[2, 3, 1]] * np.conj(pressure_derivs_summed[[3, 1, 2]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[7, 8, 4]] * np.conj(pressure_derivs_summed[[8, 4, 7]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[5, 9, 7]] * np.conj(pressure_derivs_summed[[9, 7, 5]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[9, 6, 8]] * np.conj(pressure_derivs_summed[[6, 8, 9]]))
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):
        jacobians = 1j * self.pressure_coefficient * (np.conj(pressure_derivs_summed[[2, 3, 1], None]) * pressure_derivs_individual[[3, 1, 2]] - np.conj(pressure_derivs_summed[[3, 1, 2], None]) * pressure_derivs_individual[[2, 3, 1]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[7, 8, 4], None]) * pressure_derivs_individual[[8, 4, 7]] - np.conj(pressure_derivs_summed[[8, 4, 7], None]) * pressure_derivs_individual[[7, 8, 4]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[5, 9, 7], None]) * pressure_derivs_individual[[9, 7, 5]] - np.conj(pressure_derivs_summed[[9, 7, 5], None]) * pressure_derivs_individual[[5, 9, 7]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[9, 6, 8], None]) * pressure_derivs_individual[[6, 8, 9]] - np.conj(pressure_derivs_summed[[6, 8, 9], None]) * pressure_derivs_individual[[9, 6, 8]])
        return jacobians


class RadiationForceGradient(AlgorithmImplementation):
    """
    Create second order radiation force gradient calculation function.

    Creates the algorithm object needed to calculated the gradient matrix of the
    radiation force on a small spherical bead.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    radius_sphere : float, default 1e-3
        Radius of the spherical beads.
    sphere_material : Material
        The material of the sphere, default Styrofoam.
    """

    ndim = 2
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.Styrofoam, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

        ka = array.k * radius_sphere
        self.k_square = array.k**2
        self.psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
        self.psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
        self.force_coeff = -np.pi / array.k**5 * array.medium.compressibility

        # Including the j factors from the paper directly in the coefficients.
        self.psi_0 *= 1j
        self.psi_1 *= 1j

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.k_square, other.k_square, atol=0)
            and np.allclose(self.psi_0, other.psi_0, atol=0)
            and np.allclose(self.psi_1, other.psi_1, atol=0)
            and np.allclose(self.force_coeff, other.force_coeff, atol=0)
        )

    def values(self, pressure_derivs_summed):
        values = np.zeros((3, 3) + pressure_derivs_summed.shape[1:])
        values[0, 0] = np.real(  # F_{x,x}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[4]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[1]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[4] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[1]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[10]) + pressure_derivs_summed[4] * np.conj(pressure_derivs_summed[4]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[13]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[14]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[8]))
        )
        values[0, 1] = np.real(  # F_{x,y}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[7]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[1]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[2]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[13]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[4]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[15]) + pressure_derivs_summed[5] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[8]))
        )
        values[0, 2] = np.real(  # F_{x,z}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[8]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[1]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[3]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[14]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[4]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[17]) + pressure_derivs_summed[6] * np.conj(pressure_derivs_summed[8]))
        )
        values[1, 0] = np.real(  # F_{y,x}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[7]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[2]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[1]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[13]) + pressure_derivs_summed[4] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[15]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[5]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[9]))
        )
        values[1, 1] = np.real(  # F_{y,y}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[5]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[2]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[5] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[2]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[15]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[11]) + pressure_derivs_summed[5] * np.conj(pressure_derivs_summed[5]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[16]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[9]))
        )
        values[1, 2] = np.real(  # F_{y,z}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[9]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[2]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[3]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[7]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[16]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[5]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[18]) + pressure_derivs_summed[6] * np.conj(pressure_derivs_summed[9]))
        )
        values[2, 0] = np.real(  # F_{z,x}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[8]) + pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[3]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[1]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[14]) + pressure_derivs_summed[4] * np.conj(pressure_derivs_summed[8]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[9]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[17]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[6]))
        )
        values[2, 1] = np.real(  # F_{z,y}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[9]) + pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[3]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[2]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[19]) + pressure_derivs_summed[7] * np.conj(pressure_derivs_summed[8]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[16]) + pressure_derivs_summed[5] * np.conj(pressure_derivs_summed[9]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[18]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[6]))
        )
        values[2, 2] = np.real(  # F_{z,z}
            self.k_square * self.psi_0 * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[6]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[3]))
            + self.k_square * self.psi_1 * (pressure_derivs_summed[6] * np.conj(pressure_derivs_summed[0]) + pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[3]))
            + 3 * self.psi_1 * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[17]) + pressure_derivs_summed[8] * np.conj(pressure_derivs_summed[8]))
            + 3 * self.psi_1 * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[18]) + pressure_derivs_summed[9] * np.conj(pressure_derivs_summed[9]))
            + 3 * self.psi_1 * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[12]) + pressure_derivs_summed[6] * np.conj(pressure_derivs_summed[6]))
        )
        return values * self.force_coeff


class Pressure(AlgorithmImplementation):
    """Create pressure calculation functions.

    Creates functions which calculates the sound pressure,
    and the corresponding jacobians.
    The main use of this is to use as a cost function.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array, optional.

    """

    ndim = 0
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=0)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_individual=0)

    def values(self, pressure_derivs_summed):
        return pressure_derivs_summed[0]

    def jacobians(self, pressure_derivs_individual):
        return pressure_derivs_individual[0]


class Velocity(AlgorithmImplementation):
    """Create particle velocity calculation functions.

    Creates functions which calculates the squared velocity magnitude,
    as a vector, and the corresponding jacobians.
    The main use of this is to use as a cost function.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.

    """

    ndim = 1
    values_require = AlgorithmImplementation.requirement(pressure_derivs_summed=1)
    jacobians_require = AlgorithmImplementation.requirement(pressure_derivs_individual=1)

    def __init__(self, array, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        self.pre_grad_2_vel = 1 / (1j * array.medium.rho * array.omega)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pre_grad_2_vel, other.pre_grad_2_vel, atol=0)
        )

    def values(self, pressure_derivs_summed):
        return self.pre_grad_2_vel * pressure_derivs_summed[1:4]

    def jacobians(self, pressure_derivs_individual):
        return self.pre_grad_2_vel * pressure_derivs_individual[1:4]


class SphericalHarmonicsForce(AlgorithmImplementation):
    ndim = 1

    def __init__(self, array, orders, radius_sphere=1e-3, sphere_material=materials.Styrofoam, scattering_model='Hard sphere', *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        self.values_require = AlgorithmImplementation.requirement(spherical_harmonics_summed=orders + 1)

        sph_idx = utils.SphericalHarmonicsIndexer(orders)
        from scipy.special import spherical_jn, spherical_yn
        # Create indexing arrays for sound field harmonics
        self.N_M = []  # Indices for the S_n^m coefficients
        self.Nr_M = []  # Indices for the S_(n+1)^m coefficients
        self.Nr_Mr = []  # Indices for the S_(n+1)^(m+1) coefficients
        self.N_mM = []  # Indices for the S_n^-m coefficients
        self.Nr_mMr = []  # Indices for the S_(n+1)^-(m+1) coefficients
        for n, m in sph_idx:
            self.N_M.append(sph_idx(n, m))
            self.Nr_M.append(sph_idx(n + 1, m))
            self.Nr_Mr.append(sph_idx(n + 1, m + 1))
            self.N_mM.append(sph_idx(n, -m))
            self.Nr_mMr.append(sph_idx(n + 1, -1 - m))

        # Calculate bessel functions, hankel functions, and their derivatives
        ka = array.k * radius_sphere
        n = np.arange(0, orders + 2)
        bessel_function = spherical_jn(n, ka)
        hankel_function = bessel_function + 1j * spherical_yn(n, ka)
        bessel_derivative = spherical_jn(n, ka, derivative=True)
        hankel_derivative = bessel_derivative + 1j * spherical_yn(n, ka, derivative=True)

        if 'hard' in scattering_model.lower():
            # See e.g. Gumerov, Duraiswami (2004): Eq. 4.2.10, p. 146
            scattering_coefficient = - bessel_derivative / hankel_derivative
        elif 'soft' in scattering_model.lower():
            # See e.g. Gumerov, Duraiswami (2004): Eq. 4.2.10, p. 146
            scattering_coefficient = - bessel_function / hankel_function
        elif 'compressible' in scattering_model.lower():
            # See Blackstock, Hamilton (2008): Eq. 6.88, p.193
            ka_interior = array.omega / sphere_material.c * radius_sphere
            bessel_function_interior = spherical_jn(n, ka_interior)
            # hankel_function_interior = bessel_function_interior + 1j * spherical_yn(n, ka_interior)
            bessel_derivative_interior = spherical_jn(n, ka_interior, derivative=True)
            # hankel_derivative_interior = bessel_derivative_interior + 1j * spherical_yn(n, ka_interior, derivative=True)

            relative_impedance = sphere_material.rho / array.medium.rho * sphere_material.c / array.medium.c
            numerator = bessel_function * bessel_derivative_interior - relative_impedance * bessel_derivative * bessel_function_interior
            denominator = hankel_function * bessel_derivative_interior - relative_impedance * hankel_derivative * bessel_function_interior
            scattering_coefficient = - numerator / denominator
        else:
            raise ValueError("Unknown scattering model '{}'".format(scattering_model))

        psi = np.zeros(orders + 1, dtype=np.complex128)
        for n in sph_idx.orders:
            psi[n] = 1j * (1 + 2 * scattering_coefficient[n]) * (1 + 2 * np.conj(scattering_coefficient[n + 1])) - 1j

        scaling = array.medium.compressibility / (8 * array.k**2)
        self.xy_coefficients = np.zeros((orders + 1)**2, dtype=np.complex128)
        self.z_coefficients = np.zeros((orders + 1)**2, dtype=np.complex128)
        idx = 0
        for n in sph_idx.orders:
            denom = 1 / ((2 * n + 1) * (2 * n + 3))**0.5
            for m in sph_idx.modes:
                self.xy_coefficients[idx] = psi[n] * ((n + m + 1) * (n + m + 2))**0.5 * denom * scaling
                self.z_coefficients[idx] = -2 * psi[n] * ((n + m + 1) * (n - m + 1))**0.5 * denom * scaling
                idx += 1

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.values_require['spherical_harmonics_summed'] == other.values_require['spherical_harmonics_summed']
            and np.allclose(self.xy_coefficients, other.xy_coefficients, atol=0)
            and np.allclose(self.z_coefficients, other.z_coefficients, atol=0)
        )

    def values(self, spherical_harmonics_summed):
        Fx = np.sum(np.real(self.xy_coefficients[self.N_M] * (
            spherical_harmonics_summed[self.N_M] * np.conj(spherical_harmonics_summed[self.Nr_Mr])
            - spherical_harmonics_summed[self.N_mM] * np.conj(spherical_harmonics_summed[self.Nr_mMr])
        )), axis=0)
        Fy = np.sum(np.imag(self.xy_coefficients[self.N_M] * (
            spherical_harmonics_summed[self.N_M] * np.conj(spherical_harmonics_summed[self.Nr_Mr])
            + spherical_harmonics_summed[self.N_mM] * np.conj(spherical_harmonics_summed[self.Nr_mMr])
        )), axis=0)
        Fz = np.sum(np.real(self.z_coefficients[self.N_M] * spherical_harmonics_summed[self.N_M] * np.conj(spherical_harmonics_summed[self.Nr_M])), axis=0)
        return np.stack([Fx, Fy, Fz])
