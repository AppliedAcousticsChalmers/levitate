"""A collection of levitation related mathematical implementations.

The fields is one of the most important parts of the package,
containing implementations of various ways to calculate levitate-related physical properties.
To simplify the management and manipulation of the implemented fields they are wrapped
in an additional abstraction layer.
The short version is that the classes implemented in the `~levitate.fields` module
will not return objects of the called class, but typically objects of `~levitate.field_wrappers.Field`.
These objects support algebraic operations, like `+`, `*`, and `abs`. The full description of
what the different operands do can be found in the documentation of `~levitate._field_wrappers`.

.. autosummary::
    :nosignatures:

    Pressure
    Velocity
    GorkovPotential
    GorkovGradient
    GorkovLaplacian
    RadiationForce
    RadiationForceStiffness
    RadiationForceCurl
    RadiationForceGradient
    SphericalHarmonicsForce

References
----------
.. [Gorkov] L. P. Gorkov, “On the Forces Acting on a Small Particle in an Acoustical Field in an Ideal Fluid”
            Soviet Physics Doklady, vol. 6, p. 773, Mar. 1962.

.. [Sapozhnikov] O. A. Sapozhnikov and M. R. Bailey, “Radiation force of an arbitrary acoustic beam on an elastic sphere in a fluid”
                 J Acoust Soc Am, vol. 133, no. 2, pp. 661–676, Feb. 2013.

"""

import numpy as np
from . import materials, utils
from ._field_wrappers import FieldImplementation
from ._algorithms_legacy import gorkov_potential, gorkov_gradient, gorkov_laplacian  # noqa: F401
from ._algorithms_legacy import second_order_force, second_order_stiffness, second_order_curl, second_order_force_gradient  # noqa: F401
from ._algorithms_legacy import pressure_squared_magnitude, velocity_squared_magnitude  # noqa: F401


class Pressure(FieldImplementation):
    """Complex sound pressure :math:`p`.

    Calculates the complex-valued sound pressure.

    """

    ndim = 0
    values_require = FieldImplementation.requirement(pressure_derivs_summed=0)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_individual=0)

    def values(self, pressure_derivs_summed):  # noqa: D102
        return pressure_derivs_summed[0]

    def jacobians(self, pressure_derivs_individual):  # noqa: D102
        return pressure_derivs_individual[0]


class Velocity(FieldImplementation):
    r"""Complex sound particle velocity :math:`v`.

    Calculates the sound particle velocity

    .. math:: v = {1 \over j\omega\rho} \nabla p

    from the relation :math:`\dot v = \rho \nabla p`
    applied for monofrequent sound fields.
    This is a vector value using a Cartesian coordinate system.

    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=1)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_individual=1)

    def __init__(self, array, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        self.pre_grad_2_vel = 1 / (1j * array.medium.rho * array.omega)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pre_grad_2_vel, other.pre_grad_2_vel, atol=0)
        )

    def values(self, pressure_derivs_summed):  # noqa: D102
        return self.pre_grad_2_vel * pressure_derivs_summed[1:4]

    def jacobians(self, pressure_derivs_individual):  # noqa: D102
        return self.pre_grad_2_vel * pressure_derivs_individual[1:4]


class GorkovPotential(FieldImplementation):
    r"""Gor'kov's potential :math:`U`.

    Calculates the Gor'kov potential [Gorkov]_

    .. math:: U = {V \over 4}(f_1 \kappa_0 |p|^2 - {3 \over 2} f_2 \rho_0 |v|^2)

    where

    .. math::
        f_1 = 1 - {\kappa_p \over \kappa_0}, \qquad
        f_2 = 2 {\rho_p - \rho_0 \over 2 \rho_p + \rho_0}

    and :math:`V` is the volume of the particle.
    Note that this is only a suitable measure for small particles, i.e. :math:`ka<<1`,
    where :math:`a` is the radius of the particle.
    """

    ndim = 0
    values_require = FieldImplementation.requirement(pressure_derivs_summed=1)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=1, pressure_derivs_individual=1)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.styrofoam, *args, **kwargs):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.
        radius_sphere : float, default 1e-3
            Radius of the spherical beads.
        sphere_material : Material
            The material of the sphere, default styrofoam.

        """
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

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = self.pressure_coefficient * np.real(pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[0]))
        values -= self.gradient_coefficient * np.real(pressure_derivs_summed[1:4] * np.conj(pressure_derivs_summed[1:4])).sum(axis=0)
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = self.pressure_coefficient * 2 * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[0])
        jacobians -= self.gradient_coefficient * 2 * (pressure_derivs_individual[1:4] * np.conj(pressure_derivs_summed[1:4, None])).sum(axis=0)
        return jacobians


class GorkovGradient(GorkovPotential):
    r"""Gradient of Gor'kov's potential, :math:`\nabla U`.

    Calculates the Cartesian spatial gradient of Gor'kov's potential,
    see `GorkovPotential` and [Gorkov]_. This is a vector value used to calculate the
    radiation force as

    .. math:: F = -\nabla U.

    Note that this value is not suitable for sound fields with strong
    traveling wave components. If this is the case, use the
    `RadiationForce` field instead.
    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = np.real(self.pressure_coefficient * np.conj(pressure_derivs_summed[0]) * pressure_derivs_summed[1:4])  # Pressure parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[1]) * pressure_derivs_summed[[4, 7, 8]])  # Vx parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[2]) * pressure_derivs_summed[[7, 5, 9]])  # Vy parts
        values -= np.real(self.gradient_coefficient * np.conj(pressure_derivs_summed[3]) * pressure_derivs_summed[[8, 9, 6]])  # Vz parts
        return values * 2

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[1:4] + np.conj(pressure_derivs_summed[1:4, None]) * pressure_derivs_individual[0])  # Pressure parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[4, 7, 8]] + np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[1])  # Vx parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[7, 5, 9]] + np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[2])  # Vy parts
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[8, 9, 6]] + np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[3])  # Vz parts
        return jacobians * 2


class GorkovLaplacian(GorkovPotential):
    r"""Laplacian of Gor'kov's potential, :math:`\nabla^2 U`.

    This calculates the Cartesian parts of the Laplacian of
    Gor'kov's potential, see `GorkovPotential` and [Gorkov]_. This is not
    really the Laplacian, since the components are not summed.
    The results can be seen as the local linear spring stiffness
    of the radiation force.

    Note that this value is not suitable for sound fields with strong
    traveling wave components. If this is the case, use the
    `RadiationForceStiffness` field instead.
    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = np.real(self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_summed[[4, 5, 6]] + pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[[1, 2, 3]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_summed[[10, 15, 17]] + pressure_derivs_summed[[4, 7, 8]] * np.conj(pressure_derivs_summed[[4, 7, 8]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_summed[[13, 11, 18]] + pressure_derivs_summed[[7, 5, 9]] * np.conj(pressure_derivs_summed[[7, 5, 9]])))
        values -= np.real(self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_summed[[14, 16, 12]] + pressure_derivs_summed[[8, 9, 6]] * np.conj(pressure_derivs_summed[[8, 9, 6]])))
        return values * 2

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = self.pressure_coefficient * (np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[4, 5, 6]] + np.conj(pressure_derivs_summed[[4, 5, 6], None]) * pressure_derivs_individual[0] + 2 * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[[1, 2, 3]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[10, 15, 17]] + np.conj(pressure_derivs_summed[[10, 15, 17], None]) * pressure_derivs_individual[1] + 2 * np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[[4, 7, 8]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[13, 11, 18]] + np.conj(pressure_derivs_summed[[13, 11, 18], None]) * pressure_derivs_individual[2] + 2 * np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[[7, 5, 9]])
        jacobians -= self.gradient_coefficient * (np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[14, 16, 12]] + np.conj(pressure_derivs_summed[[14, 16, 12], None]) * pressure_derivs_individual[3] + 2 * np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[[8, 9, 6]])
        return jacobians * 2


class RadiationForce(FieldImplementation):
    r"""Radiation force calculation for small beads in arbitrary sound fields.

    Calculates the radiation force on a small particle in a sound field which
    can have both strong standing wave components or strong traveling wave components.
    The force components :math:`q=x,y,z` are calculated as

    .. math::
        F_q &= -{\pi \over k^5}\kappa_0 \Re\left\{
        i k^2 \Psi_0 p {\partial p^* \over \partial q} + ik^2 \Psi_1 p^* {\partial p \over \partial q} \right.
        \\ &\quad +\left.
        3i \Psi_1 \left( {\partial p \over \partial x}{\partial^2 p^* \over \partial x\partial q}
        + {\partial p \over \partial y}{\partial^2 p^* \over \partial y\partial q}
        + {\partial p \over \partial z}{\partial^2 p^* \over \partial z\partial q}
        \right)\right\}

    where

    .. math::
        \Psi_0 &= -{2(ka)^6 \over 9} \left(f_1^2 + {f_2^2 \over 4} + f_1 f_2\right) -i{(ka)^3 \over 3} (2f_1+f_2) \\
        \Psi_1 &= - {(ka)^6 \over 18}f_2^2 + i{(ka)^3 \over 3} f_2 \\
        f_1 &= 1 - {\kappa_p \over \kappa_0}, \qquad
        f_2 = 2 {\rho_p - \rho_0 \over 2 \rho_p + \rho_0}

    This is more suitable than the Gor'kov formulation for use with progressive
    wave fiends, e.g. single sided arrays, see [Sapozhnikov]_.
    The actual implementation uses a further algebraic simplification of the above expresion.

    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def __init__(self, array, radius_sphere=1e-3, sphere_material=materials.styrofoam, *args, **kwargs):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.
        radius_sphere : float, default 1e-3
            Radius of the spherical beads.
        sphere_material : Material
            The material of the sphere, default styrofoam.

        """
        super().__init__(array, *args, **kwargs)
        f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
        f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

        ka = array.k * radius_sphere
        overall_coeff = -np.pi / array.k**5 * array.medium.compressibility
        self.pressure_coefficient = (ka**3 * 2 / 3 * f_1 - 2j / 9 * ka**6 * (f_1**2 + f_1 * f_2)) * array.k**2 * overall_coeff
        self.velocity_coefficient = (-ka**3 * f_2 - 1j / 6 * ka**6 * f_2**2) * overall_coeff

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.pressure_coefficient, other.pressure_coefficient, atol=0)
            and np.allclose(self.velocity_coefficient, other.velocity_coefficient, atol=0)
        )

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = np.real(self.pressure_coefficient * pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[[1, 2, 3]]))
        values += np.real(self.velocity_coefficient * pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[[4, 7, 8]]))
        values += np.real(self.velocity_coefficient * pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[[7, 5, 9]]))
        values += np.real(self.velocity_coefficient * pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[[8, 9, 6]]))
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = self.pressure_coefficient * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[[1, 2, 3], None]) + np.conj(self.pressure_coefficient) * np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[1, 2, 3]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[1] * np.conj(pressure_derivs_summed[[4, 7, 8], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[4, 7, 8]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[2] * np.conj(pressure_derivs_summed[[7, 5, 9], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[7, 5, 9]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[3] * np.conj(pressure_derivs_summed[[8, 9, 6], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[8, 9, 6]]
        return jacobians


class RadiationForceStiffness(RadiationForce):
    r"""Radiation force gradient for small beads in arbitrary sound fields.

    Calculates the non-mixed spatial derivatives of the radiation force,

    .. math::
        ({\partial F_x \over \partial x}, {\partial F_y \over \partial y}, {\partial F_z \over \partial z})

    where :math:`F` is the radiation force by [Sapozhnikov]_, see `RadiationForce`.

    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = np.real(self.pressure_coefficient * (pressure_derivs_summed[0] * np.conj(pressure_derivs_summed[[4, 5, 6]]) + pressure_derivs_summed[[1, 2, 3]] * np.conj(pressure_derivs_summed[[1, 2, 3]])))
        values += np.real(self.velocity_coefficient * (pressure_derivs_summed[1] * np.conj(pressure_derivs_summed[[10, 15, 17]]) + pressure_derivs_summed[[4, 7, 8]] * np.conj(pressure_derivs_summed[[4, 7, 8]])))
        values += np.real(self.velocity_coefficient * (pressure_derivs_summed[2] * np.conj(pressure_derivs_summed[[13, 11, 18]]) + pressure_derivs_summed[[7, 5, 9]] * np.conj(pressure_derivs_summed[[7, 5, 9]])))
        values += np.real(self.velocity_coefficient * (pressure_derivs_summed[3] * np.conj(pressure_derivs_summed[[14, 16, 12]]) + pressure_derivs_summed[[8, 9, 6]] * np.conj(pressure_derivs_summed[[8, 9, 6]])))
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = self.pressure_coefficient * pressure_derivs_individual[0] * np.conj(pressure_derivs_summed[[4, 5, 6], None]) + np.conj(self.pressure_coefficient) * np.conj(pressure_derivs_summed[0]) * pressure_derivs_individual[[4, 5, 6]] + (self.pressure_coefficient + np.conj(self.pressure_coefficient)) * np.conj(pressure_derivs_summed[[1, 2, 3], None]) * pressure_derivs_individual[[1, 2, 3]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[1] * np.conj(pressure_derivs_summed[[10, 15, 17], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[1]) * pressure_derivs_individual[[10, 15, 17]] + (self.velocity_coefficient + np.conj(self.velocity_coefficient)) * np.conj(pressure_derivs_summed[[4, 7, 8], None]) * pressure_derivs_individual[[4, 7, 8]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[2] * np.conj(pressure_derivs_summed[[13, 11, 18], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[2]) * pressure_derivs_individual[[13, 11, 18]] + (self.velocity_coefficient + np.conj(self.velocity_coefficient)) * np.conj(pressure_derivs_summed[[7, 5, 9], None]) * pressure_derivs_individual[[7, 5, 9]]
        jacobians += self.velocity_coefficient * pressure_derivs_individual[3] * np.conj(pressure_derivs_summed[[14, 16, 12], None]) + np.conj(self.velocity_coefficient) * np.conj(pressure_derivs_summed[3]) * pressure_derivs_individual[[14, 16, 12]] + (self.velocity_coefficient + np.conj(self.velocity_coefficient)) * np.conj(pressure_derivs_summed[[8, 9, 6], None]) * pressure_derivs_individual[[8, 9, 6]]
        return jacobians


class RadiationForceCurl(RadiationForce):
    r"""Curl or rotation of the radiation force.

    Calculates the curl of the radiation force field as

    .. math::
        ({\partial F_z \over \partial y} - {\partial F_y \over \partial z},
         {\partial F_x \over \partial z} - {\partial F_z \over \partial x},
         {\partial F_y \over \partial x} - {\partial F_x \over \partial y})

    where :math:`F` is the radiation force by [Sapozhnikov]_, see `RadiationForce`.

    """

    ndim = 1
    values_require = FieldImplementation.requirement(pressure_derivs_summed=2)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=2, pressure_derivs_individual=2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressure_coefficient = -2 * np.imag(self.pressure_coefficient)
        self.velocity_coefficient = -2 * np.imag(self.velocity_coefficient)

    def values(self, pressure_derivs_summed):  # noqa: D102
        values = self.pressure_coefficient * np.imag(pressure_derivs_summed[[2, 3, 1]] * np.conj(pressure_derivs_summed[[3, 1, 2]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[7, 8, 4]] * np.conj(pressure_derivs_summed[[8, 4, 7]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[5, 9, 7]] * np.conj(pressure_derivs_summed[[9, 7, 5]]))
        values += self.velocity_coefficient * np.imag(pressure_derivs_summed[[9, 6, 8]] * np.conj(pressure_derivs_summed[[6, 8, 9]]))
        return values

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        jacobians = 1j * self.pressure_coefficient * (np.conj(pressure_derivs_summed[[2, 3, 1], None]) * pressure_derivs_individual[[3, 1, 2]] - np.conj(pressure_derivs_summed[[3, 1, 2], None]) * pressure_derivs_individual[[2, 3, 1]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[7, 8, 4], None]) * pressure_derivs_individual[[8, 4, 7]] - np.conj(pressure_derivs_summed[[8, 4, 7], None]) * pressure_derivs_individual[[7, 8, 4]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[5, 9, 7], None]) * pressure_derivs_individual[[9, 7, 5]] - np.conj(pressure_derivs_summed[[9, 7, 5], None]) * pressure_derivs_individual[[5, 9, 7]])
        jacobians += 1j * self.velocity_coefficient * (np.conj(pressure_derivs_summed[[9, 6, 8], None]) * pressure_derivs_individual[[6, 8, 9]] - np.conj(pressure_derivs_summed[[6, 8, 9], None]) * pressure_derivs_individual[[9, 6, 8]])
        return jacobians


class RadiationForceGradient(RadiationForce):
    r"""Full matrix gradient of the radiation force.

    Calculates the full gradient matrix of the radiation force on a small spherical bead.
    Component :math:`(i,j)` in the matrix is :math:`{\partial F_i \over \partial q_j}`
    i.e. the first index is force the force components and the second index is for derivatives.
    This is based on analytical differentiation of the radiation force on small beads from
    [Sapozhnikov]_, see `RadiationForce`.

    """

    ndim = 2
    values_require = FieldImplementation.requirement(pressure_derivs_summed=3)
    jacobians_require = FieldImplementation.requirement(pressure_derivs_summed=3, pressure_derivs_individual=3)

    _0 = (0, None, None)
    _x = (1, None, None)
    _y = (2, None, None)
    _z = (3, None, None)

    _q = ([1, 2, 3], None)
    _w = (None, [1, 2, 3])
    _qw = ([[4, 7, 8], [7, 5, 9], [8, 9, 6]], )

    _xq = ([4, 7, 8], None)
    _xw = (None, [4, 7, 8])
    _yq = ([7, 5, 9], None)
    _yw = (None, [7, 5, 9])
    _zq = ([8, 9, 6], None)
    _zw = (None, [8, 9, 6])

    _xqw = ([[10, 13, 14], [13, 15, 19], [14, 19, 17]], )
    _yqw = ([[13, 15, 19], [15, 11, 16], [19, 16, 18]], )
    _zqw = ([[14, 19, 17], [19, 16, 18], [17, 18, 12]], )

    def values(self, pressure_derivs_summed):  # noqa: D102
        p = pressure_derivs_summed

        return np.real(
            self.pressure_coefficient * (p[self._0] * np.conj(p[self._qw]) + p[self._w] * np.conj(p[self._q]))
            + self.velocity_coefficient * (
                p[self._xw] * np.conj(p[self._xq]) + p[self._x] * np.conj(p[self._xqw])
                + p[self._yw] * np.conj(p[self._yq]) + p[self._y] * np.conj(p[self._yqw])
                + p[self._zw] * np.conj(p[self._zq]) + p[self._z] * np.conj(p[self._zqw])
            )
        )

    def jacobians(self, pressure_derivs_summed, pressure_derivs_individual):  # noqa: D102
        p = pressure_derivs_summed[:, None]
        dp = pressure_derivs_individual

        return (
            self.pressure_coefficient * (dp[self._0] * np.conj(p[self._qw]) + dp[self._w] * np.conj(p[self._q]))
            + np.conj(self.pressure_coefficient) * (np.conj(p[self._0]) * dp[self._qw] + np.conj(p[self._w]) * dp[self._q])
            + self.velocity_coefficient * (
                dp[self._xw] * np.conj(p[self._xq]) + dp[self._x] * np.conj(p[self._xqw])
                + dp[self._yw] * np.conj(p[self._yq]) + dp[self._y] * np.conj(p[self._yqw])
                + dp[self._zw] * np.conj(p[self._zq]) + dp[self._z] * np.conj(p[self._zqw])
            )
            + np.conj(self.velocity_coefficient) * (
                np.conj(p[self._xw]) * dp[self._xq] + np.conj(p[self._x]) * dp[self._xqw]
                + np.conj(p[self._yw]) * dp[self._yq] + np.conj(p[self._y]) * dp[self._yqw]
                + np.conj(p[self._zw]) * dp[self._zq] + np.conj(p[self._z]) * dp[self._zqw]
            )
        )


class SphericalHarmonicsForceDecomposition(FieldImplementation):
    r"""Radiation force decomposed in spherical harmonics.

    This is mostly intended for research purposes, when the radiation force
    decomposed in individual spherical harmonics bases is of interest.
    """

    ndim = 2

    def __init__(self, array, orders, radius_sphere=1e-3, sphere_material=materials.styrofoam, scattering_model='Hard sphere', *args, **kwargs):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.
        orders : int
            The number of force orders to include. Note that the sound field will
            be expanded at one order higher that the force order.
        radius_sphere : float, default 1e-3
            Radius of the spherical beads.
        sphere_material : Material
            The material of the sphere, default styrofoam.
        scattering_model:
            Chooses which scattering model to use. Currently `Hard sphere`, `Soft sphere`, and `Compressible sphere`
            are implemented.

        """
        super().__init__(array, *args, **kwargs)
        self.values_require = FieldImplementation.requirement(spherical_harmonics_summed=orders + 1)
        self.jacobians_require = FieldImplementation.requirement(spherical_harmonics_summed=orders + 1, spherical_harmonics_individual=orders + 1)

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

            relative_impedance = sphere_material.impedance / array.medium.impedance
            numerator = bessel_function * bessel_derivative_interior - relative_impedance * bessel_derivative * bessel_function_interior
            denominator = hankel_function * bessel_derivative_interior - relative_impedance * hankel_derivative * bessel_function_interior
            scattering_coefficient = - numerator / denominator
        else:
            raise ValueError("Unknown scattering model '{}'".format(scattering_model))

        scaling = array.medium.compressibility / (8 * array.k**2)
        self.xy_coefficients = np.zeros((orders + 1)**2, dtype=np.complex128)
        self.z_coefficients = np.zeros((orders + 1)**2, dtype=np.complex128)
        idx = 0
        for n in sph_idx.orders:
            psi = 1j * (1 + 2 * scattering_coefficient[n]) * (1 + 2 * np.conj(scattering_coefficient[n + 1])) - 1j
            denom = 1 / ((2 * n + 1) * (2 * n + 3))**0.5
            coeff = psi * scaling * denom
            for m in sph_idx.modes:
                self.xy_coefficients[idx] = ((n + m + 1) * (n + m + 2))**0.5 * coeff
                self.z_coefficients[idx] = -2 * ((n + m + 1) * (n - m + 1))**0.5 * coeff
                idx += 1

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.values_require['spherical_harmonics_summed'] == other.values_require['spherical_harmonics_summed']
            and np.allclose(self.xy_coefficients, other.xy_coefficients, atol=0)
            and np.allclose(self.z_coefficients, other.z_coefficients, atol=0)
        )

    def values(self, spherical_harmonics_summed):  # noqa: D102
        # Reshape coefficients to allow multiple receiver positions
        xy_coefs = self.xy_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_summed.ndim - 1))
        z_coefs = self.z_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_summed.ndim - 1))
        S = spherical_harmonics_summed

        Fxy = xy_coefs * S[self.N_M] * np.conj(S[self.Nr_Mr]) - np.conj(xy_coefs) * np.conj(S[self.N_mM]) * S[self.Nr_mMr]
        Fx = np.real(Fxy)
        Fy = np.imag(Fxy)
        Fz = np.real(z_coefs * S[self.N_M] * np.conj(S[self.Nr_M]))

        return np.stack([Fx, Fy, Fz])

    def jacobians(self, spherical_harmonics_summed, spherical_harmonics_individual):  # noqa: D102
        xy_coefs = self.xy_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_individual.ndim - 1))
        z_coefs = self.z_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_individual.ndim - 1))

        S = spherical_harmonics_summed[:, None]
        dS = spherical_harmonics_individual

        # Since y is the imaginary part of the expression, we will get a sign change for the parts which is conjugated be the derivatives.
        dFxy_same = xy_coefs * dS[self.N_M] * np.conj(S[self.Nr_Mr]) - np.conj(xy_coefs) * np.conj(S[self.N_mM]) * dS[self.Nr_mMr]
        dFxy_conj = np.conj(xy_coefs) * np.conj(S[self.N_M]) * dS[self.Nr_Mr] - xy_coefs * dS[self.N_mM] * np.conj(S[self.Nr_mMr])
        dFx = dFxy_same + dFxy_conj
        dFy = -1j * (dFxy_same - dFxy_conj)
        dFz = z_coefs * dS[self.N_M] * np.conj(S[self.Nr_M]) + np.conj(z_coefs) * np.conj(S[self.N_M]) * dS[self.Nr_M]

        return np.stack([dFx, dFy, dFz], axis=0)


class SphericalHarmonicsForce(SphericalHarmonicsForceDecomposition):
    r"""Spherical harmonics based radiation force.

    Expands the local sound field in spherical harmonics and calculates
    the radiation force in the spherical harmonics domain.
    The expansion coefficients are calculated using superposition
    of the translated expansions of the transducer radiation patterns.
    The radiation force is calculated using a similar derivation as [Sapozhnikov]_,
    but without any plane wave decomposition.

    Todo
    ----
    This function does not yet support jacobians, and cannot be used as a cost function.

    """

    ndim = 1

    def values(self, *args, **kwargs):  # noqa: D102
        return np.sum(super().values(*args, **kwargs), axis=1)

    def jacobians(self, *args, **kwargs):  # noqa: D102
        return np.sum(super().jacobians(*args, **kwargs), axis=1)


class SphericalHarmonicsForceGradientDecomposition(SphericalHarmonicsForceDecomposition):
    """Spatial gradient of spherical harmonics force decomposition.

    Takes the spatial gradient in Cartesian coordinates of each order and mode of
    the radiation force calculated from a spherical harmonics expansion.
    See `SphericalHarmonicsForce` for details on algorithms and parameters.

    """

    ndim = 3

    def __init__(self, array, orders, *args, **kwargs):
        super().__init__(array, orders, *args, **kwargs)
        self.values_require = FieldImplementation.requirement(spherical_harmonics_summed=orders + 1, spherical_harmonics_gradient_summed=orders + 1)
        self.jacobians_require = FieldImplementation.requirement(
            spherical_harmonics_summed=orders + 1, spherical_harmonics_gradient_summed=orders + 1,
            spherical_harmonics_individual=orders + 1, spherical_harmonics_gradient_individual=orders + 1)

    def values(self, spherical_harmonics_summed, spherical_harmonics_gradient_summed):  # noqa: D102
        # Reshape coefficients to allow multiple receiver positions
        xy_coefs = self.xy_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_summed.ndim - 1))
        z_coefs = self.z_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_summed.ndim - 1))
        S = spherical_harmonics_summed
        DS = spherical_harmonics_gradient_summed

        DFxy = (
            xy_coefs * (DS[:, self.N_M] * np.conj(S[self.Nr_Mr]) + S[self.N_M] * np.conj(DS[:, self.Nr_Mr]))
            - np.conj(xy_coefs) * (np.conj(S[self.N_mM]) * DS[:, self.Nr_mMr] + S[self.Nr_mMr] * np.conj(DS[:, self.N_mM]))
        )
        DFx = np.real(DFxy)
        DFy = np.imag(DFxy)
        DFz = np.real(z_coefs * DS[:, self.N_M] * np.conj(S[self.Nr_M]) + np.conj(z_coefs) * np.conj(S[self.N_M]) * DS[:, self.Nr_M])

        return np.stack([DFx, DFy, DFz], axis=0)

    def jacobians(self, spherical_harmonics_summed, spherical_harmonics_individual,
                  spherical_harmonics_gradient_summed, spherical_harmonics_gradient_individual):  # noqa: D102
        xy_coefs = self.xy_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_individual.ndim - 1))
        z_coefs = self.z_coefficients[self.N_M].reshape((-1,) + (1,) * (spherical_harmonics_individual.ndim - 1))

        S = spherical_harmonics_summed[:, None]
        DS = spherical_harmonics_gradient_summed[:, :, None]
        dS = spherical_harmonics_individual
        dDS = spherical_harmonics_gradient_individual

        dDFxy_same = (
            xy_coefs * (dDS[:, self.N_M] * np.conj(S[self.Nr_Mr]) + dS[self.N_M] * np.conj(DS[:, self.Nr_Mr]))
            - np.conj(xy_coefs) * (np.conj(S[self.N_mM]) * dDS[:, self.Nr_mMr] + dS[self.Nr_mMr] * np.conj(DS[:, self.N_mM]))
        )
        dDFxy_conj = (
            np.conj(xy_coefs) * (np.conj(DS[:, self.N_M]) * dS[self.Nr_Mr] + np.conj(S[self.N_M]) * dDS[:, self.Nr_Mr])
            - xy_coefs * (dS[self.N_mM] * np.conj(DS[:, self.Nr_mMr]) + np.conj(S[self.Nr_mMr]) * dDS[:, self.N_mM])
        )
        dDFx = dDFxy_same + dDFxy_conj
        dDFy = -1j * (dDFxy_same - dDFxy_conj)
        dDFz = (
            z_coefs * (dDS[:, self.N_M] * np.conj(S[self.Nr_M]) + dS[self.N_M] * np.conj(DS[:, self.Nr_M]))
            + np.conj(z_coefs) * (np.conj(DS[:, self.N_M]) * dS[self.Nr_M] + np.conj(S[self.N_M]) * dDS[:, self.Nr_M])
        )

        return np.stack([dDFx, dDFy, dDFz], axis=0)


class SphericalHarmonicsForceGradient(SphericalHarmonicsForceGradientDecomposition):
    """Spatial gradient of the total spherical radiation force.

    The three Cartesian derivatives of the radiation force on a spherical object,
    calculated using spherical harmonics expansion of the sound field.
    See `SphericalHarmonicsForce` for details on the parameters.

    """

    ndim = 2

    def values(self, *args, **kwargs):  # noqa: D102
        return np.sum(super().values(*args, **kwargs), axis=2)

    def jacobians(self, *args, **kwargs):  # noqa: D102
        return np.sum(super().jacobians(*args, **kwargs), axis=2)


class SphericalHarmonicsExpansion(FieldImplementation):
    """Spherical harmonics expansion coefficients of the sound pressure.

    The expansion coefficients up to a certain order, where the complex
    amplitudes of the transducers will be accounted for.

    """

    ndim = 1

    def __eq__(self, other):
        return(
            super().__eq__(other)
            and self.max_idx == other.max_idx
        )

    def __init__(self, array, orders, *args, **kwargs):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.
        orders : int
            The number of expansion orders to include.

        """
        super().__init__(array, *args, **kwargs)
        self.max_idx = len(utils.SphericalHarmonicsIndexer(orders))
        self.values_require = FieldImplementation.requirement(spherical_harmonics_summed=orders)
        self.jacobians_require = FieldImplementation.requirement(spherical_harmonics_individual=orders)

    def values(self, spherical_harmonics_summed):  # noqa: D102
        return spherical_harmonics_summed[:self.max_idx]

    def jacobians(self, spherical_harmonics_individual):  # noqa: D102
        return spherical_harmonics_individual[:self.max_idx]


class SphericalHarmonicsExpansionGradient(SphericalHarmonicsExpansion):
    """Spatial gradient of spherical harmonics expansion coefficients.

    Gives the Cartesian gradient of the expansion coefficient with respect
    to the expansion center.
    See `SphericalHarmonicsExpansion` for documentation of parameters.
    """

    ndim = 2

    def __init__(self, array, orders, *args, **kwargs):
        super().__init__(array, orders, *args, **kwargs)
        self.values_require = FieldImplementation.requirement(spherical_harmonics_gradient_summed=orders)
        self.jacobians_require = FieldImplementation.requirement(spherical_harmonics_gradient_individual=orders)

    def values(self, spherical_harmonics_gradient_summed):  # noqa: D102
        return spherical_harmonics_gradient_summed[:, :self.max_idx]

    def jacobians(self, spherical_harmonics_gradient_individual):  # noqa: D102
        return spherical_harmonics_gradient_individual[:, :self.max_idx]
