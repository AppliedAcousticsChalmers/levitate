"""A collection of levitation related mathematical implementations."""

import numpy as np
import functools
import collections
import textwrap
from . import materials


def requires(**requirements):
    # TODO: Document the list of possible choises, and compare the input to what is possible.
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        wrapped.requires = requirements
        return wrapped
    return wrapper


def algorithm(*output_names):
    def wrapper(func):
        named_outputs = collections.namedtuple(func.__name__, output_names)
        func.__doc__ = func.__doc__ or 'Parameters\n----------\n'
        @functools.wraps(func)
        def wrapped(*args, weights=None, **kwargs):
            output = func(*args, **kwargs)
            if weights is not None:
                weights = np.atleast_1d(weights)
                try:
                    for f in output:
                        f.weights = weights
                    output = named_outputs(*output)
                except TypeError:
                    output.weights = weights
                    output = named_outputs(output)
            return output
        wrapped.__doc__ = textwrap.dedent(wrapped.__doc__).rstrip('\n') + textwrap.dedent("""
            weights : numeric, optional
                Attaching weights to the algorithms for use in optimizations.

            Returns
            -------
            calc_values : func
                Function to calculate the divergence of the gorkov potential in cartesian coordinates.
            calc_jacobians : func
                Function to calculate the jacobian of the above, w.r.t the transducers.
            """)
        return wrapped
    return wrapper


@algorithm('calc_values', 'calc_jacobians')
def gorkov_divergence(array, radius_sphere=1e-3, sphere_material=materials.Styrofoam):
    """
    Create gorkov divergence calculation functions.

    Creates functions which calculates the gorkov divergence and the jacobian of
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
    V = 4 / 3 * np.pi * radius_sphere**3
    monopole_coefficient = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * array.medium.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * array.medium.compressibility * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * array.medium.rho

    @requires(pressure_orders_summed=2)
    def calc_values(summed_derivs):
        values = np.real(pressure_coefficient * np.conj(summed_derivs[0]) * summed_derivs[1:4])  # Pressure parts
        values -= np.real(gradient_coefficient * np.conj(summed_derivs[1]) * summed_derivs[[4, 7, 8]])  # Vx parts
        values -= np.real(gradient_coefficient * np.conj(summed_derivs[2]) * summed_derivs[[7, 5, 9]])  # Vy parts
        values -= np.real(gradient_coefficient * np.conj(summed_derivs[3]) * summed_derivs[[8, 9, 6]])  # Vz parts
        return values * 2

    @requires(pressure_orders_summed=2, pressure_orders_individual=2)
    def calc_jacobians(summed_derivs, individual_derivs):
        jacobians = pressure_coefficient * (np.conj(summed_derivs[0]) * individual_derivs[1:4] + np.conj(summed_derivs[1:4, None]) * individual_derivs[0])  # Pressure parts
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[1]) * individual_derivs[[4, 7, 8]] + np.conj(summed_derivs[[4, 7, 8], None]) * individual_derivs[1])  # Vx parts
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[2]) * individual_derivs[[7, 5, 9]] + np.conj(summed_derivs[[7, 5, 9], None]) * individual_derivs[2])  # Vy parts
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[3]) * individual_derivs[[8, 9, 6]] + np.conj(summed_derivs[[8, 9, 6], None]) * individual_derivs[3])  # Vz parts
        return jacobians * 2
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def gorkov_laplacian(array, radius_sphere=1e-3, sphere_material=materials.Styrofoam):
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
    V = 4 / 3 * np.pi * radius_sphere**3
    monopole_coefficient = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * array.medium.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * array.medium.compressibility * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * array.medium.rho

    @requires(pressure_orders_summed=3)
    def calc_values(summed_derivs):
        values = np.real(pressure_coefficient * (np.conj(summed_derivs[0]) * summed_derivs[[4, 5, 6]] + summed_derivs[[1, 2, 3]] * np.conj(summed_derivs[[1, 2, 3]])))
        values -= np.real(gradient_coefficient * (np.conj(summed_derivs[1]) * summed_derivs[[10, 15, 17]] + summed_derivs[[4, 7, 8]] * np.conj(summed_derivs[[4, 7, 8]])))
        values -= np.real(gradient_coefficient * (np.conj(summed_derivs[2]) * summed_derivs[[13, 11, 18]] + summed_derivs[[7, 5, 9]] * np.conj(summed_derivs[[7, 5, 9]])))
        values -= np.real(gradient_coefficient * (np.conj(summed_derivs[3]) * summed_derivs[[14, 16, 12]] + summed_derivs[[8, 9, 6]] * np.conj(summed_derivs[[8, 9, 6]])))
        return values * 2

    @requires(pressure_orders_summed=3, pressure_orders_individual=3)
    def calc_jacobians(summed_derivs, individual_derivs):
        jacobians = pressure_coefficient * (np.conj(summed_derivs[0]) * individual_derivs[[4, 5, 6]] + np.conj(summed_derivs[[4, 5, 6], None]) * individual_derivs[0] + 2 * np.conj(summed_derivs[[1, 2, 3], None]) * individual_derivs[[1, 2, 3]])
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[1]) * individual_derivs[[10, 15, 17]] + np.conj(summed_derivs[[10, 15, 17], None]) * individual_derivs[1] + 2 * np.conj(summed_derivs[[4, 7, 8], None]) * individual_derivs[[4, 7, 8]])
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[2]) * individual_derivs[[13, 11, 18]] + np.conj(summed_derivs[[13, 11, 18], None]) * individual_derivs[2] + 2 * np.conj(summed_derivs[[7, 5, 9], None]) * individual_derivs[[7, 5, 9]])
        jacobians -= gradient_coefficient * (np.conj(summed_derivs[3]) * individual_derivs[[14, 16, 12]] + np.conj(summed_derivs[[14, 16, 12], None]) * individual_derivs[3] + 2 * np.conj(summed_derivs[[8, 9, 6], None]) * individual_derivs[[8, 9, 6]])
        return jacobians * 2
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def second_order_force(array, radius_sphere=1e-3, sphere_material=materials.Styrofoam):
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
    f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
    f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    k_square = array.k**2
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * array.medium.compressibility

    # Including the j factors from the paper directly in the coefficients.
    psi_0 *= 1j
    psi_1 *= 1j

    @requires(pressure_orders_summed=2)
    def calc_values(summed_derivs):
        values = np.real(k_square * psi_0 * summed_derivs[0] * np.conj(summed_derivs[[1, 2, 3]]))
        values += np.real(k_square * psi_1 * summed_derivs[[1, 2, 3]] * np.conj(summed_derivs[0]))
        values += np.real(3 * psi_1 * summed_derivs[1] * np.conj(summed_derivs[[4, 7, 8]]))
        values += np.real(3 * psi_1 * summed_derivs[2] * np.conj(summed_derivs[[7, 5, 9]]))
        values += np.real(3 * psi_1 * summed_derivs[3] * np.conj(summed_derivs[[8, 9, 6]]))
        return values * force_coeff

    @requires(pressure_orders_summed=2, pressure_orders_individual=2)
    def calc_jacobians(summed_derivs, individual_derivs):
        jacobians = k_square * (psi_0 * individual_derivs[0] * np.conj(summed_derivs[[1, 2, 3], None]) + np.conj(psi_0) * np.conj(summed_derivs[0]) * individual_derivs[[1, 2, 3]])
        jacobians += k_square * (psi_1 * individual_derivs[[1, 2, 3]] * np.conj(summed_derivs[0]) + np.conj(psi_1) * np.conj(summed_derivs[[1, 2, 3], None]) * individual_derivs[0])
        jacobians += 3 * (psi_1 * individual_derivs[1] * np.conj(summed_derivs[[4, 7, 8], None]) + np.conj(psi_1) * np.conj(summed_derivs[1]) * individual_derivs[[4, 7, 8]])
        jacobians += 3 * (psi_1 * individual_derivs[2] * np.conj(summed_derivs[[7, 5, 9], None]) + np.conj(psi_1) * np.conj(summed_derivs[2]) * individual_derivs[[7, 5, 9]])
        jacobians += 3 * (psi_1 * individual_derivs[3] * np.conj(summed_derivs[[8, 9, 6], None]) + np.conj(psi_1) * np.conj(summed_derivs[3]) * individual_derivs[[8, 9, 6]])
        return jacobians * force_coeff
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def second_order_stiffness(array, radius_sphere=1e-3, sphere_material=materials.Styrofoam):
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
    f_1 = 1 - sphere_material.compressibility / array.medium.compressibility  # f_1 in H. Bruus 2012
    f_2 = 2 * (sphere_material.rho / array.medium.rho - 1) / (2 * sphere_material.rho / array.medium.rho + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    k_square = array.k**2
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * array.medium.compressibility

    # Including the j factors from the paper directly in the coefficients.
    psi_0 *= 1j
    psi_1 *= 1j

    @requires(pressure_orders_summed=3)
    def calc_values(summed_derivs):
        values = np.real(k_square * psi_0 * (summed_derivs[0] * np.conj(summed_derivs[[4, 5, 6]]) + summed_derivs[[1, 2, 3]] * np.conj(summed_derivs[[1, 2, 3]])))
        values += np.real(k_square * psi_1 * (summed_derivs[[4, 5, 6]] * np.conj(summed_derivs[0]) + summed_derivs[[1, 2, 3]] * np.conj(summed_derivs[[1, 2, 3]])))
        values += np.real(3 * psi_1 * (summed_derivs[1] * np.conj(summed_derivs[[10, 15, 17]]) + summed_derivs[[4, 7, 8]] * np.conj(summed_derivs[[4, 7, 8]])))
        values += np.real(3 * psi_1 * (summed_derivs[2] * np.conj(summed_derivs[[13, 11, 18]]) + summed_derivs[[7, 5, 9]] * np.conj(summed_derivs[[7, 5, 9]])))
        values += np.real(3 * psi_1 * (summed_derivs[3] * np.conj(summed_derivs[[14, 16, 12]]) + summed_derivs[[8, 9, 6]] * np.conj(summed_derivs[[8, 9, 6]])))
        return values * force_coeff

    @requires(pressure_orders_summed=3, pressure_orders_individual=3)
    def calc_jacobians(summed_derivs, individual_derivs):
        jacobians = k_square * (psi_0 * individual_derivs[0] * np.conj(summed_derivs[[4, 5, 6], None]) + np.conj(psi_0) * np.conj(summed_derivs[0]) * individual_derivs[[4, 5, 6]] + (psi_0 + np.conj(psi_0)) * np.conj(summed_derivs[[1, 2, 3], None]) * individual_derivs[[1, 2, 3]])
        jacobians += k_square * (psi_1 * individual_derivs[[4, 5, 6]] * np.conj(summed_derivs[0]) + np.conj(psi_1) * np.conj(summed_derivs[[4, 5, 6], None]) * individual_derivs[0] + (psi_1 + np.conj(psi_1)) * np.conj(summed_derivs[[1, 2, 3], None]) * individual_derivs[[1, 2, 3]])
        jacobians += 3 * (psi_1 * individual_derivs[1] * np.conj(summed_derivs[[10, 15, 17], None]) + np.conj(psi_1) * np.conj(summed_derivs[1]) * individual_derivs[[10, 15, 17]] + (psi_1 + np.conj(psi_1)) * np.conj(summed_derivs[[4, 7, 8], None]) * individual_derivs[[4, 7, 8]])
        jacobians += 3 * (psi_1 * individual_derivs[2] * np.conj(summed_derivs[[13, 11, 18], None]) + np.conj(psi_1) * np.conj(summed_derivs[2]) * individual_derivs[[13, 11, 18]] + (psi_1 + np.conj(psi_1)) * np.conj(summed_derivs[[7, 5, 9], None]) * individual_derivs[[7, 5, 9]])
        jacobians += 3 * (psi_1 * individual_derivs[3] * np.conj(summed_derivs[[14, 16, 12], None]) + np.conj(psi_1) * np.conj(summed_derivs[3]) * individual_derivs[[14, 16, 12]] + (psi_1 + np.conj(psi_1)) * np.conj(summed_derivs[[8, 9, 6], None]) * individual_derivs[[8, 9, 6]])
        return jacobians * force_coeff
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def pressure_squared_magnitude(array=None):
    """
    Create pressure squared magnitude calculation functions.

    Creates functions which calculates the squared pressure magnitude,
    and the corresponding jacobians.
    The main use of this is to use as a cost function.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array, optional.
    """
    @requires(pressure_orders_summed=0)
    def calc_values(summed_derivs):
        return np.real(summed_derivs[0] * np.conj(summed_derivs[0]))[None, ...]

    @requires(pressure_orders_summed=0, pressure_orders_individual=0)
    def calc_jacobians(summed_derivs, individual_derivs):
        return (2 * np.conj(summed_derivs[0]) * individual_derivs[0])[None, ...]
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def velocity_squared_magnitude(array):
    """
    Create velocity squared magnitude calculation functions.

    Creates functions which calculates the squared velocity magnitude,
    as a vector, and the corresponding jacobians.
    The main use of this is to use as a cost function.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array.
    """
    pre_grad_2_vel_squared = 1 / (array.medium.rho * array.omega)**2

    @requires(pressure_orders_summed=1)
    def calc_values(summed_derivs):
        return np.real(pre_grad_2_vel_squared * summed_derivs[1:4] * np.conj(summed_derivs[1:4]))

    @requires(pressure_orders_summed=1, pressure_orders_individual=1)
    def calc_jacobians(summed_derivs, individual_derivs):
        return 2 * pre_grad_2_vel_squared * np.conj(summed_derivs[1:4, None]) * individual_derivs[1:4]
    return calc_values, calc_jacobians


@algorithm('calc_values', 'calc_jacobians')
def vector_target(vector_calculator, target_vector=(0, 0, 0)):
    """
    Create a function which calculates the difference of a vector and a target vector.

    This is probably only usable as a cost function, where the goal is to minimize
    :math:`||(v - v_0)||^2_w`, i.e. the weighted square norm between a varying
    vector and a fixed vector. Note that the values in the weights will NOT be squared,
    i.e. have the inverse unit squares compared to the two vectors.

    Parameters
    ----------
    vector_calculator : (callable, callable)
        A tuple of callables where the first callable calculate the vector values,
        and the second callable calculate the jacobians of the vector.
        This fits with the output format of the other algorithms in this module.
    targert_vector : 3 element numeric, default (0, 0, 0)
        The target vector :math:`v_0` above.
    """
    target_vector = np.asarray(target_vector)
    # weights = np.asarray(weights)
    calc_vector_values, calc_vector_jacobians = vector_calculator

    @requires(**calc_vector_values.requires)
    def calc_values(summed_derivs):
        values = calc_vector_values(summed_derivs)
        values -= target_vector.reshape([-1] + (values.ndim - 1) * [1])
        # values *= weights.reshape([-1] + (values.ndim - 1) * [1])
        # return np.real(np.einsum('i...,i...', values, np.conj(values)))
        return np.real(values * np.conj(values))

    @requires(**calc_vector_jacobians.requires)
    def calc_jacobians(summed_derivs, individual_derivs):
        # raise NotImplementedError('Vector target not fully implemented!')
        values = calc_vector_values(summed_derivs)
        values -= target_vector.reshape([-1] + (values.ndim - 1) * [1])
        jacobians = calc_vector_jacobians(summed_derivs, individual_derivs)
        # return 2 * np.einsum('i, ij...,i...', weights**2, jacobians, values)
        return 2 * np.einsum('ij...,i...->ij', jacobians, values)
    return calc_values, calc_jacobians
