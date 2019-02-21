"""A collection of levitation related mathematical implementations."""

import numpy as np
import functools
from . import materials


def requires(**requirements):
    # This function must define an actual decorator which gets the defined cost function as input.
    # TODO: Document the list of possible choises, and compare the input to what is possible.
    def wrapper(func):
        # This function will be called at load time, i.e. when the levitate code is imported
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Save the weights and requirements as attribures in the function, since the cost_function_point needs it.
            return func(*args, **kwargs)
        wrapped.requires = requirements
        # cost_function_defer_initialization.requires = requirements
        return wrapped
    return wrapper


def gorkov_divergence(array, radius_sphere=1e-3, medium=materials.Air, sphere_material=materials.Styrofoam):
    V = 4 / 3 * np.pi * radius_sphere**3
    monopole_coefficient = 1 - sphere_material.compressibility / medium.compressibility  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (sphere_material.rho / medium.rho - 1) / (2 * sphere_material.rho / medium.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * medium.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * medium.compressibility * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * medium.rho

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


def gorkov_laplacian(array, radius_sphere=1e-3, medium=materials.Air, sphere_material=materials.Styrofoam):
    V = 4 / 3 * np.pi * radius_sphere**3
    monopole_coefficient = 1 - sphere_material.compressibility / medium.compressibility  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (sphere_material.rho / medium.rho - 1) / (2 * sphere_material.rho / medium.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * medium.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * medium.compressibility * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * medium.rho

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
