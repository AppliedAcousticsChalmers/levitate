import numpy as np
from scipy.optimize import minimize, basinhopping
import logging

from . import models

logger = logging.getLogger(__name__)


class Optimizer:

    def __init__(self, array=None):
        if array is None:
            self.array = models.TransducerArray()
        else:
            self.array = array
        self.objective_list = []
        self.basinhopping = False
        self.variable_amplitudes = False

    def __call__(self):
        # Initialize all parts of the objective function
        # Assemble objective function and jacobian function
        # Start optimization
        # Basin hopping? Check number of iterations?
        # Return phases?
        self.initialize()

        # Set starting points 
        if self.variable_amplitudes:
            start = np.concatenate((self.array.phases, self.array.amplitudes))
        else:
            start = self.array.phases

        # Set bounds for L-BFGS-B
        bounds = [(None, None)] * self.array.num_transducers
        if self.variable_amplitudes:
            bounds += [(1e-3, 1)] * self.array.num_transducers
        # TODO: The method selection should be configureable
        args = {'jac': self.jacobian,
            # 'method': 'BFGS', 'options': {'return_all': False, 'gtol': 5e-5, 'norm': 2, 'disp': True}}
            'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'gtol': 1e-9, 'ftol': 1e-15}}
        if self.basinhopping:
            take_step = RadndomDisplacer(self.array.num_transducers, self.variable_amplitudes, stepsize=0.1)
            self.result = basinhopping(self.function, start, T=1e-7, take_step=None, minimizer_kwargs=args, disp=True)
        else:
            self.result = minimize(self.function, start, callback=None, **args)

        if self.variable_amplitudes:
            self.phases = self.result.x[:self.array.num_transducers]
            self.amplitudes = self.result.x[self.array.num_transducers:]
        else:
            self.phases = self.result.x
            self.amplitudes = self.array.amplitudes

        # self.result = minimize(self.function, self.array.phases, jac=self.jacobian, callback=None,
        # method='L-BFGS-B', bounds=[(-3*np.pi, 3*np.pi)]*self.array.num_transducers, options={'gtol': 1e-7, 'ftol': 1e-12})
        # method='BFGS', options={'return_all': True, 'gtol': 1e-5, 'norm': 2})

    def function(self, phases_amplitudes):
        value = 0
        for objective, weight in self.objective_list:
            value += objective.function(phases_amplitudes) * weight
        return value

    def jacobian(self, phases_amplitudes):
        value = np.zeros(phases_amplitudes.size)
        for objective, weight in self.objective_list:
            value += objective.jacobian(phases_amplitudes) * weight
        return value

    def add_objective(self, objective, weight):
        # TODO: Check that the object has the required methods
        if type(objective) is type:
            # Handed a class, initialise an instance with default parameters
            objective = objective(self.array)
        self.objective_list.append((objective, weight))

    def initialize(self):
        for objective, weight in self.objective_list:
            objective.initialize()


class RadndomDisplacer:
    def __init__(self, num_transducers, variable_amplitude=False, stepsize=0.05):
        self.stepsize = stepsize
        self.num_transducers = num_transducers
        self.variable_amplitude = variable_amplitude

    def __call__(self, x):
        if self.variable_amplitude:
            x[:self.num_transducers] += np.random.uniform(-np.pi * self.stepsize, np.pi * self.stepsize, self.num_transducers)
            x[:self.num_transducers] = np.mod(x[:self.num_transducers] + np.pi, 2 * np.pi) - np.pi  # Don't step out of bounds, instead wrap the phase
            x[self.num_transducers:] += np.random.uniform(-self.stepsize, self.stepsize, self.num_transducers)
            x[self.num_transducers:] = np.clip(x[self.num_transducers:], 1e-3, 1)  # Don't step out of bounds!
        else:
            x += np.random.uniform(-np.pi * self.stepsize, np.pi * self.stepsize, self.num_transducers)
        return x


class CostFunction:

    def initialize(self):
        pass

    def function(self, phases_amplitudes):
        raise NotImplementedError('Required method `function` not implemented in {}'.format(self.__class__.__name__))

    def jacobian(self, phases_amplitudes):
        raise NotImplementedError('Required method `jacobian` not implemented in {}'.format(self.__class__.__name__))


class AmplitudeLimiting(CostFunction):

    def __init__(self, array, bounds=(1e-2, 0.99)):
        self.array = array
        self.lower_bound = np.asarray(bounds).min()
        self.upper_bound = np.asarray(bounds).max()
        self.coefficient = 10
        self.order = 4

    def function(self, phases_amplitudes):
        if np.iscomplexobj(phases_amplitudes):
            amplitudes = np.abs(phases_amplitudes)
        elif phases_amplitudes.size == self.array.num_transducers:
            return 0
            # TODO: Warn that this class is not ment to be used without variable amplitudes
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]

        under_idx = amplitudes < self.lower_bound
        over_idx = amplitudes > self.upper_bound
        under = self.coefficient * (self.lower_bound - amplitudes[under_idx])
        over = self.coefficient * (amplitudes[over_idx] - self.upper_bound)

        under = under**self.order
        over = over**self.order
        # assert over.sum() < 256e3
        return under.sum() + over.sum()

    def jacobian(self, phases_amplitudes):
        if np.iscomplexobj(phases_amplitudes):
            raise NotImplementedError('Jacobian not implemented for complex inputs!')
        elif phases_amplitudes.size == self.array.num_transducers:
            return np.zeros(self.array.num_transducers)
            # TODO: Warn that this class is not ment to be used without variable amplitudes
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]

        derivatives = np.zeros(self.array.num_transducers)
        under_idx = amplitudes < self.lower_bound
        over_idx = amplitudes > self.upper_bound
        under = self.coefficient * (self.lower_bound - amplitudes[under_idx])
        over = self.coefficient * (amplitudes[over_idx] - self.upper_bound)

        under = under**(self.order - 1) * self.order
        over = over**(self.order - 1) * self.order
        derivatives[under_idx] = under
        derivatives[over_idx] = over

        return np.concatenate((np.zeros(self.array.num_transducers), derivatives))


class PressurePoint(CostFunction):
    '''
    A class used to minimize pressure in a small region.
    The objective funciton is to minimize both pressure and pressure gradient.
    '''

    def __init__(self, array, focus=None, order=None, radius=0):
        self.array = array
        if focus is None:
            self.focus = array.focus_point
        else:
            self.focus = focus
        self.order = order
        self.radius = radius
        self.gradient_weights = (1, 1, 1)

    def initialize(self):
        self.spatial_derivatives = self.array.spatial_derivatives(self.focus)
        # self.gradient_normalization = (self.array.k * norm(self.focus - np.mean(self.array.transducer_positions, axis=0)))**2
        self.gradient_normalization = self.array.k**2
        # TODO: Different radius of the different orders?
        # TODO: Different weights for the different radius?
        # TODO: Lebedev grids?
        # TODO: Fibonacci spiral on sphere?
        if self.order is not None and self.radius > 0:
            extra_points = set()
            if self.order >= 1:
                extra_points = extra_points.union(set(permutations([1, 0, 0])))
                extra_points = extra_points.union(set(permutations([-1, 0, 0])))
            if self.order >= 2:
                extra_points = extra_points.union(set(permutations([1, 1, 1])))
                extra_points = extra_points.union(set(permutations([1, 1, -1])))
                extra_points = extra_points.union(set(permutations([1, -1, -1])))
                extra_points = extra_points.union(set(permutations([-1, -1, -1])))
            self.extra_points = []
            for point in extra_points:
                diff = np.array(point)
                focus = diff / norm(diff) * self.radius + self.focus
                p_point = pressure_point(self.array, focus=focus, order=None)
                p_point.gradient_weights = self.gradient_weights
                p_point.initialize()
                self.extra_points.append(p_point)

            # Loop through orders from 1 and up
            # Add new grid points each order to the list of poins
            # Scale with radius and shift with focus
            # Create a bunch of new pressure_point and initialize them
            # Add the new pressure_points along with weights to a list that
            # should be evaluated in the function and jacobian.

    def function(self, phases_amplitudes):
        '''
        Calculates the squared pressure + squared sum of pressure gradient
        Input: 
            phases_amplitudes: array
                Specify either phases for all transducers or phases and amplitudes for all transducers.
                If the length matches the number of transducers it should be phase values, the amplitudes are taken from the array.
                If the length is twice the number of transducers the first half is used as phases, and the second half as amplitudes.

        '''
        if np.iscomplexobj(phases_amplitudes):
            phases = np.angle(phases_amplitudes)
            amplitudes = np.abs(phases_amplitudes)
        elif phases_amplitudes.size == self.array.num_transducers:
            phases = phases_amplitudes
            amplitudes = self.array.amplitudes
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            phases = phases_amplitudes.ravel()[:self.array.num_transducers]
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]

        complex_coeff = amplitudes * np.exp(1j * phases)
        p_part = np.abs((complex_coeff * self.spatial_derivatives['']).sum())**2
        x_part = np.abs((complex_coeff * self.spatial_derivatives['x']).sum())**2
        y_part = np.abs((complex_coeff * self.spatial_derivatives['y']).sum())**2
        z_part = np.abs((complex_coeff * self.spatial_derivatives['z']).sum())**2

        wx, wy, wz = self.gradient_weights
        value = p_part + (wx * x_part + wy * y_part + wz * z_part) / self.gradient_normalization

        if self.order is not None:
            for p_point in self.extra_points:
                value += 0.1 * p_point.function(phases_amplitudes)
        return value

    def jacobian(self, phases_amplitudes):
        '''
        Calculates the jacobian of squared pressure + squared sum of pressure gradient
        Input:
            phases_amplitudes: array
                Specify either phases for all transducers or phases and amplitudes for all transducers.
                If the length matches the number of transducers it should be phase values, the amplitudes are taken from the array.
                If the length is twice the number of transducers the first half is used as phases, and the second half as amplitudes.

        '''
        if np.iscomplexobj(phases_amplitudes):
            raise NotImplementedError('Jacobian not implemented for complex inputs!')
            # Not sure that is is possible at all since the values in the jacobian needs to correspond to the values in the input.
            # Therefore it is not possible to just return the separate phase and amplitude jacobian.
            # Seen as a function of a single complex variable, the objective function is not analytic, and the derivative does not exist
        elif phases_amplitudes.size == self.array.num_transducers:
            phases = phases_amplitudes
            amplitudes = self.array.amplitudes
            return_amplitudes = False
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            phases = phases_amplitudes.ravel()[:self.array.num_transducers]
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]
            return_amplitudes = True

        complex_coeff = amplitudes * np.exp(1j * phases)
        phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            phased_derivatives[key] = complex_coeff * value
            total_derivatives[key] = np.sum(phased_derivatives[key])
        # TODO: Check that this actually works for the gradient parts
        p_part = 2 * total_derivatives[''] * np.conj(phased_derivatives[''])
        x_part = 2 * total_derivatives['x'] * np.conj(phased_derivatives['x'])
        y_part = 2 * total_derivatives['y'] * np.conj(phased_derivatives['y'])
        z_part = 2 * total_derivatives['z'] * np.conj(phased_derivatives['z'])

        wx, wy, wz = self.gradient_weights
        derivatives = p_part + (wx * x_part + wy * y_part + wz * z_part) / self.gradient_normalization
        if return_amplitudes:
            derivatives = np.concatenate((derivatives.imag, derivatives.real / amplitudes))
        else:
            derivatives = derivatives.imag

        if self.order is not None:
            for p_point in self.extra_points:
                derivatives += 0.1 * p_point.jacobian(phases_amplitudes)
        return derivatives


class GorkovLaplacian(CostFunction):

    def __init__(self, array, focus=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
        # Table at https://spiremt.com/support/SoundSpeedTable states longitudinal
        # wavespeed of Polystyrene at 2350 (Asier: 2400)
        # The density of Polystyrene is 1040 (Google), but styrofoam is ~30 (Asier: 25)
        self.array = array
        if focus is None:
            self.focus = array.focus_point
        else:
            self.focus = focus
        # TODO: raise error if the focus point is not set

        self.c_sphere = c_sphere
        self.rho_sphere = rho_sphere
        self.radius_sphere = radius_sphere
        # TODO: Make these two parameters that are updated if the array changes
        self.rho_air = array.rho
        self.c_air = array.c

        self.pressure_weight = 1
        self.gradient_weights = (1, 1, 1)

    def initialize(self):
        '''
        Prepare for running optimization
        '''
        V = 4 / 3 * np.pi * self.radius_sphere**3
        compressibility_air = 1 / (self.rho_air * self.c_air**2)
        compressibility_sphere = 1 / (self.rho_sphere * self.c_sphere**2)
        monopole_coefficient = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
        dipole_coefficient = 2 * (self.rho_sphere / self.rho_air - 1) / (2 * self.rho_sphere / self.rho_air + 1)   # f_2 in H. Bruus 2012
        preToVel = 1/(2*np.pi*self.array.freq * self.rho_air)  # Converting velocity to pressure gradient using equation of motion

        self.pressure_coefficient = V / 2 * compressibility_air * monopole_coefficient
        self.gradient_coefficient = V * 3 / 4 * dipole_coefficient * preToVel**2 * self.rho_air

        # self.pressure_coefficient = V / 2 * (1/(self.rho_air * self.c_air**2) - 1/(self.rho_sphere * self.c_sphere**2))
        # c1 = 3 / 2 * V / (2 * np.pi * self.array.freq)**2 / self.rho_air
        # self.gradient_coefficient = c1 * (self.rho_sphere - self.rho_air) / (2 * self.rho_sphere + self.rho_air)

        self.objective_evals = 0
        self.jacobian_evals = 0
        self.total_evals = 0
        self.previous_phase = self.array.phases
        self.phase_list = []
        self.spatial_derivatives = self.array.spatial_derivatives(self.focus)

    def function(self, phases_amplitudes):
        '''
        Calculates the squared pressure - the Gor'kov laplacian
        Input: 
            phases_amplitudes: array
                Specify either phases for all transducers or phases and amplitudes for all transducers.
                If the length matches the number of transducers it should be phase values, the amplitudes are taken from the array.
                If the length is twice the number of transducers the first half is used as phases, and the second half as amplitudes.

        '''
        if np.iscomplexobj(phases_amplitudes):
            phases = np.angle(phases_amplitudes)
            amplitudes = np.abs(phases_amplitudes)
        elif phases_amplitudes.size == self.array.num_transducers:
            phases = phases_amplitudes
            amplitudes = self.array.amplitudes
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            phases = phases_amplitudes.ravel()[:self.array.num_transducers]
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]
        self.objective_evals += 1
        self.total_evals += 1
        # logger.info('Objective function call {}:\tPhase difference: RMS = {:.4e}\t Max = {:.4e}'.format(self.objective_evals, np.sqrt(np.mean((phases - self.previous_phase)**2)), np.max(np.abs(phases - self.previous_phase))))
        self.previous_phase = phases

        complex_coeff = amplitudes * np.exp(1j * phases)
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            total_derivatives[key] = np.sum(complex_coeff * value)

        p_part = total_derivatives['']
        x_part = self.laplacian_axis('x', total_derivatives)
        y_part = self.laplacian_axis('y', total_derivatives)
        z_part = self.laplacian_axis('z', total_derivatives)

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        value = wp * np.abs(p_part)**2 - wx * x_part - wy * y_part - wz * z_part

        # logger.debug('\tPressure contribution: {:.6e}'.format(wp * np.abs(p_part)**2))
        # logger.debug('\tUxx contribution: {:.6e}'.format(-wx * x_part))
        # logger.debug('\tUyy contribution: {:.6e}'.format(-wy * y_part))
        # logger.debug('\tUzz contribution: {:.6e}'.format(-wz * z_part))

        return value

    def jacobian(self, phases_amplitudes):
        '''
        Calculates the jacobian of squared pressure - the Gor'kov laplacian
        Input: 
            phases_amplitudes: array
                Specify either phases for all transducers or phases and amplitudes for all transducers.
                If the length matches the number of transducers it should be phase values, the amplitudes are taken from the array.
                If the length is twice the number of transducers the first half is used as phases, and the second half as amplitudes.

        '''
        if np.iscomplexobj(phases_amplitudes):
            raise NotImplementedError('Jacobian not implemented for complex inputs!')
            # Not sure that is is possible at all since the values in the jacobian needs to correspond to the values in the input.
            # Therefore it is not possible to just return the separate phase and amplitude jacobian.
            # Seen as a function of a single complex variable, the objective function is not analytic, and the derivative does not exist
        elif phases_amplitudes.size == self.array.num_transducers:
            phases = phases_amplitudes
            amplitudes = self.array.amplitudes
            return_amplitudes = False
        elif phases_amplitudes.size == 2 * self.array.num_transducers:
            phases = phases_amplitudes.ravel()[:self.array.num_transducers]
            amplitudes = phases_amplitudes.ravel()[self.array.num_transducers:]
            return_amplitudes = True
        self.jacobian_evals += 1
        self.total_evals += 1
        # logger.info('Jacobian function call {}:\tPhase difference: RMS = {:.4e}\t Max = {:.4e}'.format(self.jacobian_evals, np.sqrt(np.mean((phases - self.previous_phase)**2)), np.max(np.abs(phases - self.previous_phase))))
        self.previous_phase = phases

        complex_coeff = amplitudes * np.exp(1j * phases)
        phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            phased_derivatives[key] = complex_coeff * value
            total_derivatives[key] = np.sum(phased_derivatives[key])

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        p_part = 2 * total_derivatives[''] * np.conj(phased_derivatives[''])
        # p_part = self.phase_derivative('', '', total_derivatives, phased_derivatives)
        x_part = self.laplacian_axis_derivative('x', total_derivatives, phased_derivatives)
        y_part = self.laplacian_axis_derivative('y', total_derivatives, phased_derivatives)
        z_part = self.laplacian_axis_derivative('z', total_derivatives, phased_derivatives)

        derivatives = wp * p_part - wx * x_part - wy * y_part - wz * z_part
        # return derivatives
        if return_amplitudes:
            return np.concatenate((derivatives.imag, derivatives.real / amplitudes))
        else:
            return derivatives.imag

    def laplacian_axis(self, axis, total_derivatives):
        '''
        Calculates a part of the Gor'kov Laplacian, along the specified axis

        'axis': a single character 'x', 'y', or 'z'
        '''

        p = total_derivatives['']
        pa = total_derivatives[axis]
        paa = total_derivatives[2 * axis]

        px = total_derivatives['x']
        py = total_derivatives['y']
        pz = total_derivatives['z']

        pax = total_derivatives['x' + axis]
        pay = total_derivatives[''.join(sorted(axis + 'y'))]  # ''.join(sorted(axis + 'y')) always give xy, yy, yz
        paz = total_derivatives[axis + 'z']

        paax = total_derivatives[2 * axis + 'x']
        paay = total_derivatives[2 * axis + 'y']
        paaz = total_derivatives[2 * axis + 'z']

        # Calculate individual parts (old method)
        # p_part = 2 * (self.complex_dot(paa , p ) + self.complex_dot(pa , pa ))
        # x_part = 2 * (self.complex_dot(paax, px) + self.complex_dot(pax, pax))
        # y_part = 2 * (self.complex_dot(paay, py) + self.complex_dot(pay, pay))
        # z_part = 2 * (self.complex_dot(paaz, pz) + self.complex_dot(paz, paz))

        # Calculate individual parts (new method)
        p_part = 2 * (paa * np.conj(p) + pa * np.conj(pa)).real
        x_part = 2 * (paax * np.conj(px) + pax * np.conj(pax)).real
        y_part = 2 * (paay * np.conj(py) + pay * np.conj(pay)).real
        z_part = 2 * (paaz * np.conj(pz) + paz * np.conj(paz)).real

        # logger.debug("\tGor'Kov Laplacian along {}-axis:".format(axis))
        # logger.debug('\t\tp contribution is: {:.6e}'.format(self.pressure_coefficient * p_part))
        # logger.debug('\t\tx contribution is: {:.6e}'.format(- self.gradient_coefficient * x_part))
        # logger.debug('\t\ty contribution is: {:.6e}'.format(- self.gradient_coefficient * y_part))
        # logger.debug('\t\tz contribution is: {:.6e}'.format(- self.gradient_coefficient * z_part))

        return self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)

    def laplacian_axis_derivative(self, axis, total_derivatives, phased_derivatives):
        '''
        Calculates the derivative of the Gor'kov laplacian along an axis, w.r.t. the phase of a single transducer
        '''
        ysort = ''.join(sorted(axis + 'y'))  # ''.join(sorted(axis + 'y')) always give xy, yy, yz
        # p_part = 2 * (self.phase_derivative(2 * axis, '', total_derivatives, phased_derivatives) +
        #               self.phase_derivative(axis, axis, total_derivatives, phased_derivatives))
        # x_part = 2 * (self.phase_derivative(2 * axis + 'x', 'x', total_derivatives, phased_derivatives) +
        #               self.phase_derivative('x' + axis, 'x' + axis, total_derivatives, phased_derivatives))
        # y_part = 2 * (self.phase_derivative(2 * axis + 'y', 'y', total_derivatives, phased_derivatives) +
        #               self.phase_derivative(ysort, ysort, total_derivatives, phased_derivatives))
        # z_part = 2 * (self.phase_derivative(2 * axis + 'z', 'z', total_derivatives, phased_derivatives) +
        #               self.phase_derivative(axis + 'z', axis + 'z', total_derivatives, phased_derivatives))

        # return self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)

        # New method to remove dependance on 'phase_derivative'
        # This will reuce the number of calculations needed when the amplitude jacobian is needed as well
        p = total_derivatives['']
        pa = total_derivatives[axis]
        paa = total_derivatives[2 * axis]

        px = total_derivatives['x']
        py = total_derivatives['y']
        pz = total_derivatives['z']

        pax = total_derivatives['x' + axis]
        pay = total_derivatives[ysort]  # ''.join(sorted(axis + 'y')) always give xy, yy, yz
        paz = total_derivatives[axis + 'z']

        paax = total_derivatives[2 * axis + 'x']
        paay = total_derivatives[2 * axis + 'y']
        paaz = total_derivatives[2 * axis + 'z']

        p_i = phased_derivatives['']
        pa_i = phased_derivatives[axis]
        paa_i = phased_derivatives[2 * axis]

        px_i = phased_derivatives['x']
        py_i = phased_derivatives['y']
        pz_i = phased_derivatives['z']

        pax_i = phased_derivatives['x' + axis]
        pay_i = phased_derivatives[ysort]  # ''.join(sorted(axis + 'y')) always give xy, yy, yz
        paz_i = phased_derivatives[axis + 'z']

        paax_i = phased_derivatives[2 * axis + 'x']
        paay_i = phased_derivatives[2 * axis + 'y']
        paaz_i = phased_derivatives[2 * axis + 'z']

        p_part = 2 * (paa * np.conj(p_i) + p * np.conj(paa_i) + 2 * pa * np.conj(pa_i))
        x_part = 2 * (paax * np.conj(px_i) + px * np.conj(paax_i) + 2 * pax * np.conj(pax_i))
        y_part = 2 * (paay * np.conj(py_i) + py * np.conj(paay_i) + 2 * pay * np.conj(pay_i))
        z_part = 2 * (paaz * np.conj(pz_i) + pz * np.conj(paaz_i) + 2 * paz * np.conj(paz_i))

        return self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)
        # total = self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)
        # return total
        # if amplitudes is None:
        #     return total.imag
        # else:
        #     return np.concatenate((total.imag, total.real / amplitudes))

    def complex_dot(self, z1, z2):
        '''
        Calculate the "complex dot product" defined as
            Re(z1) Re(z2) + Im(z1) Im(z2)
        '''
        assert False  # Deprecated method!
        return z1.real * z2.real + z1.imag * z2.imag

    def phase_derivative(self, der_1, der_2, total_derivatives, phased_derivatives):
        '''
        Calculates the partial derivative of a part of the objective function w.r.t. a single phase
        'der_1' and 'der_2' are strings with the two derivatives from the objective function
        '''
        assert False  # Deprecated method!
        p1 = total_derivatives[der_1]
        p2 = total_derivatives[der_2]

        pi1 = phased_derivatives[der_1]
        pi2 = phased_derivatives[der_2]

        return (p1 * np.conj(pi2) + p2 * np.conj(pi1)).imag
        #return p1.imag * pi2.real + p2.imag * pi1.real - p1.real * pi2.imag - p2.real * pi1.imag

    def initialize_spatial_derivatives(self):
        '''
        Calculate and set the spatial derivatives for each transducer.
        These are the same regardless of the amplitude and phase of the transducers,
        and remains constant throughout the optimization.
        '''
        # Pre-initialize dictionary with arrays
        assert False  # Deprecated method!
        num_trans = self.array.num_transducers
        self.spatial_derivatives = {
            '': np.empty(num_trans, complex),
            'x': np.empty(num_trans, complex),
            'y': np.empty(num_trans, complex),
            'z': np.empty(num_trans, complex),
            'xx': np.empty(num_trans, complex),
            'yy': np.empty(num_trans, complex),
            'zz': np.empty(num_trans, complex),
            'xy': np.empty(num_trans, complex),
            'xz': np.empty(num_trans, complex),
            'yz': np.empty(num_trans, complex),
            'xxx': np.empty(num_trans, complex),
            'yyy': np.empty(num_trans, complex),
            'zzz': np.empty(num_trans, complex),
            'xxy': np.empty(num_trans, complex),
            'xxz': np.empty(num_trans, complex),
            'yyx': np.empty(num_trans, complex),
            'yyz': np.empty(num_trans, complex),
            'zzx': np.empty(num_trans, complex),
            'zzy': np.empty(num_trans, complex)
        }
        for idx in range(num_trans):
            # Derivatives of the omnidirectional green's function
            difference = self.focus - self.array.transducer_positions[idx]
            r = norm(difference)
            kr = self.array.k * r
            jkr = 1j * kr
            phase = np.exp(jkr)

            # Zero derivatives (Pressure)
            self.spatial_derivatives[''][idx] = phase / r

            # First order derivatives
            coeff = (jkr - 1) * phase / r**3
            self.spatial_derivatives['x'][idx] = difference[0] * coeff
            self.spatial_derivatives['y'][idx] = difference[1] * coeff
            self.spatial_derivatives['z'][idx] = difference[2] * coeff

            # Second order derivatives
            coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
            constant = (jkr - 1) * phase / r**3
            self.spatial_derivatives['xx'][idx] = difference[0]**2 * coeff + constant
            self.spatial_derivatives['yy'][idx] = difference[1]**2 * coeff + constant
            self.spatial_derivatives['zz'][idx] = difference[2]**2 * coeff + constant
            self.spatial_derivatives['xy'][idx] = difference[0] * difference[1] * coeff
            self.spatial_derivatives['xz'][idx] = difference[0] * difference[2] * coeff
            self.spatial_derivatives['yz'][idx] = difference[1] * difference[2] * coeff

            # Third order derivatives
            constant = (3 - 3 * jkr - kr**2) * phase / r**5
            coeff = ((jkr - 1) * (15 - kr**2) + 5 * kr**2) * phase / r**7
            self.spatial_derivatives['xxx'][idx] = difference[0] * (3 * constant + difference[0]**2 * coeff)
            self.spatial_derivatives['yyy'][idx] = difference[1] * (3 * constant + difference[1]**2 * coeff)
            self.spatial_derivatives['zzz'][idx] = difference[2] * (3 * constant + difference[2]**2 * coeff)
            self.spatial_derivatives['xxy'][idx] = difference[1] * (constant + difference[0]**2 * coeff)
            self.spatial_derivatives['xxz'][idx] = difference[2] * (constant + difference[0]**2 * coeff)
            self.spatial_derivatives['yyx'][idx] = difference[0] * (constant + difference[1]**2 * coeff)
            self.spatial_derivatives['yyz'][idx] = difference[2] * (constant + difference[1]**2 * coeff)
            self.spatial_derivatives['zzx'][idx] = difference[0] * (constant + difference[2]**2 * coeff)
            self.spatial_derivatives['zzy'][idx] = difference[1] * (constant + difference[2]**2 * coeff)
