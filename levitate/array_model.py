import numpy as np
from numpy.linalg import norm
from scipy.special import jn
from scipy.optimize import minimize, basinhopping
import logging
from itertools import permutations

logger = logging.getLogger(__name__)


def rectangular_grid(shape, spread):
    x = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0]) * spread
    y = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1]) * spread

    numel = np.prod(shape)
    positions = np.empty((numel, 3))
    normals = np.empty((numel, 3))
    counter = 0
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            positions[counter, :] = np.r_[x[ix], y[iy], 0]
            normals[counter, :] = np.r_[0, 0, 1]
            counter += 1
    return positions, normals


class transducer_array:
    c = 343  # Speed of sound, shared between all instances
    rho = 1.2

    def __init__(self, focus_point=[0, 0, 0.2], grid=None, transducer_size=10e-3, shape=16, freq=40e3):
        self.focus_point = focus_point
        self.transducer_size = transducer_size
        self.freq = freq
        self.k = 2 * np.pi * self.freq / self.c

        if not hasattr(shape, '__len__') or len(shape) == 1:
            self.shape = (shape, shape)
        else:
            self.shape = shape
        if grid is None:
            self.transducer_positions, self.transducer_normals = rectangular_grid(self.shape, self.transducer_size)
        else:
            self.transducer_positions, self.transducer_normals = grid
        self.num_transducers = self.transducer_positions.shape[0]
        self.amplitudes = np.ones(self.num_transducers)
        self.phases = np.zeros(self.num_transducers)

    def focus_phases(self, focus):
        phase = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            phase[idx] = -norm(self.transducer_positions[idx, :] - focus) * self.k
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        self.focus_point = focus
        return phase
        # WARNING: Setting the initial condition for the phases to have an actual pressure focus point
        # at the desired levitation point will cause the optimization to fail!
        # self.phases = phase  # TODO: This is temporary until a proper optimisation scheme has been developed

    def twin_signature(self, position=(0, 0), angle=0):
        x = position[0]
        y = position[1]
        # TODO: Rotate, shift, and make sure that the calculateion below actually works
        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if self.transducer_positions[idx, 0] < x:
                signature[idx] = -np.pi / 2
            else:
                signature[idx] = np.pi / 2
        return signature

    def vortex_signature(self, position=(0, 0), angle=0):
        x = position[0]
        y = position[1]
        # TODO: Rotate, shift, and make sure that the calculateion below actually works
        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            signature[idx] = np.arctan2(self.transducer_positions[idx, 1], self.transducer_positions[idx, 0])
        return signature

    def bottle_signature(self, position=(0, 0), radius=None):
        x = position[0]
        y = position[1]
        # TODO: Rotate, shift, and make sure that the calculateion below actually works

        if radius is None:
            A = np.prod(self.shape)*self.transducer_size**2
            radius = (A/2/np.pi)**0.5

        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if norm(self.transducer_positions[idx, 0:2]) > radius:
                signature[idx] = np.pi
            else:
                signature[idx] = 0
        return signature

    def signature(self, phases=None, focus=None):
        if phases is None:
            phases = self.phases
        if focus is None:
            focus = self.focus_point
        focus_phases = self.focus_phases(focus)
        return np.mod(phases - focus_phases + np.pi, 2 * np.pi) - np.pi

    def calculate_pressure(self, point, transducer=None):
        '''
            Calculates the complex pressure amplitude created by the array.

            Parameters
            ----------
            point : ndarray or tuple
                Pass either a Nx3 ndarray with [x,y,z] as rows or a tuple with three matrices for x, y, z.
            transducer : int, optional
                Calculate only the pressure for the transducer with this index.
                If None (default) the sum from all transducers is calculated.
        '''
        if type(point) is tuple:
            reshape = True
            shape = point[0].shape
            raveled = [pi.ravel() for pi in point]
            point = np.stack(raveled, axis=1)
        else:
            reshape = False

        if transducer is None:
            # Calculate for the sum of all transducers
            p = 0
            for idx in range(self.num_transducers):
                p += self.greens_function(idx, point) * self.amplitudes[idx] * np.exp(1j * self.phases[idx])
        else:
            p = self.greens_function(transducer, point) * self.amplitudes[transducer] * np.exp(1j * self.phases[transducer])

        if reshape:
            return p.reshape(shape)
        else:
            return p

    def directivity(self, transducer_id, receiver_position):
        source_position = self.transducer_positions[transducer_id]
        source_normal = self.transducer_normals[transducer_id]
        difference = receiver_position - source_position

        cos_angle = np.sum(source_normal * difference, axis=-1) / norm(source_normal, axis=-1) / norm(difference, axis=-1)
        sin_angle = (1 - cos_angle**2)**0.5
        k_a_sin = sin_angle * self.k * self.transducer_size / 2
        # Circular postion in baffle?
        #val = jn(0, k_a_sin)
        # Circular piston in baffle, version 2
        #  TODO: Check this formula!
        #val = jn(1, k_a_sin) / k_a_sin
        val = 1
        return val

    def spherical_spreading(self, transducer_id, receiver_position):
        source_position = self.transducer_positions[transducer_id]
        dist = norm(source_position - receiver_position, axis=-1)
        return 1 / dist * np.exp(1j * self.k * dist)

    def greens_function(self, transducer_id, receiver_position):
        directional_part = self.directivity(transducer_id, receiver_position)
        spherical_part = self.spherical_spreading(transducer_id, receiver_position)
        return directional_part * spherical_part

    def spatial_derivatives(self, focus, h=None):
        '''
        Calculate and set the spatial derivatives for each transducer.
        These are the same regardless of the amplitude and phase of the transducers,
        and remains constant throughout the optimization.
        '''
        # Pre-initialize dictionary with arrays
        # TODO: enable selective calculation of the derivatives actually needed
        num_trans = self.num_transducers
        spatial_derivatives = {
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
            difference = focus - self.transducer_positions[idx]
            r = norm(difference)
            kr = self.k * r
            jkr = 1j * kr
            phase = np.exp(jkr)

            # Zero derivatives (Pressure)
            spatial_derivatives[''][idx] = phase / r

            # First order derivatives
            coeff = (jkr - 1) * phase / r**3
            spatial_derivatives['x'][idx] = difference[0] * coeff
            spatial_derivatives['y'][idx] = difference[1] * coeff
            spatial_derivatives['z'][idx] = difference[2] * coeff

            # Second order derivatives
            coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
            constant = (jkr - 1) * phase / r**3
            spatial_derivatives['xx'][idx] = difference[0]**2 * coeff + constant
            spatial_derivatives['yy'][idx] = difference[1]**2 * coeff + constant
            spatial_derivatives['zz'][idx] = difference[2]**2 * coeff + constant
            spatial_derivatives['xy'][idx] = difference[0] * difference[1] * coeff
            spatial_derivatives['xz'][idx] = difference[0] * difference[2] * coeff
            spatial_derivatives['yz'][idx] = difference[1] * difference[2] * coeff

            # Third order derivatives
            constant = (3 - 3 * jkr - kr**2) * phase / r**5
            coeff = ((jkr - 1) * (15 - kr**2) + 5 * kr**2) * phase / r**7
            spatial_derivatives['xxx'][idx] = difference[0] * (3 * constant + difference[0]**2 * coeff)
            spatial_derivatives['yyy'][idx] = difference[1] * (3 * constant + difference[1]**2 * coeff)
            spatial_derivatives['zzz'][idx] = difference[2] * (3 * constant + difference[2]**2 * coeff)
            spatial_derivatives['xxy'][idx] = difference[1] * (constant + difference[0]**2 * coeff)
            spatial_derivatives['xxz'][idx] = difference[2] * (constant + difference[0]**2 * coeff)
            spatial_derivatives['yyx'][idx] = difference[0] * (constant + difference[1]**2 * coeff)
            spatial_derivatives['yyz'][idx] = difference[2] * (constant + difference[1]**2 * coeff)
            spatial_derivatives['zzx'][idx] = difference[0] * (constant + difference[2]**2 * coeff)
            spatial_derivatives['zzy'][idx] = difference[1] * (constant + difference[2]**2 * coeff)

        return spatial_derivatives

class optimizer:

    def __init__(self, array=None):
        if array is None:
            self.array = transducer_array()
        else:
            self.array = array
        self.objective_list = []
        self.basinhopping = False

    def __call__(self):
        # Initialize all parts of the objective function
        # Assemble objective function and jacobian function
        # Start optimization
        # Basin hopping? Check number of iterations?
        # Return phases?
        self.initialize()
        args = {'jac': self.jacobian,
            'method': 'BFGS', 'options': {'return_all': False, 'gtol': 5e-5, 'norm': 2, 'disp': True}}
        if self.basinhopping:
            self.result = basinhopping(self.function, self.array.phases, T=1e-5, minimizer_kwargs=args, disp=True)
        else:
            #self.result = minimize(self.function, self.array.phases, jac=self.jacobian, callback=None,
                # method='L-BFGS-B', bounds=[(-3*np.pi, 3*np.pi)]*self.array.num_transducers, options={'gtol': 1e-7, 'ftol': 1e-12})
            #    method='BFGS', options={'return_all': True, 'gtol': 1e-5, 'norm': 2})
            self.result = minimize(self.function, self.array.phases, callback=None, **args)
        self.phases = self.result.x

    def function(self, phases):
        value = 0
        for objective, weight in self.objective_list:
            value += objective.function(phases) * weight
        return value

    def jacobian(self, phases):
        value = np.zeros(self.array.num_transducers)
        for objective, weight in self.objective_list:
            value += objective.jacobian(phases) * weight
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


class pressure_point:
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
        self.gradient_normalization = self.array.k * norm(self.focus - np.mean(self.array.transducer_positions, axis=0))
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
                self.extra_points.append(p_point)

            # Loop through orders from 1 and up
            # Add new grid points each order to the list of poins
            # Scale with radius and shift with focus
            # Create a bunch of new pressure_point and initialize them
            # Add the new pressure_points along with weights to a list that
            # should be evaluated in the function and jacobian.

    def function(self, phases):
        phase_coeff = np.exp(1j * phases)
        p_part = np.abs((phase_coeff * self.spatial_derivatives['']).sum())**2
        x_part = np.abs((phase_coeff * self.spatial_derivatives['x']).sum())**2
        y_part = np.abs((phase_coeff * self.spatial_derivatives['y']).sum())**2
        z_part = np.abs((phase_coeff * self.spatial_derivatives['z']).sum())**2

        wx, wy, wz = self.gradient_weights
        value = p_part + (wx * x_part + wy * y_part + wz * z_part)/self.gradient_normalization

        if self.order is not None:
            for p_point in self.extra_points:
                value += p_point.function(phases)
        return value

    def jacobian(self, phases):
        phase_coeff = np.exp(1j * phases)
        phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            phased_derivatives[key] = phase_coeff * value
            total_derivatives[key] = np.sum(phased_derivatives[key])
        # TODO: Check that this actually works for the gradient parts
        p_part = 2 * (total_derivatives[''] * np.conj(phased_derivatives[''])).imag
        x_part = 2 * (total_derivatives['x'] * np.conj(phased_derivatives['x'])).imag
        y_part = 2 * (total_derivatives['y'] * np.conj(phased_derivatives['y'])).imag
        z_part = 2 * (total_derivatives['z'] * np.conj(phased_derivatives['z'])).imag

        wx, wy, wz = self.gradient_weights
        values = p_part + (wx * x_part + wy * y_part + wz * z_part) / self.gradient_normalization

        if self.order is not None:
            for p_point in self.extra_points:
                values += p_point.jacobian(phases)
        return values


class gorkov_laplacian:

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

        self.diff_step = 1 / self.array.k
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

    def function(self, phases):
        # TODO: Move the addition of pressure - Gor'kov laplacian to a higher level optimizer
        # The higher level optimizer is then only responsible for assebling an objective function and the derivatives
        # Use sub-functions to calculate the different parts of the objective function, e.g. one function for the Gor'kov laplacian
        # and its derivatives, an another function for a pressure focus point and its derivatives.
        # This would result in higher flexibility since an objective function can be built by disjoint parts.
        # This could be done using a separate class instance for every 'feature' in the objective function, e.g. one instance for
        # maximising the Gor'kov laplacian, some more instances for minimizing/maximising the pressure at some locations.
        # This results in a small optimizer class that 'owns/shares' instances connected to 'features' in the objective function.
        self.objective_evals += 1
        self.total_evals += 1
        logger.info('Objective function call {}:\tPhase difference: RMS = {:.4e}\t Max = {:.4e}'.format(self.objective_evals, np.sqrt(np.mean((phases - self.previous_phase)**2)), np.max(np.abs(phases - self.previous_phase))))
        self.previous_phase = phases

        phase_coeff = np.exp(1j * phases)
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            total_derivatives[key] = np.sum(phase_coeff * value)

        p_part = total_derivatives['']
        x_part = self.laplacian_axis('x', total_derivatives)
        y_part = self.laplacian_axis('y', total_derivatives)
        z_part = self.laplacian_axis('z', total_derivatives)

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        value = wp * np.abs(p_part)**2 - wx * x_part - wy * y_part - wz * z_part

        logger.debug('\tPressure contribution: {:.6e}'.format(wp * np.abs(p_part)**2))
        logger.debug('\tUxx contribution: {:.6e}'.format(-wx * x_part))
        logger.debug('\tUyy contribution: {:.6e}'.format(-wy * y_part))
        logger.debug('\tUzz contribution: {:.6e}'.format(-wz * z_part))

        return value

    def jacobian(self, phases):
        self.jacobian_evals += 1
        self.total_evals += 1
        logger.info('Jacobian function call {}:\tPhase difference: RMS = {:.4e}\t Max = {:.4e}'.format(self.jacobian_evals, np.sqrt(np.mean((phases - self.previous_phase)**2)), np.max(np.abs(phases - self.previous_phase))))
        self.previous_phase = phases

        phase_coeff = np.exp(1j * phases)
        phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            phased_derivatives[key] = phase_coeff * value
            total_derivatives[key] = np.sum(phased_derivatives[key])

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        p_part = self.phase_derivative('', '', total_derivatives, phased_derivatives)
        x_part = self.laplacian_axis_derivative('x', total_derivatives, phased_derivatives)
        y_part = self.laplacian_axis_derivative('y', total_derivatives, phased_derivatives)
        z_part = self.laplacian_axis_derivative('z', total_derivatives, phased_derivatives)
        p_part = 2 * (total_derivatives[''] * np.conj(phased_derivatives[''])).imag

        derivatives = wp * p_part - wx * x_part - wy * y_part - wz * z_part

        return derivatives

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

        logger.debug("\tGor'Kov Laplacian along {}-axis:".format(axis))
        logger.debug('\t\tp contribution is: {:.6e}'.format(self.pressure_coefficient * p_part))
        logger.debug('\t\tx contribution is: {:.6e}'.format(- self.gradient_coefficient * x_part))
        logger.debug('\t\ty contribution is: {:.6e}'.format(- self.gradient_coefficient * y_part))
        logger.debug('\t\tz contribution is: {:.6e}'.format(- self.gradient_coefficient * z_part))

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

        total = self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)
        return total.imag
        # amplitude jacobian is total.real / amplitudes

    def complex_dot(self, z1, z2):
        '''
        Calculate the "complex dot product" defined as
            Re(z1) Re(z2) + Im(z1) Im(z2)
        '''
        assert False
        return z1.real * z2.real + z1.imag * z2.imag

    def phase_derivative(self, der_1, der_2, total_derivatives, phased_derivatives):
        '''
        Calculates the partial derivative of a part of the objective function w.r.t. a single phase
        'der_1' and 'der_2' are strings with the two derivatives from the objective function
        '''
        assert False
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
        assert False
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



    # def calculate_pressure_matrix(self):
    #     '''
    #     Calculates the pressure from all the individual transducers at the points required in the finite difference scheme
    #     '''
    #     mid_point = self.array.focus_point

    #     x, y, z = np.mgrid[-1:2, -1:2, -1:2]
    #     inner = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    #     outer = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]])

    #     points = np.concatenate((inner, outer), axis=0) * self.diff_step + mid_point

    #     self.transducer_pressures = np.empty((self.array.num_transducers, 33), dtype='complex128')
    #     for idx in range(self.array.num_transducers):
    #         self.transducer_pressures[idx] = self.array.calculate_pressure(points, idx)
    #     self.total_pressure = np.sum(self.transducer_pressures, axis=0)

    # zero_order_coefficients = {'': ([13], [1])}
    # first_order_coefficients = {
    #     'x': ([22, 4], [0.5, -0.5]),
    #     'y': ([16, 10], [0.5, -0.5]),
    #     'z': ([14, 12], [0.5, -0.5])
    # }
    # second_order_coefficients = {  # Duplicates are needed since the key access varies
    #     'xx': ([22, 13, 4], [1, -2, 1]),
    #     'xy': ([25, 2, 19, 7], [0.25, 0.25, -0.25, -0.25]),
    #     'xz': ([23, 3, 21, 5], [0.25, 0.25, -0.25, -0.25]),
    #     'yx': ([25, 2, 19, 7], [0.25, 0.25, -0.25, -0.25]),
    #     'yy': ([16, 13, 10], [1, -2, 1]),
    #     'yz': ([17, 9, 15, 11], [0.25, 0.25, -0.25, -0.25]),
    #     'zx': ([23, 3, 21, 5], [0.25, 0.25, -0.25, -0.25]),
    #     'zy': ([17, 9, 15, 11], [0.25, 0.25, -0.25, -0.25]),
    #     'zz': ([14, 13, 12], [1, -2, 1]),
    # }
    # third_order_coefficients = {
    #     'xxx': ([28, 27, 22, 4], [0.5, -0.5, -1, 1]),
    #     'xxy': ([25, 1, 19, 7, 16, 10], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'xxz': ([23, 3, 21, 5, 14, 12], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'yyx': ([25, 1, 7, 19, 22, 4], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'yyy': ([30, 29, 16, 10], [0.5, -0.5, -1, 1]),
    #     'yyz': ([17, 9, 15, 11, 14, 12], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'zzx': ([23, 3, 5, 21, 22, 4], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'zzy': ([17, 9, 11, 15, 16, 10], [0.5, -0.5, -0.5, 0.5, -1, 1]),
    #     'zzz': ([32, 31, 14, 12], [0.25, -0.25, -1, 1])
    # }
    # finite_difference_coefficients = {**zero_order_coefficients, **first_order_coefficients,
    #                                   **second_order_coefficients, **third_order_coefficients}

    '''
    def zero_order_coefficients():
        return [[0, 0, 0]], [1]

    def first_order_coefficients(derivative):
        if derivative == 'x':
            return [[1, 0, 0], [-1, 0, 0]], [0.5, -0.5]
        elif derivative == 'y':
            return [[0, 1, 0], [0, -1, 0]], [0.5, -0.5]
        elif derivative == 'z':
            return [[0, 0, 1], [0, 0, 1]], [0.5, -0.5]

    def second_order_coefficients(derivative):
        if derivative == 'xx':
            return [[1, 0, 0], [0, 0, 0], [-1, 0, 0]], [1, -2, 1]
            #return [[2, 0, 0], [0, 0, 0], [-2, 0, 0]], [0.25, -0.5, 0.25]
        elif derivative == 'yy':
            return [[0, 1, 0], [0, 0, 0], [0, -1, 0]], [1, -2, 1]
            #return [[0, 2, 0], [0, 0, 0], [0, -2, 0]], [0.25, -0.5, 0.25]
        elif derivative == 'zz':
            return [[0, 0, 1], [0, 0, 0], [0, 0, -1]], [1, -2, 1]
            #return [[0, 0, 2], [0, 0, 0], [0, 0, -2]], [0.25, -0.5, 0.25]
        elif derivative == 'xy':
            return [[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]], [0.25, 0.25, -0.25, -0.25]
        elif derivative == 'xz':
            return [[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1]], [0.25, 0.25, -0.25, -0.25]
        elif derivative == 'yz':
            return [[0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], [0.25, 0.25, -0.25, -0.25]

    def third_order_coefficients(derivative):
        if derivative == 'xxx':
            return [[2, 0, 0], [-2, 0, 0], [1, 0, 0], [-1, 0, 0]], [0.5, -0.5, -1, 1]
            #return [[3, 0, 0], [-3, 0, 0], [1, 0, 0], [-1, 0, 0]], [0.125, -0.125, -0.375, 0.375]
        elif derivative == 'yyy':
            return [[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0]], [0.5, -0.5, -1, 1]
            #return [[0, 3, 0], [0, -3, 0], [0, 1, 0], [0, -1, 0]], [0.125, -0.125, -0.375, 0.375]
        elif derivative == 'zzz':
            return [[0, 0, 2], [0, 0, -2], [0, 0, 1], [0, 0, -1]], [0.5, -0.5, -1, 1]
            #return [[0, 0, 3], [0, 0, -3], [0, 0, 1], [0, 0, -1]], [0.125, -0.125, -0.375, 0.375]
        elif derivative == 'xxy':
            return [[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [0, 1, 0], [0, -1, 0]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[2, 1, 0], [-2, -1, 0], [2, -1, 0], [-2, 1, 0], [0, 1, 0], [0, -1, 0]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
        elif derivative == 'xxz':
            return [[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [0, 0, 1], [0, 0, -1]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[2, 0, 1], [-2, 0, -1], [2, 0, -1], [-2, 0, 1], [0, 0, 1], [0, 0, -1]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
        elif derivative == 'xyy':
            return [[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[1, 2, 0], [-1, -2, 0], [-1, 2, 0], [1, -2, 0], [1, 0, 0], [-1, -0, 0]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
        elif derivative == 'xzz':
            return [[1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 0], [-1, 0, 0]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[1, 0, 2], [-1, 0, -2], [-1, 0, 2], [1, 0, -2], [1, 0, 0], [-1, 0, 0]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
        elif derivative == 'yyz':
            return [[0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 0, 1], [0, 0, -1]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[0, 2, 1], [0, -2, -1], [0, 2, -1], [0, -2, 1], [0, 0, 1], [0, 0, -1]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
        elif derivative == 'yzz':
            return [[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 0], [0, -1, 0]], [0.5, -0.5, -0.5, 0.5, -1, 1]
            #return [[0, 1, 2], [0, -1, -2], [0, -1, 2], [0, 1, -2], [0, 1, 0], [0, -1, 0]], [0.125, -0.125, -0.125, 0.125, -0.25, 0.25]
    '''
