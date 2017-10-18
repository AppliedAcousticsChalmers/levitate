import numpy as np
from numpy.linalg import norm
from scipy.special import jn
from scipy.optimize import minimize
import logging

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

    def __init__(self, grid=None, transducer_size=10e-3, shape=16, freq=40e3):
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

    def set_focus(self, focus):
        phase = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            phase[idx] = -norm(self.transducer_positions[idx, :] - focus) * self.k
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        self.focus_phase = phase
        self.focus_point = focus
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
            radius = np.max(self.transducer_positions[:, 0]) / 2

        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if norm(self.transducer_positions[idx, 0:2]) > radius:
                signature[idx] = np.pi
            else:
                signature[idx] = 0
        return signature

    def current_signature(self):
        return np.mod(self.phases - self.focus_phase + np.pi, 2 * np.pi) - np.pi

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


class gorkov_optimizer:

    def __init__(self, array, c_sphere=2350, rho_sphere=1040, radius_sphere=1e-3):
        # Default values for the speed of sound are from https://spiremt.com/support/SoundSpeedTable
        # Polystyrene, longitudinal waves. I have no idea if this is correct or not
        # The density is from google. Be aware that the density of polystyrene and styrofoam are radically different,
        # styrofoam is more like 30!
        self.array = array
        self.focus = array.focus_point  # TODO: This should be configurable from start
        # Rationale: If we want multiple focus points the array does not have a single defined focus point
        # but the Gor'Kov optimizer still only has a single focus point. (For now)

        self.c_sphere = c_sphere
        self.rho_sphere = rho_sphere
        self.radius_sphere = radius_sphere
        # TODO: Make these two parameters that are updated if the array changes
        self.rho_air = array.rho
        self.c_air = array.c

        self.diff_step = 1 / self.array.k
        self.pressure_weight = 1
        self.gradient_weights = (1, 1, 1)

    def run(self):
        V = 4 / 3 * np.pi * self.radius_sphere**3
        self.pressure_coefficient = V / 2 * self.rho_air * (1/(self.rho_air * self.c_air**2) - 1/(self.rho_sphere * self.c_sphere**2))
        c1 = 3 / 2 * V / (2 * np.pi * self.array.freq)**2 / self.rho_air
        self.gradient_coefficient = c1 * (self.rho_sphere - self.rho_air) / (2 * self.rho_sphere + self.rho_air)
        self.objective_evals = 0
        self.jacobian_evals = 0
        self.total_evals = 0
        self.previous_phase = self.array.phases
        self.initialize_spatial_derivatives()
        self.result = minimize(self.objective_function, self.array.phases, method='BFGS', jac=self.objective_jacobian, options={'return_all': True})
        self.array.phases = self.result.x

    def initialize_spatial_derivatives(self):
        '''
        Calculate and set the spatial derivatives for each transducer.
        These are the same regardless of the amplitude and phase of the transducers,
        and remains constant throughout the optimization.
        '''
        # Pre-initialize dictionary with arrays

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

    def objective_function(self, phases):
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
        logger.debug('Obective function: called {} times'.format(self.objective_evals))
        logger.debug('\tRMS phase difference between this call and the previous is {}'.format(np.sqrt(np.mean((phases - self.previous_phase)**2))))
        self.previous_phase = phases

        phase_coeff = np.exp(1j * phases)
        #phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            #phased_derivatives[key] = phase_coeff * value
            total_derivatives[key] = np.sum(phase_coeff * value)

        pressure = total_derivatives['']
        uxx = self.gorkov_laplacian('x', total_derivatives)
        uyy = self.gorkov_laplacian('y', total_derivatives)
        uzz = self.gorkov_laplacian('z', total_derivatives)

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        value = wp * np.abs(pressure)**2 - wx * uxx - wy * uyy - wz * uzz

        logger.debug('\tPressure contribution: {:.6e}'.format(wp * np.abs(pressure)**2))
        logger.debug('\tUxx contribution: {:.6e}'.format(-wx * uxx))
        logger.debug('\tUyy contribution: {:.6e}'.format(-wy * uyy))
        logger.debug('\tUzz contribution: {:.6e}'.format(-wz * uzz))

        # derivatives = np.zeros(self.array.num_transducers)
        # for idx in range(self.array.num_transducers):
        #     dp = self.phase_derivative(idx, '', '', total_derivatives, phased_derivatives)
        #     duxx = self.gorkov_laplacian_derivative('x', idx, total_derivatives, phased_derivatives)
        #     duyy = self.gorkov_laplacian_derivative('y', idx, total_derivatives, phased_derivatives)
        #     duzz = self.gorkov_laplacian_derivative('z', idx, total_derivatives, phased_derivatives)

        #     derivatives[idx] = wp * dp - wx * duxx - wy * duyy - wz * duzz

        return value #, derivatives

    def gorkov_laplacian(self, axis, total_derivatives):
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

        # Calculate individual parts
        p_part = 2 * (self.complex_dot(paa , p ) + self.complex_dot(pa , pa ))
        x_part = 2 * (self.complex_dot(paax, px) + self.complex_dot(pax, pax))
        y_part = 2 * (self.complex_dot(paay, py) + self.complex_dot(pay, pay))
        z_part = 2 * (self.complex_dot(paaz, pz) + self.complex_dot(paz, paz))

        logger.debug("\tGor'Kov Laplacian along {}-axis:".format(axis))
        logger.debug('\t\tp contribution is: {:.6e}'.format(self.pressure_coefficient * p_part))
        logger.debug('\t\tx contribution is: {:.6e}'.format(- self.gradient_coefficient * x_part))
        logger.debug('\t\ty contribution is: {:.6e}'.format(- self.gradient_coefficient * y_part))
        logger.debug('\t\tz contribution is: {:.6e}'.format(- self.gradient_coefficient * z_part))

        #set_trace()
        return self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)

    def objective_jacobian(self, phases):
        self.jacobian_evals += 1
        self.total_evals += 1
        logger.debug('Jacoian function: called {} times'.format(self.jacobian_evals))
        logger.debug('\tRMS phase difference between this call and the previous is {}'.format(np.sqrt(np.mean((phases - self.previous_phase)**2))))
        self.previous_phase = phases

        phase_coeff = np.exp(1j * phases)
        phased_derivatives = {}
        total_derivatives = {}
        for key, value in self.spatial_derivatives.items():
            phased_derivatives[key] = phase_coeff * value
            total_derivatives[key] = np.sum(phased_derivatives[key])

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight

        derivatives = np.zeros(self.array.num_transducers)
        for idx in range(self.array.num_transducers):
            dp = self.phase_derivative(idx, '', '', total_derivatives, phased_derivatives)
            duxx = self.gorkov_laplacian_derivative('x', idx, total_derivatives, phased_derivatives)
            duyy = self.gorkov_laplacian_derivative('y', idx, total_derivatives, phased_derivatives)
            duzz = self.gorkov_laplacian_derivative('z', idx, total_derivatives, phased_derivatives)

            derivatives[idx] = wp * dp - wx * duxx - wy * duyy - wz * duzz

        return derivatives

    def gorkov_laplacian_derivative(self, axis, idx, total_derivatives, phased_derivitives):
        '''
        Calculates the derivative of the Gor'kov laplacian along an axis, w.r.t. the phase of a single transducer
        '''
        ysort = ''.join(sorted(axis + 'y'))  # ''.join(sorted(axis + 'y')) always give xy, yy, yz
        p_part = 2 * (self.phase_derivative(idx, 2 * axis, '', total_derivatives, phased_derivitives) +
                      self.phase_derivative(idx, axis, axis, total_derivatives, phased_derivitives))
        x_part = 2 * (self.phase_derivative(idx, 2 * axis + 'x', 'x', total_derivatives, phased_derivitives) +
                      self.phase_derivative(idx, 'x' + axis, 'x' + axis, total_derivatives, phased_derivitives))
        y_part = 2 * (self.phase_derivative(idx, 2 * axis + 'y', 'y', total_derivatives, phased_derivitives) +
                      self.phase_derivative(idx, ysort, ysort, total_derivatives, phased_derivitives))
        z_part = 2 * (self.phase_derivative(idx, 2 * axis + 'z', 'z', total_derivatives, phased_derivitives) +
                      self.phase_derivative(idx, axis + 'z', axis + 'z', total_derivatives, phased_derivitives))

        return self.pressure_coefficient * p_part - self.gradient_coefficient * (x_part + y_part + z_part)

    def phase_derivative(self, idx, der_1, der_2, total_derivatives, phased_derivatives):
        '''
        Calculates the partial derivative of a part of the objective function w.r.t. a single phase
        'der_1' and 'der_2' are strings with the two derivatives from the objective function
        '''

        p1 = total_derivatives[der_1]
        p2 = total_derivatives[der_2]

        pi1 = phased_derivatives[der_1][idx]
        pi2 = phased_derivatives[der_2][idx]

        return p1.imag * pi2.real + p2.imag * pi1.real - p1.real * pi2.imag - p2.real * pi1.imag

    def complex_dot(self, z1, z2):
        '''
        Calculate the "complex dot product" defined as
            Re(z1) Re(z2) + Im(z1) Im(z2)
        '''
        return z1.real * z2.real + z1.imag * z2.imag

    def calculate_pressure_matrix(self):
        '''
        Calculates the pressure from all the individual transducers at the points required in the finite difference scheme
        '''
        mid_point = self.array.focus_point

        x, y, z = np.mgrid[-1:2, -1:2, -1:2]
        inner = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        outer = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0], [0, 0, -2], [0, 0, 2]])

        points = np.concatenate((inner, outer), axis=0) * self.diff_step + mid_point

        self.transducer_pressures = np.empty((self.array.num_transducers, 33), dtype='complex128')
        for idx in range(self.array.num_transducers):
            self.transducer_pressures[idx] = self.array.calculate_pressure(points, idx)
        self.total_pressure = np.sum(self.transducer_pressures, axis=0)

    zero_order_coefficients = {'': ([13], [1])}
    first_order_coefficients = {
        'x': ([22, 4], [0.5, -0.5]),
        'y': ([16, 10], [0.5, -0.5]),
        'z': ([14, 12], [0.5, -0.5])
    }
    second_order_coefficients = {  # Duplicates are needed since the key access varies
        'xx': ([22, 13, 4], [1, -2, 1]),
        'xy': ([25, 2, 19, 7], [0.25, 0.25, -0.25, -0.25]),
        'xz': ([23, 3, 21, 5], [0.25, 0.25, -0.25, -0.25]),
        'yx': ([25, 2, 19, 7], [0.25, 0.25, -0.25, -0.25]),
        'yy': ([16, 13, 10], [1, -2, 1]),
        'yz': ([17, 9, 15, 11], [0.25, 0.25, -0.25, -0.25]),
        'zx': ([23, 3, 21, 5], [0.25, 0.25, -0.25, -0.25]),
        'zy': ([17, 9, 15, 11], [0.25, 0.25, -0.25, -0.25]),
        'zz': ([14, 13, 12], [1, -2, 1]),
    }
    third_order_coefficients = {
        'xxx': ([28, 27, 22, 4], [0.5, -0.5, -1, 1]),
        'xxy': ([25, 1, 19, 7, 16, 10], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'xxz': ([23, 3, 21, 5, 14, 12], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'yyx': ([25, 1, 7, 19, 22, 4], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'yyy': ([30, 29, 16, 10], [0.5, -0.5, -1, 1]),
        'yyz': ([17, 9, 15, 11, 14, 12], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'zzx': ([23, 3, 5, 21, 22, 4], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'zzy': ([17, 9, 11, 15, 16, 10], [0.5, -0.5, -0.5, 0.5, -1, 1]),
        'zzz': ([32, 31, 14, 12], [0.25, -0.25, -1, 1])
    }
    finite_difference_coefficients = {**zero_order_coefficients, **first_order_coefficients,
                                      **second_order_coefficients, **third_order_coefficients}

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
