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
        #phase -= phase.max()
        self.focus_phase = phase
        self.focus_point = focus
        # WARNING: Setting the initial condition for the phases to have an actual pressure focus point
        # at the desired levitation point will cause the optimization to fail!
        # self.phases = phase  # TODO: This is temporary until a proper optimisation scheme has been developed

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

    def greens_function(self, transducer_id, receiver_position):
        def sin_angle(v1, v2):
            cos = np.sum(v1 * v2, axis=-1) / norm(v1, axis=-1) / norm(v2, axis=-1)
            return (1 - cos**2)**0.5

        source_position = self.transducer_positions[transducer_id]
        source_normal = self.transducer_normals[transducer_id]

        dist = norm(source_position - receiver_position, axis=-1)
        sin = sin_angle(source_normal, receiver_position - source_position)
        directivity = jn(0, self.k * self.transducer_size / 2 * sin)  # Piston far field
        #directivity = 1  # Omnidirectional
        return directivity / dist * np.exp(1j * self.k * dist)


class gorkov_optimizer:

    def __init__(self, array, c_sphere=2350, rho_sphere=1040, radius_sphere=1e-3):
        # Default values for the speed of sound are from https://spiremt.com/support/SoundSpeedTable
        # Polystyrene, longitudinal waves. I have no idea if this is correct or not
        # The density is from google. Be aware that the density of polystyrene and styrofoam are radically different, 
        # styrofoam is more like 30!
        self.array = array

        self.c_sphere = c_sphere
        self.rho_sphere = rho_sphere
        self.radius_sphere = radius_sphere
        self.rho_air = array.rho
        self.c_air = array.c

        self.diff_step = 1 / self.array.k
        self.pressure_weight = 1
        self.gradient_weights = (1, 1, 1)

    def run(self):
        V = 4 / 3 * np.pi * self.radius_sphere**3
        self.pressure_coefficient = V / 2 * self.rho_air * ((self.rho_air * self.c_air)**-2 - (self.rho_sphere * self.c_sphere)**-2)
        c1 = 3 / 2 * V / (2 * np.pi * self.array.freq)**2 / self.rho_air
        self.gradient_coefficient = c1 * (self.rho_sphere - self.rho_air) / (2 * self.rho_sphere + self.rho_air)
        self.called = 0
        self.previous_phase = self.array.phases
        self.result = minimize(self.objective_function, self.array.phases, method='BFGS', jac=True)
        self.array.phases = self.result.x

    def objective_function(self, phases):
        self.called += 1
        logger.debug('Obective function: called {} times'.format(self.called))
        # Store the state of the array.
        # The optimization rutine assumes that the phase values does not change internally.
        current_phases = self.array.phases
        logger.debug('\tRMS phase difference between this call and the previous is {}'.format(np.sqrt(np.mean((phases - self.previous_phase)**2))))
        self.array.phases = phases
        self.previous_phases = phases
        self.calculate_pressure_matrix()

        pressure = self.total_pressure[self.finite_difference_coefficients[''][0]] * self.finite_difference_coefficients[''][1]
        uxx = self.gorkov_laplacian('x')
        uyy = self.gorkov_laplacian('y')
        uzz = self.gorkov_laplacian('z')

        wx, wy, wz = self.gradient_weights
        wp = self.pressure_weight
        value = wp * np.abs(pressure)**2 - wx * uxx - wy * uyy - wz * uzz

        logger.debug('\tPressure contribution: {}'.format(wp * np.abs(pressure)**2))
        logger.debug('\tUxx contribution: {}'.format(-wx * uxx))
        logger.debug('\tUyy contribution: {}'.format(-wy * uyy))
        logger.debug('\tUzz contribution: {}'.format(-wz * uzz))

        derivatives = np.zeros(self.array.num_transducers)
        for idx in range(self.array.num_transducers):
            dp = self.single_derivative(idx, '', '')
            duxx = self.gorkov_laplacian_derivative('x', idx)
            duyy = self.gorkov_laplacian_derivative('y', idx)
            duzz = self.gorkov_laplacian_derivative('z', idx)

            derivatives[idx] = wp * dp - wx * duxx - wy * duyy - wz * duzz

        self.array.phases = current_phases
        return value, derivatives

    def complex_dot(self, z1, z2):
        '''
        Calculate the "complex dot product" defined as
            Re(z1) Re(z2) + Im(z1) Im(z2)
        '''
        return z1.real * z2.real + z1.imag * z2.imag

    def single_derivative(self, idx, der_1, der_2):
        '''
        Calculates the partial derivative of a part of the objective function w.r.t. a single phase
        'der_1' and 'der_2' are strings with the two derivatives from the objective function
        '''
        hinv = 1 / self.diff_step

        p1 = hinv**len(der_1) * np.sum(self.total_pressure[self.finite_difference_coefficients[der_1][0]] * self.finite_difference_coefficients[der_1][1])
        p2 = hinv**len(der_2) * np.sum(self.total_pressure[self.finite_difference_coefficients[der_2][0]] * self.finite_difference_coefficients[der_2][1])

        pi1 = hinv**len(der_1) * np.sum(self.transducer_pressures[idx][self.finite_difference_coefficients[der_1][0]] * self.finite_difference_coefficients[der_1][1])
        pi2 = hinv**len(der_2) * np.sum(self.transducer_pressures[idx][self.finite_difference_coefficients[der_2][0]] * self.finite_difference_coefficients[der_2][1])

        return p1.imag * pi2.real + p2.imag * pi1.real - p1.real * pi2.imag - p2.real * pi1.imag

    def gorkov_laplacian_derivative(self, axis, idx):
        '''
        Calculates the derivative of the Gor'kov laplacian along an axis, w.r.t. the phase of a single transducer
        '''
        pressure_part = 2 * (self.single_derivative(idx, 2 * axis, '') + self.single_derivative(idx, axis, axis))
        x_part = 2 * (self.single_derivative(idx, 2 * axis + 'x', axis) + self.single_derivative(idx, axis + 'x', axis + 'x'))
        y_part = 2 * (self.single_derivative(idx, 2 * axis + 'y', axis) + self.single_derivative(idx, axis + 'y', axis + 'y'))
        z_part = 2 * (self.single_derivative(idx, 2 * axis + 'z', axis) + self.single_derivative(idx, axis + 'z', axis + 'z'))

        return self.pressure_coefficient * pressure_part + self.gradient_coefficient * (x_part + y_part + z_part)

    def gorkov_laplacian(self, axis):
        '''
        Calculates a part of the Gor'kov Laplacian, along the specified axis

        'axis': a single character 'x', 'y', or 'z'
        '''

        hinv = 1 / self.diff_step
        # TODO: 'pa' will be one of px, py, pz. 'paa' will be one of the pna variants
        # This can be optimized better
        p = np.sum(self.total_pressure[self.finite_difference_coefficients[''][0]] * self.finite_difference_coefficients[''][1])
        pa = hinv * np.sum(self.total_pressure[self.finite_difference_coefficients[axis][0]] * self.finite_difference_coefficients[axis][1])
        paa = hinv**2 * np.sum(self.total_pressure[self.finite_difference_coefficients[2 * axis][0]] * self.finite_difference_coefficients[2 * axis][1])

        px = hinv * np.sum(self.total_pressure[self.finite_difference_coefficients['x'][0]] * self.finite_difference_coefficients['x'][1])
        py = hinv * np.sum(self.total_pressure[self.finite_difference_coefficients['y'][0]] * self.finite_difference_coefficients['y'][1])
        pz = hinv * np.sum(self.total_pressure[self.finite_difference_coefficients['z'][0]] * self.finite_difference_coefficients['z'][1])

        pax = hinv**2 * np.sum(self.total_pressure[self.finite_difference_coefficients[axis + 'x'][0]] * self.finite_difference_coefficients[axis + 'x'][1])
        pay = hinv**2 * np.sum(self.total_pressure[self.finite_difference_coefficients[axis + 'y'][0]] * self.finite_difference_coefficients[axis + 'y'][1])
        paz = hinv**2 * np.sum(self.total_pressure[self.finite_difference_coefficients[axis + 'z'][0]] * self.finite_difference_coefficients[axis + 'z'][1])

        paax = hinv**3 * np.sum(self.total_pressure[self.finite_difference_coefficients[2 * axis + 'x'][0]] * self.finite_difference_coefficients[2 * axis + 'x'][1])
        paay = hinv**3 * np.sum(self.total_pressure[self.finite_difference_coefficients[2 * axis + 'y'][0]] * self.finite_difference_coefficients[2 * axis + 'y'][1])
        paaz = hinv**3 * np.sum(self.total_pressure[self.finite_difference_coefficients[2 * axis + 'z'][0]] * self.finite_difference_coefficients[2 * axis + 'z'][1])

        # Calculate individual parts
        pressure_part = 2 * (self.complex_dot(paa, p) + self.complex_dot(pa, pa))
        x_part = 2 * (self.complex_dot(paax, pa) + self.complex_dot(pax, pax))
        y_part = 2 * (self.complex_dot(paay, pa) + self.complex_dot(pay, pay))
        z_part = 2 * (self.complex_dot(paaz, pa) + self.complex_dot(paz, paz))

        logger.debug("\tGor'Kov Laplacian along {}-axis:".format(axis))
        logger.debug('\t\tPressure part is : {}'.format(pressure_part))
        logger.debug('\t\tx part is : {}'.format(x_part))
        logger.debug('\t\ty part is : {}'.format(y_part))
        logger.debug('\t\tz part is : {}'.format(z_part))
        logger.debug('\t\tPressure contribution: {:.6e}'.format(self.pressure_coefficient * pressure_part))
        logger.debug('\t\tGradient contribution: {:.6e}'.format(-self.gradient_coefficient * (x_part + y_part + z_part)))

        return self.pressure_coefficient * pressure_part - self.gradient_coefficient * (x_part + y_part + z_part)

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
        
