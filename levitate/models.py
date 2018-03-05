import numpy as np
import logging
from numpy.linalg import norm
from scipy.special import jn

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


class TransducerArray:

    finite_difference_coefficients = {'': (np.array([0, 0, 0]), 1),
        'x': (np.array([[1, 0, 0], [-1, 0, 0]]), [0.5, -0.5]),
        'y': (np.array([[0, 1, 0], [0, -1, 0]]), [0.5, -0.5]),
        'z': (np.array([[0, 0, 1], [0, 0, -1]]), [0.5, -0.5]),
        'xx': (np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]]), [1, -2, 1]),  # Alt: (np.array([[2, 0, 0], [0, 0, 0], [-2, 0, 0]]), [0.25, -0.5, 0.25])
        'yy': (np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]), [1, -2, 1]),  # Alt: (np.array([[0, 2, 0], [0, 0, 0], [0, -2, 0]]), [0.25, -0.5, 0.25])
        'zz': (np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]]), [1, -2, 1]),  # Alt: (np.array([[0, 0, 2], [0, 0, 0], [0, 0, -2]]), [0.25, -0.5, 0.25])
        'xy': (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]), [0.25, 0.25, -0.25, -0.25]),
        'xz': (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1]]), [0.25, 0.25, -0.25, -0.25]),
        'yz': (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1]]), [0.25, 0.25, -0.25, -0.25]),
        'xxx': (np.array([[2, 0, 0], [-2, 0, 0], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -1, 1]),  # Alt: (np.array([[3, 0, 0], [-3, 0, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.375, 0.375])
        'yyy': (np.array([[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -1, 1]),  # Alt: (np.array([[0, 3, 0], [0, -3, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.375, 0.375])
        'zzz': (np.array([[0, 0, 2], [0, 0, -2], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -1, 1]),  # Alt: (np.array([[0, 0, 3], [0, 0, -3], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.375, 0.375])
        'xxy': (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1]),  # Alt: (np.array([[2, 1, 0], [-2, -1, 0], [2, -1, 0], [-2, 1, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
        'xxz': (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -0.5, 0.5, -1, 1]),  # Alt: (np.array([[2, 0, 1], [-2, 0, -1], [2, 0, -1], [-2, 0, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
        'yyx': (np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1]),  # Alt: (np.array([[1, 2, 0], [-1, -2, 0], [-1, 2, 0], [1, -2, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
        'yyz': (np.array([[0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -0.5, 0.5, -1, 1]),  # Alt: (np.array([[0, 2, 1], [0, -2, -1], [0, 2, -1], [0, -2, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
        'zzx': (np.array([[1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1]),  # Alt: (np.array([[1, 0, 2], [-1, 0, -2], [-1, 0, 2], [1, 0, -2], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
        'zzy': (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt: (np.array([[0, 1, 2], [0, -1, -2], [0, -1, 2], [0, 1, -2], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
    }
    c = 343  # Speed of sound, shared between all instances
    rho = 1.2

    def __init__(self, focus_point=[0, 0, 0.2], grid=None, transducer_size=10e-3, shape=16, freq=40e3, directivity=None):
        self.focus_point = focus_point
        self.transducer_size = transducer_size
        self.freq = freq
        self.k = 2 * np.pi * self.freq / self.c
        self.use_directivity = directivity

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

        self.p0 = 6  # Pa @ 1 m distance on-axis. 
        # The murata transducers are measured to 85 dB SPL at 1 V at 1 m, which corresponds to ~6 Pa at 20 V
        # The datasheet specifies 120 dB SPL @ 0.3 m, which corresponds to ~6 Pa @ 1 m

    def focus_phases(self, focus):
        # TODO: Is this method really useful?
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
        # TODO: Is this method really useful?
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
        # TODO: Is this method really useful?
        x = position[0]
        y = position[1]
        # TODO: Rotate, shift, and make sure that the calculateion below actually works
        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            signature[idx] = np.arctan2(self.transducer_positions[idx, 1], self.transducer_positions[idx, 0])
        return signature

    def bottle_signature(self, position=(0, 0), radius=None):
        # TODO: Is this method really useful?
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
        # TODO: Is this method really useful?
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
            return self.p0 * p.reshape(shape)
        else:
            return self.p0 * p

    def directivity(self, transducer_id, receiver_position):
        if self.use_directivity is None:
            if receiver_position.ndim == 1:
                return 1
            else:
                return np.ones(receiver_position.shape[0])

        source_position = self.transducer_positions[transducer_id]
        source_normal = self.transducer_normals[transducer_id]
        difference = receiver_position - source_position

        cos_angle = np.sum(source_normal * difference, axis=-1) / norm(source_normal, axis=-1) / norm(difference, axis=-1)
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.transducer_size / 2

        if self.use_directivity.lower() == 'j0':
            # Circular postion in baffle?
            return jn(0, ka * sin_angle)
        if self.use_directivity.lower() == 'j1':
            # Circular piston in baffle, version 2
            #  TODO: Check this formula!
            # TODO: Needs to ignore warning as well!
            # vals = 2 * jn(1, k_a_sin) / k_a_sin
            # vals[np.isnan(vals)] = 1
            with np.errstate(invalid='ignore'):
                return np.where(sin_angle == 0, 1, 2 * jn(1, ka * sin_angle) / (ka * sin_angle))

        # If no match in the implemented directivities, use omnidirectional
        # TODO: Add a warning?
        assert False
        self.use_directivity = None
        return self.directivity(transducer_id, receiver_position)

    def spherical_spreading(self, transducer_id, receiver_position):
        source_position = self.transducer_positions[transducer_id]
        dist = norm(source_position - receiver_position, axis=-1)
        return 1 / dist * np.exp(1j * self.k * dist)

    def greens_function(self, transducer_id, receiver_position):
        directional_part = self.directivity(transducer_id, receiver_position)
        spherical_part = self.spherical_spreading(transducer_id, receiver_position)
        return directional_part * spherical_part

    def spatial_derivatives(self, focus, h=None, orders=3):
        '''
        Calculate and set the spatial derivatives for each transducer.
        These are the same regardless of the amplitude and phase of the transducers,
        and remains constant throughout the optimization.
        '''
        # Pre-initialize dictionary with arrays
        # TODO: enable selective calculation of the derivatives actually needed
        num_trans = self.num_transducers

        spherical_derivatives = {'': np.empty(num_trans, complex)}
        if orders > 0:
            spherical_derivatives['x'] = np.empty(num_trans, complex)
            spherical_derivatives['y'] = np.empty(num_trans, complex)
            spherical_derivatives['z'] = np.empty(num_trans, complex)
        if orders > 1:
            spherical_derivatives['xx'] = np.empty(num_trans, complex)
            spherical_derivatives['yy'] = np.empty(num_trans, complex)
            spherical_derivatives['zz'] = np.empty(num_trans, complex)
            spherical_derivatives['xy'] = np.empty(num_trans, complex)
            spherical_derivatives['xz'] = np.empty(num_trans, complex)
            spherical_derivatives['yz'] = np.empty(num_trans, complex)
        if orders > 2:
            spherical_derivatives['xxx'] = np.empty(num_trans, complex)
            spherical_derivatives['yyy'] = np.empty(num_trans, complex)
            spherical_derivatives['zzz'] = np.empty(num_trans, complex)
            spherical_derivatives['xxy'] = np.empty(num_trans, complex)
            spherical_derivatives['xxz'] = np.empty(num_trans, complex)
            spherical_derivatives['yyx'] = np.empty(num_trans, complex)
            spherical_derivatives['yyz'] = np.empty(num_trans, complex)
            spherical_derivatives['zzx'] = np.empty(num_trans, complex)
            spherical_derivatives['zzy'] = np.empty(num_trans, complex)

        spatial_derivatives = spherical_derivatives.copy()
        for idx in range(num_trans):
            # Derivatives of the omnidirectional green's function
            difference = focus - self.transducer_positions[idx]
            r = norm(difference)
            kr = self.k * r
            jkr = 1j * kr
            phase = np.exp(jkr)

            # Zero derivatives (Pressure)
            spherical_derivatives[''][idx] = phase / r

            # First order derivatives
            if orders > 0:
                coeff = (jkr - 1) * phase / r**3
                spherical_derivatives['x'][idx] = difference[0] * coeff
                spherical_derivatives['y'][idx] = difference[1] * coeff
                spherical_derivatives['z'][idx] = difference[2] * coeff

            # Second order derivatives
            if orders > 1:
                coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
                constant = (jkr - 1) * phase / r**3
                spherical_derivatives['xx'][idx] = difference[0]**2 * coeff + constant
                spherical_derivatives['yy'][idx] = difference[1]**2 * coeff + constant
                spherical_derivatives['zz'][idx] = difference[2]**2 * coeff + constant
                spherical_derivatives['xy'][idx] = difference[0] * difference[1] * coeff
                spherical_derivatives['xz'][idx] = difference[0] * difference[2] * coeff
                spherical_derivatives['yz'][idx] = difference[1] * difference[2] * coeff

            # Third order derivatives
            if orders > 2:
                constant = (3 - 3 * jkr - kr**2) * phase / r**5
                coeff = ((jkr - 1) * (15 - kr**2) + 5 * kr**2) * phase / r**7
                spherical_derivatives['xxx'][idx] = difference[0] * (3 * constant + difference[0]**2 * coeff)
                spherical_derivatives['yyy'][idx] = difference[1] * (3 * constant + difference[1]**2 * coeff)
                spherical_derivatives['zzz'][idx] = difference[2] * (3 * constant + difference[2]**2 * coeff)
                spherical_derivatives['xxy'][idx] = difference[1] * (constant + difference[0]**2 * coeff)
                spherical_derivatives['xxz'][idx] = difference[2] * (constant + difference[0]**2 * coeff)
                spherical_derivatives['yyx'][idx] = difference[0] * (constant + difference[1]**2 * coeff)
                spherical_derivatives['yyz'][idx] = difference[2] * (constant + difference[1]**2 * coeff)
                spherical_derivatives['zzx'][idx] = difference[0] * (constant + difference[2]**2 * coeff)
                spherical_derivatives['zzy'][idx] = difference[1] * (constant + difference[2]**2 * coeff)

            if self.use_directivity is not None:
                if h is None:
                    h = 1 / self.k
                directivity_derivatives = {}
                for key in spherical_derivatives.keys():
                    shifts, weights = self.finite_difference_coefficients[key]
                    directivity_derivatives[key] = np.sum(self.directivity(idx, shifts * h + focus) * weights) / h**len(key)

                spatial_derivatives[''][idx] = spherical_derivatives[''][idx] * directivity_derivatives['']

                if orders > 0:
                    spatial_derivatives['x'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['x'] + directivity_derivatives[''] * spherical_derivatives['x'][idx]
                    spatial_derivatives['y'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['y'] + directivity_derivatives[''] * spherical_derivatives['y'][idx]
                    spatial_derivatives['z'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['z'] + directivity_derivatives[''] * spherical_derivatives['z'][idx]

                if orders > 1:
                    spatial_derivatives['xx'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xx'] + directivity_derivatives[''] * spherical_derivatives['xx'][idx] + 2 * directivity_derivatives['x'] * spherical_derivatives['x'][idx]
                    spatial_derivatives['yy'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['yy'] + directivity_derivatives[''] * spherical_derivatives['yy'][idx] + 2 * directivity_derivatives['y'] * spherical_derivatives['y'][idx]
                    spatial_derivatives['zz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['zz'] + directivity_derivatives[''] * spherical_derivatives['zz'][idx] + 2 * directivity_derivatives['z'] * spherical_derivatives['z'][idx]
                    spatial_derivatives['xy'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xy'] + directivity_derivatives[''] * spherical_derivatives['xy'][idx] + spherical_derivatives['x'][idx] * directivity_derivatives['y'] + directivity_derivatives['x'] * spherical_derivatives['y'][idx]
                    spatial_derivatives['xz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xz'] + directivity_derivatives[''] * spherical_derivatives['xz'][idx] + spherical_derivatives['x'][idx] * directivity_derivatives['z'] + directivity_derivatives['x'] * spherical_derivatives['z'][idx]
                    spatial_derivatives['yz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['yz'] + directivity_derivatives[''] * spherical_derivatives['yz'][idx] + spherical_derivatives['y'][idx] * directivity_derivatives['z'] + directivity_derivatives['y'] * spherical_derivatives['z'][idx]

                if orders > 2:
                    spatial_derivatives['xxx'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xxx'] + directivity_derivatives[''] * spherical_derivatives['xxx'][idx] + 3 * (directivity_derivatives['xx'] * spherical_derivatives['x'][idx] + spherical_derivatives['xx'][idx] * directivity_derivatives['x'])
                    spatial_derivatives['yyy'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['yyy'] + directivity_derivatives[''] * spherical_derivatives['yyy'][idx] + 3 * (directivity_derivatives['yy'] * spherical_derivatives['y'][idx] + spherical_derivatives['yy'][idx] * directivity_derivatives['y'])
                    spatial_derivatives['zzz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['zzz'] + directivity_derivatives[''] * spherical_derivatives['zzz'][idx] + 3 * (directivity_derivatives['zz'] * spherical_derivatives['z'][idx] + spherical_derivatives['zz'][idx] * directivity_derivatives['z'])
                    spatial_derivatives['xxy'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xxy'] + directivity_derivatives[''] * spherical_derivatives['xxy'][idx] + spherical_derivatives['y'][idx] * directivity_derivatives['xx'] + directivity_derivatives['y'] * spherical_derivatives['xx'][idx] + 2 * (spherical_derivatives['x'][idx] * directivity_derivatives['xy'] + directivity_derivatives['x'] * spherical_derivatives['xy'][idx])
                    spatial_derivatives['xxz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['xxz'] + directivity_derivatives[''] * spherical_derivatives['xxz'][idx] + spherical_derivatives['z'][idx] * directivity_derivatives['xx'] + directivity_derivatives['z'] * spherical_derivatives['xx'][idx] + 2 * (spherical_derivatives['x'][idx] * directivity_derivatives['xz'] + directivity_derivatives['x'] * spherical_derivatives['xz'][idx])
                    spatial_derivatives['yyx'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['yyx'] + directivity_derivatives[''] * spherical_derivatives['yyx'][idx] + spherical_derivatives['x'][idx] * directivity_derivatives['yy'] + directivity_derivatives['x'] * spherical_derivatives['yy'][idx] + 2 * (spherical_derivatives['y'][idx] * directivity_derivatives['xy'] + directivity_derivatives['y'] * spherical_derivatives['xy'][idx])
                    spatial_derivatives['yyz'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['yyz'] + directivity_derivatives[''] * spherical_derivatives['yyz'][idx] + spherical_derivatives['z'][idx] * directivity_derivatives['yy'] + directivity_derivatives['z'] * spherical_derivatives['yy'][idx] + 2 * (spherical_derivatives['y'][idx] * directivity_derivatives['yz'] + directivity_derivatives['y'] * spherical_derivatives['yz'][idx])
                    spatial_derivatives['zzx'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['zzx'] + directivity_derivatives[''] * spherical_derivatives['zzx'][idx] + spherical_derivatives['x'][idx] * directivity_derivatives['zz'] + directivity_derivatives['x'] * spherical_derivatives['zz'][idx] + 2 * (spherical_derivatives['z'][idx] * directivity_derivatives['xz'] + directivity_derivatives['z'] * spherical_derivatives['xz'][idx])
                    spatial_derivatives['zzy'][idx] = spherical_derivatives[''][idx] * directivity_derivatives['zzy'] + directivity_derivatives[''] * spherical_derivatives['zzy'][idx] + spherical_derivatives['y'][idx] * directivity_derivatives['zz'] + directivity_derivatives['y'] * spherical_derivatives['zz'][idx] + 2 * (spherical_derivatives['z'][idx] * directivity_derivatives['yz'] + directivity_derivatives['z'] * spherical_derivatives['yz'][idx])
            else:
                spatial_derivatives = spherical_derivatives

        return spatial_derivatives
