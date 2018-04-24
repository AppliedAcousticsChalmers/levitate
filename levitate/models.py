import numpy as np
import logging
from scipy.special import j0, j1

logger = logging.getLogger(__name__)

def rectangular_grid(shape, spread, offset=(0,0,0), normal=(0,0,1), rotation=0):
    """ Creates a grid with positions and normals

    Defines the locations and normals of elements (transducers) in an array.
    For rotated arrays, the rotations is a follows:
        1) A grid of the correct layout is crated in the xy-plane
        2) The grid is rotated to the disired plane, as defined by the normal.
        3) The grid is rotated around the normal.
    The rotation to the disired plane is arount the line where the desired
    plane intersects with the xy-plane.

    Parameters
    ----------
    shape : (int, int)
        The number of grid points in each dimention.
    spread : float
        The separation between grid points, in meters.
    offset : 3 element array_like, optional, default (0,0,0).
        The location of the middle of the array, in meters. 
    normal : 3 element array_like, optional, default (0,0,1).
        The normal direction of the resulting array.
    rotation : float, optional, default 0.
        The in-plane rotation of the array.

    Returns
    -------
    positions : ndarray
        nx3 array with the positions of the elements.
    normals : ndarray
        nx3 array with normals of tge elements.
    
    """
    normal = np.asarray(normal, dtype='float64')
    normal /= (normal**2).sum()**0.5
    x = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0]) * spread
    y = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1]) * spread

    X,Y,Z = np.meshgrid(x,y,0)
    positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    normals = np.tile(normal, (positions.shape[0], 1))

    
    if normal[0] != 0 or normal[1] != 0:
        # We need to rotate the grid to get the correct normal
        rotation_vector = np.cross(normal, (0,0,1))
        rotation_vector /= (rotation_vector**2).sum()**0.5
        cross_product_matrix = np.array([[0, -rotation_vector[2], rotation_vector[1]], 
                                         [rotation_vector[2], 0, -rotation_vector[0]], 
                                         [-rotation_vector[1], rotation_vector[0], 0]])
        cos = normal[2]
        sin = (1-cos**2)**0.5
        rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1-cos) * np.outer(rotation_vector, rotation_vector))
    else:
        rotation_matrix = np.eye(3)    
    if rotation != 0:
        cross_product_matrix = np.array([[0, -normal[2], normal[1]], 
                                         [normal[2], 0, -normal[0]], 
                                         [-normal[1], normal[0], 0]])
        cos = np.cos(-rotation)
        sin = np.sin(-rotation)
        rotation_matrix = rotation_matrix.dot(cos * np.eye(3) + sin * cross_product_matrix + (1-cos) * np.outer(normal, normal))

    positions = positions.dot(rotation_matrix) + offset
    return positions, normals

def double_sided_grid(shape, spread, separation, offset=(0,0,0), normal=(0,0,1), rotation=0, grid_generator=rectangular_grid, **kwargs):
    normal = np.asarray(normal, dtype='float64')
    normal /= (normal**2).sum()**0.5

    pos_1, norm_1 = grid_generator(shape=shape, spread=spread, offset=offset, normal=normal, rotation=rotation, **kwargs)
    pos_2, norm_2 = grid_generator(shape=shape, spread=spread, offset=offset + separation * normal, normal=-normal, rotation=-rotation, **kwargs)
    return np.concatenate([pos_1, pos_2], axis=0), np.concatenate([norm_1, norm_2], axis=0)



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
            phase[idx] = -np.sum((self.transducer_positions[idx, :] - focus)**2)**0.5 * self.k
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
            if np.sum((self.transducer_positions[idx, 0:2])**2)**0.5 > radius:
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

        # These three lines are benchmarked with 20100 receiver positions
        # einsum is twice as fast for large matrices e.g. difference, but slower for small e.g. source_normal
        dots = difference.dot(source_normal)
        norm1 = np.sum(source_normal**2)**0.5
        norm2 = np.einsum('...i,...i', difference, difference)**0.5
        cos_angle = dots / norm2 / norm1
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.transducer_size / 2

        if self.use_directivity.lower() == 'j0':
            # Circular postion in baffle?
            return j0(ka * sin_angle)
        if self.use_directivity.lower() == 'j1':
            # Circular piston in baffle, version 2
            #  TODO: Check this formula!
            # TODO: Needs to ignore warning as well!
            # vals = 2 * jn(1, k_a_sin) / k_a_sin
            # vals[np.isnan(vals)] = 1
            with np.errstate(invalid='ignore'):
                denom = ka * sin_angle
                numer = j1(denom)
                return np.where(denom == 0, 1, 2 * numer / denom)
                # return np.where(sin_angle == 0, 1, 2 * jn(1, ka * sin_angle) / (ka * sin_angle))

        # If no match in the implemented directivities, use omnidirectional
        # TODO: Add a warning?
        assert False
        self.use_directivity = None
        return self.directivity(transducer_id, receiver_position)

    def spherical_spreading(self, transducer_id, receiver_position):
        source_position = self.transducer_positions[transducer_id]
        diff = source_position - receiver_position
        dist = np.einsum('...i,...i', diff, diff)**0.5
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
            r = np.sum(difference**2)**0.5
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
