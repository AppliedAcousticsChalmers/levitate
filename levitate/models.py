import numpy as np
import logging
from scipy.special import j0, j1
import warnings
warnings.filterwarnings('default', category=DeprecationWarning, module='levitate.models')

logger = logging.getLogger(__name__)

c_air = 343
rho_air = 1.2


def rectangular_grid(shape, spread, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0):
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

    X, Y, Z = np.meshgrid(x, y, 0)
    positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    normals = np.tile(normal, (positions.shape[0], 1))

    if normal[0] != 0 or normal[1] != 0:
        # We need to rotate the grid to get the correct normal
        rotation_vector = np.cross(normal, (0, 0, 1))
        rotation_vector /= (rotation_vector**2).sum()**0.5
        cross_product_matrix = np.array([[0, -rotation_vector[2], rotation_vector[1]],
                                         [rotation_vector[2], 0, -rotation_vector[0]],
                                         [-rotation_vector[1], rotation_vector[0], 0]])
        cos = normal[2]
        sin = (1 - cos**2)**0.5
        rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(rotation_vector, rotation_vector))
    else:
        rotation_matrix = np.eye(3)
    if rotation != 0:
        cross_product_matrix = np.array([[0, -normal[2], normal[1]],
                                         [normal[2], 0, -normal[0]],
                                         [-normal[1], normal[0], 0]])
        cos = np.cos(-rotation)
        sin = np.sin(-rotation)
        rotation_matrix = rotation_matrix.dot(cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(normal, normal))

    positions = positions.dot(rotation_matrix) + offset
    return positions, normals


def double_sided_grid(shape, spread, separation, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, grid_generator=rectangular_grid, **kwargs):
    normal = np.asarray(normal, dtype='float64')
    normal /= (normal**2).sum()**0.5

    pos_1, norm_1 = grid_generator(shape=shape, spread=spread, offset=offset, normal=normal, rotation=rotation, **kwargs)
    pos_2, norm_2 = grid_generator(shape=shape, spread=spread, offset=offset + separation * normal, normal=-normal, rotation=-rotation, **kwargs)
    return np.concatenate([pos_1, pos_2], axis=0), np.concatenate([norm_1, norm_2], axis=0)


class TransducerModel:

    def __init__(self, freq=40e3, effective_radius=None):
        self.freq = freq
        self.effective_radius = effective_radius

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._omega = value * c_air

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value
        self._k = value / c_air

    @property
    def freq(self):
        return self.omega / 2 / np.pi

    @freq.setter
    def freq(self, value):
        self.omega = value * 2 * np.pi

    @property
    def wavelength(self):
        return 2 * np.pi / self.k

    @wavelength.setter
    def wavelength(self, value):
        self.k = 2 * np.pi / value

    def greens_function(self, source_position, source_normal, receiver_position):
        return self.spherical_spreading(source_position, receiver_position) * self.directivity(source_position, source_normal, receiver_position)

    def spherical_spreading(self, source_position, receiver_position):
        diff = receiver_position - source_position
        distance = np.einsum('...i,...i', diff, diff)**0.5
        return np.exp(1j * self.k * distance) / distance

    def directivity(self, source_position, source_normal, receiver_position):
        if receiver_position.ndim == 1:
            return 1
        else:
            return np.ones(receiver_position.shape[:-1])

    def spatial_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        spherical_derivatives = self.spherical_derivatives(source_position, receiver_position, orders)
        directivity_derivatives = self.directivity_derivatives(source_position, source_normal, receiver_position, orders)

        derivatives = {'': spherical_derivatives[''] * directivity_derivatives['']}

        if orders > 0:
            derivatives['x'] = spherical_derivatives[''] * directivity_derivatives['x'] + directivity_derivatives[''] * spherical_derivatives['x']
            derivatives['y'] = spherical_derivatives[''] * directivity_derivatives['y'] + directivity_derivatives[''] * spherical_derivatives['y']
            derivatives['z'] = spherical_derivatives[''] * directivity_derivatives['z'] + directivity_derivatives[''] * spherical_derivatives['z']

        if orders > 1:
            derivatives['xx'] = spherical_derivatives[''] * directivity_derivatives['xx'] + directivity_derivatives[''] * spherical_derivatives['xx'] + 2 * directivity_derivatives['x'] * spherical_derivatives['x']
            derivatives['yy'] = spherical_derivatives[''] * directivity_derivatives['yy'] + directivity_derivatives[''] * spherical_derivatives['yy'] + 2 * directivity_derivatives['y'] * spherical_derivatives['y']
            derivatives['zz'] = spherical_derivatives[''] * directivity_derivatives['zz'] + directivity_derivatives[''] * spherical_derivatives['zz'] + 2 * directivity_derivatives['z'] * spherical_derivatives['z']
            derivatives['xy'] = spherical_derivatives[''] * directivity_derivatives['xy'] + directivity_derivatives[''] * spherical_derivatives['xy'] + spherical_derivatives['x'] * directivity_derivatives['y'] + directivity_derivatives['x'] * spherical_derivatives['y']
            derivatives['xz'] = spherical_derivatives[''] * directivity_derivatives['xz'] + directivity_derivatives[''] * spherical_derivatives['xz'] + spherical_derivatives['x'] * directivity_derivatives['z'] + directivity_derivatives['x'] * spherical_derivatives['z']
            derivatives['yz'] = spherical_derivatives[''] * directivity_derivatives['yz'] + directivity_derivatives[''] * spherical_derivatives['yz'] + spherical_derivatives['y'] * directivity_derivatives['z'] + directivity_derivatives['y'] * spherical_derivatives['z']

        if orders > 2:
            derivatives['xxx'] = spherical_derivatives[''] * directivity_derivatives['xxx'] + directivity_derivatives[''] * spherical_derivatives['xxx'] + 3 * (directivity_derivatives['xx'] * spherical_derivatives['x'] + spherical_derivatives['xx'] * directivity_derivatives['x'])
            derivatives['yyy'] = spherical_derivatives[''] * directivity_derivatives['yyy'] + directivity_derivatives[''] * spherical_derivatives['yyy'] + 3 * (directivity_derivatives['yy'] * spherical_derivatives['y'] + spherical_derivatives['yy'] * directivity_derivatives['y'])
            derivatives['zzz'] = spherical_derivatives[''] * directivity_derivatives['zzz'] + directivity_derivatives[''] * spherical_derivatives['zzz'] + 3 * (directivity_derivatives['zz'] * spherical_derivatives['z'] + spherical_derivatives['zz'] * directivity_derivatives['z'])
            derivatives['xxy'] = spherical_derivatives[''] * directivity_derivatives['xxy'] + directivity_derivatives[''] * spherical_derivatives['xxy'] + spherical_derivatives['y'] * directivity_derivatives['xx'] + directivity_derivatives['y'] * spherical_derivatives['xx'] + 2 * (spherical_derivatives['x'] * directivity_derivatives['xy'] + directivity_derivatives['x'] * spherical_derivatives['xy'])
            derivatives['xxz'] = spherical_derivatives[''] * directivity_derivatives['xxz'] + directivity_derivatives[''] * spherical_derivatives['xxz'] + spherical_derivatives['z'] * directivity_derivatives['xx'] + directivity_derivatives['z'] * spherical_derivatives['xx'] + 2 * (spherical_derivatives['x'] * directivity_derivatives['xz'] + directivity_derivatives['x'] * spherical_derivatives['xz'])
            derivatives['yyx'] = spherical_derivatives[''] * directivity_derivatives['yyx'] + directivity_derivatives[''] * spherical_derivatives['yyx'] + spherical_derivatives['x'] * directivity_derivatives['yy'] + directivity_derivatives['x'] * spherical_derivatives['yy'] + 2 * (spherical_derivatives['y'] * directivity_derivatives['xy'] + directivity_derivatives['y'] * spherical_derivatives['xy'])
            derivatives['yyz'] = spherical_derivatives[''] * directivity_derivatives['yyz'] + directivity_derivatives[''] * spherical_derivatives['yyz'] + spherical_derivatives['z'] * directivity_derivatives['yy'] + directivity_derivatives['z'] * spherical_derivatives['yy'] + 2 * (spherical_derivatives['y'] * directivity_derivatives['yz'] + directivity_derivatives['y'] * spherical_derivatives['yz'])
            derivatives['zzx'] = spherical_derivatives[''] * directivity_derivatives['zzx'] + directivity_derivatives[''] * spherical_derivatives['zzx'] + spherical_derivatives['x'] * directivity_derivatives['zz'] + directivity_derivatives['x'] * spherical_derivatives['zz'] + 2 * (spherical_derivatives['z'] * directivity_derivatives['xz'] + directivity_derivatives['z'] * spherical_derivatives['xz'])
            derivatives['zzy'] = spherical_derivatives[''] * directivity_derivatives['zzy'] + directivity_derivatives[''] * spherical_derivatives['zzy'] + spherical_derivatives['y'] * directivity_derivatives['zz'] + directivity_derivatives['y'] * spherical_derivatives['zz'] + 2 * (spherical_derivatives['z'] * directivity_derivatives['yz'] + directivity_derivatives['z'] * spherical_derivatives['yz'])

        return derivatives

    def spherical_derivatives(self, source_position, receiver_position, orders=3):
        diff = receiver_position - source_position
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=-1)**0.5
        kr = self.k * r
        jkr = 1j * kr
        phase = np.exp(jkr)

        derivatives = {'': phase / r}

        if orders > 0:
            coeff = (jkr - 1) * phase / r**3
            derivatives['x'] = diff[..., 0] * coeff
            derivatives['y'] = diff[..., 1] * coeff
            derivatives['z'] = diff[..., 2] * coeff

        if orders > 1:
            coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
            const = (jkr - 1) * phase / r**3
            derivatives['xx'] = diff[..., 0]**2 * coeff + const
            derivatives['yy'] = diff[..., 1]**2 * coeff + const
            derivatives['zz'] = diff[..., 2]**2 * coeff + const
            derivatives['xy'] = diff[..., 0] * diff[..., 1] * coeff
            derivatives['xz'] = diff[..., 0] * diff[..., 2] * coeff
            derivatives['yz'] = diff[..., 1] * diff[..., 2] * coeff

        if orders > 2:
            const = (3 - 3 * jkr - kr**2) * phase / r**5
            coeff = ((jkr - 1) * (15 - kr**2) + 5 * kr**2) * phase / r**7
            derivatives['xxx'] = diff[..., 0] * (3 * const + diff[..., 0]**2 * coeff)
            derivatives['yyy'] = diff[..., 1] * (3 * const + diff[..., 1]**2 * coeff)
            derivatives['zzz'] = diff[..., 2] * (3 * const + diff[..., 2]**2 * coeff)
            derivatives['xxy'] = diff[..., 1] * (const + diff[..., 0]**2 * coeff)
            derivatives['xxz'] = diff[..., 2] * (const + diff[..., 0]**2 * coeff)
            derivatives['yyx'] = diff[..., 0] * (const + diff[..., 1]**2 * coeff)
            derivatives['yyz'] = diff[..., 2] * (const + diff[..., 1]**2 * coeff)
            derivatives['zzx'] = diff[..., 0] * (const + diff[..., 2]**2 * coeff)
            derivatives['zzy'] = diff[..., 1] * (const + diff[..., 2]**2 * coeff)

        return derivatives

    def directivity_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        finite_difference_coefficients = {'': (np.array([0, 0, 0]), 1)}
        if orders > 0:
            finite_difference_coefficients['x'] = (np.array([[1, 0, 0], [-1, 0, 0]]), [0.5, -0.5])
            finite_difference_coefficients['y'] = (np.array([[0, 1, 0], [0, -1, 0]]), [0.5, -0.5])
            finite_difference_coefficients['z'] = (np.array([[0, 0, 1], [0, 0, -1]]), [0.5, -0.5])
        if orders > 1:
            finite_difference_coefficients['xx'] = (np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]]), [1, -2, 1])  # Alt -- (np.array([[2, 0, 0], [0, 0, 0], [-2, 0, 0]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['yy'] = (np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]), [1, -2, 1])  # Alt-- (np.array([[0, 2, 0], [0, 0, 0], [0, -2, 0]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['zz'] = (np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]]), [1, -2, 1])  # Alt -- (np.array([[0, 0, 2], [0, 0, 0], [0, 0, -2]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['xy'] = (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]), [0.25, 0.25, -0.25, -0.25])
            finite_difference_coefficients['xz'] = (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1]]), [0.25, 0.25, -0.25, -0.25])
            finite_difference_coefficients['yz'] = (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1]]), [0.25, 0.25, -0.25, -0.25])
        if orders > 2:
            finite_difference_coefficients['xxx'] = (np.array([[2, 0, 0], [-2, 0, 0], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -1, 1])  # Alt -- (np.array([[3, 0, 0], [-3, 0, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['yyy'] = (np.array([[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -1, 1])  # Alt -- (np.array([[0, 3, 0], [0, -3, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['zzz'] = (np.array([[0, 0, 2], [0, 0, -2], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -1, 1])  # Alt -- (np.array([[0, 0, 3], [0, 0, -3], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['xxy'] = (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[2, 1, 0], [-2, -1, 0], [2, -1, 0], [-2, 1, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['xxz'] = (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[2, 0, 1], [-2, 0, -1], [2, 0, -1], [-2, 0, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['yyx'] = (np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[1, 2, 0], [-1, -2, 0], [-1, 2, 0], [1, -2, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['yyz'] = (np.array([[0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 0, 1], [0, 0, -1]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[0, 2, 1], [0, -2, -1], [0, 2, -1], [0, -2, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['zzx'] = (np.array([[1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 0], [-1, 0, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[1, 0, 2], [-1, 0, -2], [-1, 0, 2], [1, 0, -2], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['zzy'] = (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 0], [0, -1, 0]]), [0.5, -0.5, -0.5, 0.5, -1, 1])  # Alt -- (np.array([[0, 1, 2], [0, -1, -2], [0, -1, 2], [0, 1, -2], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])

        derivatives = {}
        h = 1 / self.k
        for derivative, (shifts, weights) in finite_difference_coefficients.items():
            derivatives[derivative] = np.sum(self.directivity(source_position, source_normal, shifts * h + receiver_position[..., np.newaxis, :]) * weights, axis=-1) / h**len(derivative)
        return derivatives


class CircularPiston(TransducerModel):
    def __init__(self, effective_radius, freq=40e3):
        self.effective_radius = effective_radius
        self.freq = freq

    def directivity(self, source_position, source_normal, receiver_position):
        diff = receiver_position - source_position
        dots = diff.dot(source_normal)
        norm1 = np.sum(source_normal**2)**0.5
        norm2 = np.einsum('...i,...i', diff, diff)**0.5
        cos_angle = dots / norm2 / norm1
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.effective_radius

        denom = ka * sin_angle
        numer = j1(denom)
        with np.errstate(invalid='ignore'):
            return np.where(denom == 0, 1, 2 * numer / denom)


class CircularRing(TransducerModel):
    def __init__(self, effective_radius, freq=40e3):
        self.effective_radius = effective_radius
        self.freq = freq

    def directivity(self, source_position, source_normal, receiver_position):
        diff = receiver_position - source_position
        dots = diff.dot(source_normal)
        norm1 = np.sum(source_normal**2)**0.5
        norm2 = np.einsum('...i,...i', diff, diff)**0.5
        cos_angle = dots / norm2 / norm1
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.effective_radius
        return j0(ka * sin_angle)

    def directivity_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        diff = receiver_position - source_position
        dot = diff.dot(source_normal)
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=-1)**0.5
        n = source_normal
        norm = np.sum(n**2)**0.5
        cos = dot / r / norm
        sin = (1 - cos**2)**0.5
        ka = self.k * self.effective_radius
        ka_sin = ka * sin

        J0 = j0(ka_sin)
        derivatives = {'': J0}
        if orders > 0:
            r2 = r**2
            r3 = r**3
            cos_dx = (r2 * n[0] - diff[..., 0] * dot) / r3 / norm
            cos_dy = (r2 * n[1] - diff[..., 1] * dot) / r3 / norm
            cos_dz = (r2 * n[2] - diff[..., 2] * dot) / r3 / norm

            with np.errstate(invalid='ignore'):
                J1_xi = np.where(sin == 0, 0.5, j1(ka_sin) / ka_sin)
            first_order_const = J1_xi * ka**2 * cos
            derivatives['x'] = first_order_const * cos_dx
            derivatives['y'] = first_order_const * cos_dy
            derivatives['z'] = first_order_const * cos_dz

        if orders > 1:
            r5 = r2 * r3
            cos_dx2 = (3 * diff[..., 0]**2 * dot - 2 * diff[..., 0] * n[0] * r2 - dot * r2) / r5 / norm
            cos_dy2 = (3 * diff[..., 1]**2 * dot - 2 * diff[..., 1] * n[1] * r2 - dot * r2) / r5 / norm
            cos_dz2 = (3 * diff[..., 2]**2 * dot - 2 * diff[..., 2] * n[2] * r2 - dot * r2) / r5 / norm
            cos_dxdy = (3 * diff[..., 0] * diff[..., 1] * dot - r2 * (n[0] * diff[..., 1] + n[1] * diff[..., 0])) / r5 / norm
            cos_dxdz = (3 * diff[..., 0] * diff[..., 2] * dot - r2 * (n[0] * diff[..., 2] + n[2] * diff[..., 0])) / r5 / norm
            cos_dydz = (3 * diff[..., 1] * diff[..., 2] * dot - r2 * (n[1] * diff[..., 2] + n[2] * diff[..., 1])) / r5 / norm

            with np.errstate(invalid='ignore'):
                J2_xi2 = np.where(sin == 0, 0.125, (2 * J1_xi - J0) / ka_sin**2)
            second_order_const = J2_xi2 * ka**4 * cos**2 + J1_xi * ka**2
            derivatives['xx'] = second_order_const * cos_dx**2 + first_order_const * cos_dx2
            derivatives['yy'] = second_order_const * cos_dy**2 + first_order_const * cos_dy2
            derivatives['zz'] = second_order_const * cos_dz**2 + first_order_const * cos_dz2
            derivatives['xy'] = second_order_const * cos_dx * cos_dy + first_order_const * cos_dxdy
            derivatives['xz'] = second_order_const * cos_dx * cos_dz + first_order_const * cos_dxdz
            derivatives['yz'] = second_order_const * cos_dy * cos_dz + first_order_const * cos_dydz

        if orders > 2:
            r4 = r2**2
            r7 = r5 * r2
            cos_dx3 = (-15 * diff[..., 0]**3 * dot + 9 * r2 * (diff[..., 0]**2 * n[0] + diff[..., 0] * dot) - 3 * r4 * n[0]) / r7 / norm
            cos_dy3 = (-15 * diff[..., 1]**3 * dot + 9 * r2 * (diff[..., 1]**2 * n[1] + diff[..., 1] * dot) - 3 * r4 * n[1]) / r7 / norm
            cos_dz3 = (-15 * diff[..., 2]**3 * dot + 9 * r2 * (diff[..., 2]**2 * n[2] + diff[..., 2] * dot) - 3 * r4 * n[2]) / r7 / norm
            cos_dx2dy = (-15 * diff[..., 0]**2 * diff[..., 1] * dot + 3 * r2 * (diff[..., 0]**2 * n[1] + 2 * diff[..., 0] * diff[..., 1] * n[0] + diff[..., 1] * dot) - r4 * n[1]) / r7 / norm
            cos_dx2dz = (-15 * diff[..., 0]**2 * diff[..., 2] * dot + 3 * r2 * (diff[..., 0]**2 * n[2] + 2 * diff[..., 0] * diff[..., 2] * n[0] + diff[..., 2] * dot) - r4 * n[2]) / r7 / norm
            cos_dy2dx = (-15 * diff[..., 1]**2 * diff[..., 0] * dot + 3 * r2 * (diff[..., 1]**2 * n[0] + 2 * diff[..., 1] * diff[..., 0] * n[1] + diff[..., 0] * dot) - r4 * n[0]) / r7 / norm
            cos_dy2dz = (-15 * diff[..., 1]**2 * diff[..., 2] * dot + 3 * r2 * (diff[..., 1]**2 * n[2] + 2 * diff[..., 1] * diff[..., 2] * n[1] + diff[..., 2] * dot) - r4 * n[2]) / r7 / norm
            cos_dz2dx = (-15 * diff[..., 2]**2 * diff[..., 0] * dot + 3 * r2 * (diff[..., 2]**2 * n[0] + 2 * diff[..., 2] * diff[..., 0] * n[2] + diff[..., 0] * dot) - r4 * n[0]) / r7 / norm
            cos_dz2dy = (-15 * diff[..., 2]**2 * diff[..., 1] * dot + 3 * r2 * (diff[..., 2]**2 * n[1] + 2 * diff[..., 2] * diff[..., 1] * n[2] + diff[..., 1] * dot) - r4 * n[1]) / r7 / norm

            with np.errstate(invalid='ignore'):
                J3_xi3 = np.where(sin == 0, 1 / 48, (4 * J2_xi2 - J1_xi) / ka_sin**2)
            third_order_const = J3_xi3 * ka**6 * cos**3 + 3 * J2_xi2 * ka**4 * cos
            derivatives['xxx'] = third_order_const * cos_dx**3 + 3 * second_order_const * cos_dx2 * cos_dx + first_order_const * cos_dx3
            derivatives['yyy'] = third_order_const * cos_dy**3 + 3 * second_order_const * cos_dy2 * cos_dy + first_order_const * cos_dy3
            derivatives['zzz'] = third_order_const * cos_dz**3 + 3 * second_order_const * cos_dz2 * cos_dz + first_order_const * cos_dz3
            derivatives['xxy'] = third_order_const * cos_dx**2 * cos_dy + second_order_const * (cos_dx2 * cos_dy + 2 * cos_dxdy * cos_dx) + first_order_const * cos_dx2dy
            derivatives['xxz'] = third_order_const * cos_dx**2 * cos_dz + second_order_const * (cos_dx2 * cos_dz + 2 * cos_dxdz * cos_dx) + first_order_const * cos_dx2dz
            derivatives['yyx'] = third_order_const * cos_dy**2 * cos_dx + second_order_const * (cos_dy2 * cos_dx + 2 * cos_dxdy * cos_dy) + first_order_const * cos_dy2dx
            derivatives['yyz'] = third_order_const * cos_dy**2 * cos_dz + second_order_const * (cos_dy2 * cos_dz + 2 * cos_dydz * cos_dy) + first_order_const * cos_dy2dz
            derivatives['zzx'] = third_order_const * cos_dz**2 * cos_dx + second_order_const * (cos_dz2 * cos_dx + 2 * cos_dxdz * cos_dz) + first_order_const * cos_dz2dx
            derivatives['zzy'] = third_order_const * cos_dz**2 * cos_dy + second_order_const * (cos_dz2 * cos_dy + 2 * cos_dydz * cos_dz) + first_order_const * cos_dz2dy

        return derivatives


class TransducerArray:

    def __init__(self, focus_point=[0, 0, 0.2], freq=40e3,
                 grid=None, transducer_size=10e-3, shape=16,
                 transducer_model=None, directivity=None):
        self.focus_point = focus_point
        self.transducer_size = transducer_size

        if transducer_model is None:
            self.transducer_model = TransducerModel(freq=freq)
            if directivity is not None:
                warnings.warn(('Paramater `directivity` of TransducerArray is not recommended. '
                               'Create and set a transducer model directly instead.'),
                              DeprecationWarning, stacklevel=2)
                self.use_directivity = directivity
        elif type(transducer_model) is type:
            self.transducer_model = transducer_model(freq=freq, effective_radius=transducer_size / 2)
        else:
            self.transducer_model = transducer_model

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

    @property
    def k(self):
        return self.transducer_model.k

    @k.setter
    def k(self, value):
        self.transducer_model.k = value

    @property
    def omega(self):
        return self.transducer_model.omega

    @omega.setter
    def omega(self, value):
        self.transducer_model.omega = value

    @property
    def freq(self):
        return self.transducer_model.freq

    @freq.setter
    def freq(self, value):
        self.transducer_model.freq = value

    @property
    def wavelength(self):
        return self.transducer_model.wavelength

    @wavelength.setter
    def wavelength(self, value):
        self.transducer_model.wavelength = value

    # TODO: Legacy glue to change directivities
    @property
    def use_directivity(self):
        warnings.warn(('`use_directivity` of TransducerArray is not recommended. '
                       'Interact with `transducer_model` directly instead.'),
                      DeprecationWarning, stacklevel=2)
        if type(self.transducer_model) == CircularRing:
            return 'j0'
        if type(self.transducer_model) == CircularPiston:
            return 'j1'
        if type(self.transducer_model) == TransducerModel:
            return None
        else:
            return 'Unknown transducer model `{}`'.format(type(self.transducer_model))

    @use_directivity.setter
    def use_directivity(self, value):
        warnings.warn(('`use_directivity` of TransducerArray is not recommended. '
                       'Create and set a transducer model directly instead.'),
                      DeprecationWarning, stacklevel=2)
        freq = self.freq
        if value is None:
            self.transducer_model = TransducerModel(freq=freq)
            self._use_directivity = None
        elif value == 'j0':
            self.transducer_model = CircularRing(effective_radius=self.transducer_size / 2, freq=freq)
            self._use_directivity = 'j0'
        elif value == 'j1':
            self.transducer_model = CircularPiston(effective_radius=self.transducer_size / 2, freq=freq)
            self._use_directivity = 'j1'
        else:
            raise ValueError("Unknown dirictivity '{}'".format(value))

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
            A = np.prod(self.shape) * self.transducer_size**2
            radius = (A / 2 / np.pi)**0.5

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
                p += self.amplitudes[idx] * np.exp(1j * self.phases[idx]) * self.transducer_model.greens_function(
                    self.transducer_positions[idx], self.transducer_normals[idx], point)
        else:
            p = self.amplitudes[transducer] * np.exp(1j * self.phases[transducer]) * self.transducer_model.greens_function(
                    self.transducer_positions[transducer], self.transducer_normals[transducer], point)

        if reshape:
            return self.p0 * p.reshape(shape)
        else:
            return self.p0 * p

    def directivity(self, transducer_id, receiver_position):
        warnings.warn(('`directivity` of TransducerArray is not recommended. '
                       'Use the corresponding method of a transducer model instead.'),
                      DeprecationWarning, stacklevel=2)
        return self.transducer_model.directivity(self.transducer_positions[transducer_id], self.transducer_normals[transducer_id], receiver_position)
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
        warnings.warn(('`spherical_spreading` of TransducerArray is not recommended. '
                       'Use the corresponding method of a transducer model instead.'),
                      DeprecationWarning, stacklevel=2)
        return self.transducer_model.spherical_spreading(self.transducer_positions[transducer_id], receiver_position)
        source_position = self.transducer_positions[transducer_id]
        diff = source_position - receiver_position
        dist = np.einsum('...i,...i', diff, diff)**0.5
        return 1 / dist * np.exp(1j * self.k * dist)

    def greens_function(self, transducer_id, receiver_position):
        warnings.warn(('`greens_function` of TransducerArray is not recommended. '
                       'Use the corresponding method of a transducer model instead.'),
                      DeprecationWarning, stacklevel=2)
        return self.transducer_model.greens_function(
            self.transducer_positions[transducer_id],
            self.transducer_normals[transducer_id],
            receiver_position)
        directional_part = self.directivity(transducer_id, receiver_position)
        spherical_part = self.spherical_spreading(transducer_id, receiver_position)
        return directional_part * spherical_part

    def spatial_derivatives(self, focus, h=None, orders=3):
        if np.ndim(focus) == 1:
            # Single point
            shape = self.num_transducers
        elif np.ndim(focus) == 2:
            # Array/list of points
            shape = (np.shape(focus)[0], self.num_transducers)

        derivatives = {'': np.empty(shape, complex)}
        if orders > 0:
            derivatives['x'] = np.empty(shape, complex)
            derivatives['y'] = np.empty(shape, complex)
            derivatives['z'] = np.empty(shape, complex)
        if orders > 1:
            derivatives['xx'] = np.empty(shape, complex)
            derivatives['yy'] = np.empty(shape, complex)
            derivatives['zz'] = np.empty(shape, complex)
            derivatives['xy'] = np.empty(shape, complex)
            derivatives['xz'] = np.empty(shape, complex)
            derivatives['yz'] = np.empty(shape, complex)
        if orders > 2:
            derivatives['xxx'] = np.empty(shape, complex)
            derivatives['yyy'] = np.empty(shape, complex)
            derivatives['zzz'] = np.empty(shape, complex)
            derivatives['xxy'] = np.empty(shape, complex)
            derivatives['xxz'] = np.empty(shape, complex)
            derivatives['yyx'] = np.empty(shape, complex)
            derivatives['yyz'] = np.empty(shape, complex)
            derivatives['zzx'] = np.empty(shape, complex)
            derivatives['zzy'] = np.empty(shape, complex)

        for idx in range(self.num_transducers):
            transducer_derivatives = self.transducer_model.spatial_derivatives(self.transducer_positions[idx], self.transducer_normals[idx], focus, orders)
            for key in derivatives:
                derivatives[key][..., idx] = transducer_derivatives[key]
        return derivatives

    def old_spatial_derivatives(self, focus, h=None, orders=3):
        '''
        Calculate and set the spatial derivatives for each transducer.
        These are the same regardless of the amplitude and phase of the transducers,
        and remains constant throughout the optimization.
        '''
        # Pre-initialize dictionary with arrays
        # TODO: enable selective calculation of the derivatives actually needed
        warnings.warn(('`old_spatial_derivatives` is an old implementation to caculate spatial derivatives. '
                       'Use the new method to avoid issues.'),
                      DeprecationWarning, stacklevel=2)
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

        spatial_derivatives = {}
        for key in spherical_derivatives.keys():
            spatial_derivatives[key] = np.empty(num_trans, complex)
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
                directivity_derivatives = {}
                for key in spherical_derivatives.keys():
                    shifts, weights = finite_difference_coefficients[key]
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
