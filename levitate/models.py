import numpy as np
import logging
from scipy.special import j0, j1
import warnings
warnings.filterwarnings('default', category=DeprecationWarning, module='levitate.models')

logger = logging.getLogger(__name__)

c_air = 343
rho_air = 1.2


def update_air_properties(temperature=None, pressure=None):
    """ Updates the module level air properties

    Will update the module properties `c_air` and `rho_air` according to the
    temperature and static pressure given.

    Parameters
    ----------
    temperature : float
        The current temperature of the air, in degrees Celcius.
    pressure : float
        The ambient hydrostatic pressure, in Pascals.

    Note
    ----
    This should be called immidiately after module import, since changes
    will not propagate to previously created objects.

    """
    R_spec = 287.058
    gamma = 1.4
    if temperature is not None:
        globals()['c_air'] = (gamma * R_spec * (temperature + 273.15))**0.5
    if pressure is not None:
        globals()['rho_air'] = pressure / R_spec / (temperature + 273.15)


def rectangular_grid(shape=None, spread=None, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
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
    positions : numpy.ndarray
        nx3 array with the positions of the elements.
    normals : numpy.ndarray
        nx3 array with normals of the elements.
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


def double_sided_grid(separation=0, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, grid_generator=rectangular_grid, **kwargs):
    """ Creates a double sided transducer grid

    Parameters
    ----------
    separation : float
        The distance between the two halves, along the normal.
    offset : array_like, 3 elements
        The placement of the center of the first half.
    normal : array_like, 3 elements
        The normal of the first half.
    grid_generator : callable
        A callable which should return a tuple (positions, normals) for a single sided grid
    **kwargs
        All arguments will be passed to the generator

    Returns
    -------
    positions : numpy.ndarray
        nx3 array with the positions of the elements.
    normals : numpy.ndarray
        nx3 array with normals of tge elements.
    """
    normal = np.asarray(normal, dtype='float64')
    normal /= (normal**2).sum()**0.5

    pos_1, norm_1 = grid_generator(offset=offset, normal=normal, rotation=rotation, **kwargs)
    pos_2, norm_2 = grid_generator(offset=offset + separation * normal, normal=-normal, rotation=-rotation, **kwargs)
    return np.concatenate([pos_1, pos_2], axis=0), np.concatenate([norm_1, norm_2], axis=0)


spatial_derivative_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy']
num_spatial_derivatives = [1, 4, 10, 19]


class TransducerModel:
    """ Base class for ultrasonic single frequency transducers

    Parameters
    ----------
    freq : float
        The resonant frequency of the transducer.
    p0 : float
        The sound pressure crated at maximum amplitude at 1m distance, in Pa.
    **kwargs
        All remaining arguments will be used as additional properties for the object.

    Attributes
    ----------
    k : float
        Wavenumber in air
    wavelength : float
        Wavelength in air
    omega : float
        Angular frequency
    freq : float
        Wave frequency

    """

    def __init__(self, freq=40e3, p0=6, **kwargs):
        self.freq = freq
        self.p0 = p0
        # The murata transducers are measured to 85 dB SPL at 1 V at 1 m, which corresponds to ~6 Pa at 20 V
        # The datasheet specifies 120 dB SPL @ 0.3 m, which corresponds to ~6 Pa @ 1 m
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        """ Evaluates the transducer radiation

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        source_normal : numpy.ndarray
            The look direction of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the radiation. The last dimention must have length 3 and represent the coordinates of the points.

        Returns
        -------
            out : numpy.ndarray
                The pressure at the locations, assuming `p0` as the source strength.
                Has the same shape as `receiver_position` with the last axis removed.
        """
        return self.p0 * self.spherical_spreading(source_position, receiver_position) * self.directivity(source_position, source_normal, receiver_position)

    def spherical_spreading(self, source_position, receiver_position):
        """ Evaluates spherical wavefronts

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate. The last dimention must have length 3 and represent the coordinates of the points.

        Returns
        -------
            out : numpy.ndarray
                The amplitude and phase of the wavefront, assuming 1Pa at 1m distance,
                phase referenced to the transducer center.
                Has the same shape as `receiver_position` with the last axis removed.
        """
        diff = receiver_position - source_position
        distance = np.einsum('...i,...i', diff, diff)**0.5
        return np.exp(1j * self.k * distance) / distance

    def directivity(self, source_position, source_normal, receiver_position):
        """ Evaluates transducer directivity

        Subclasses will preferrably implement this to create new directivity models.
        Default implementation is omnidirectional sources.

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        source_normal : numpy.ndarray
            The look direction of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the directivity. The last dimention must have length 3 and represent the coordinates of the points.

        Returns
        -------
            out : numpy.ndarray
                The amplitude (and phase) of the directivity, assuming 1Pa at 1m distance,
                phase referenced to the transducer center.
                Has the same shape as `receiver_position` with the last axis removed.
        """
        return np.ones(receiver_position.shape[:-1])

    def spatial_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        """ Calculates the spatial derivatives of the greens function

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        source_normal : numpy.ndarray
            The look direction of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the derivatives. The last dimention must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape (M,...) where M is the number of spatial
            derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`, and the remaining
            dimentions are the same as the `receiver_position` input with the last dimention removed.
        """
        spherical_derivatives = self.spherical_derivatives(source_position, receiver_position, orders)
        directivity_derivatives = self.directivity_derivatives(source_position, source_normal, receiver_position, orders)

        derivatives = np.empty((num_spatial_derivatives[orders],) + receiver_position.shape[:-1], dtype=np.complex128)
        derivatives[0] = spherical_derivatives[0] * directivity_derivatives[0]

        if orders > 0:
            derivatives[1] = spherical_derivatives[0] * directivity_derivatives[1] + directivity_derivatives[0] * spherical_derivatives[1]
            derivatives[2] = spherical_derivatives[0] * directivity_derivatives[2] + directivity_derivatives[0] * spherical_derivatives[2]
            derivatives[3] = spherical_derivatives[0] * directivity_derivatives[3] + directivity_derivatives[0] * spherical_derivatives[3]

        if orders > 1:
            derivatives[4] = spherical_derivatives[0] * directivity_derivatives[4] + directivity_derivatives[0] * spherical_derivatives[4] + 2 * directivity_derivatives[1] * spherical_derivatives[1]
            derivatives[5] = spherical_derivatives[0] * directivity_derivatives[5] + directivity_derivatives[0] * spherical_derivatives[5] + 2 * directivity_derivatives[2] * spherical_derivatives[2]
            derivatives[6] = spherical_derivatives[0] * directivity_derivatives[6] + directivity_derivatives[0] * spherical_derivatives[6] + 2 * directivity_derivatives[3] * spherical_derivatives[3]
            derivatives[7] = spherical_derivatives[0] * directivity_derivatives[7] + directivity_derivatives[0] * spherical_derivatives[7] + spherical_derivatives[1] * directivity_derivatives[2] + directivity_derivatives[1] * spherical_derivatives[2]
            derivatives[8] = spherical_derivatives[0] * directivity_derivatives[8] + directivity_derivatives[0] * spherical_derivatives[8] + spherical_derivatives[1] * directivity_derivatives[3] + directivity_derivatives[1] * spherical_derivatives[3]
            derivatives[9] = spherical_derivatives[0] * directivity_derivatives[9] + directivity_derivatives[0] * spherical_derivatives[9] + spherical_derivatives[2] * directivity_derivatives[3] + directivity_derivatives[2] * spherical_derivatives[3]

        if orders > 2:
            derivatives[10] = spherical_derivatives[0] * directivity_derivatives[10] + directivity_derivatives[0] * spherical_derivatives[10] + 3 * (directivity_derivatives[4] * spherical_derivatives[1] + spherical_derivatives[4] * directivity_derivatives[1])
            derivatives[11] = spherical_derivatives[0] * directivity_derivatives[11] + directivity_derivatives[0] * spherical_derivatives[11] + 3 * (directivity_derivatives[5] * spherical_derivatives[2] + spherical_derivatives[5] * directivity_derivatives[2])
            derivatives[12] = spherical_derivatives[0] * directivity_derivatives[12] + directivity_derivatives[0] * spherical_derivatives[12] + 3 * (directivity_derivatives[6] * spherical_derivatives[3] + spherical_derivatives[6] * directivity_derivatives[3])
            derivatives[13] = spherical_derivatives[0] * directivity_derivatives[13] + directivity_derivatives[0] * spherical_derivatives[13] + spherical_derivatives[2] * directivity_derivatives[4] + directivity_derivatives[2] * spherical_derivatives[4] + 2 * (spherical_derivatives[1] * directivity_derivatives[7] + directivity_derivatives[1] * spherical_derivatives[7])
            derivatives[14] = spherical_derivatives[0] * directivity_derivatives[14] + directivity_derivatives[0] * spherical_derivatives[14] + spherical_derivatives[3] * directivity_derivatives[4] + directivity_derivatives[3] * spherical_derivatives[4] + 2 * (spherical_derivatives[1] * directivity_derivatives[8] + directivity_derivatives[1] * spherical_derivatives[8])
            derivatives[15] = spherical_derivatives[0] * directivity_derivatives[15] + directivity_derivatives[0] * spherical_derivatives[15] + spherical_derivatives[1] * directivity_derivatives[5] + directivity_derivatives[1] * spherical_derivatives[5] + 2 * (spherical_derivatives[2] * directivity_derivatives[7] + directivity_derivatives[2] * spherical_derivatives[7])
            derivatives[16] = spherical_derivatives[0] * directivity_derivatives[16] + directivity_derivatives[0] * spherical_derivatives[16] + spherical_derivatives[3] * directivity_derivatives[5] + directivity_derivatives[3] * spherical_derivatives[5] + 2 * (spherical_derivatives[2] * directivity_derivatives[9] + directivity_derivatives[2] * spherical_derivatives[9])
            derivatives[17] = spherical_derivatives[0] * directivity_derivatives[17] + directivity_derivatives[0] * spherical_derivatives[17] + spherical_derivatives[1] * directivity_derivatives[6] + directivity_derivatives[1] * spherical_derivatives[6] + 2 * (spherical_derivatives[3] * directivity_derivatives[8] + directivity_derivatives[3] * spherical_derivatives[8])
            derivatives[18] = spherical_derivatives[0] * directivity_derivatives[18] + directivity_derivatives[0] * spherical_derivatives[18] + spherical_derivatives[2] * directivity_derivatives[6] + directivity_derivatives[2] * spherical_derivatives[6] + 2 * (spherical_derivatives[3] * directivity_derivatives[9] + directivity_derivatives[3] * spherical_derivatives[9])

        derivatives *= self.p0
        return derivatives

    def spherical_derivatives(self, source_position, receiver_position, orders=3):
        """ Calculates the spatial derivatives of the spherical spreading

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the derivatives. The last dimention must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape (M,...) where M is the number of spatial
            derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`, and the remaining
            dimentions are the same as the `receiver_position` input with the last dimention removed.

        """
        diff = np.moveaxis(receiver_position - source_position, -1, 0)  # Move axis with coordinates to the front to line up with derivatives
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=0)**0.5
        kr = self.k * r
        jkr = 1j * kr
        phase = np.exp(jkr)

        derivatives = np.empty((num_spatial_derivatives[orders],) + receiver_position.shape[:-1], dtype=np.complex128)
        derivatives[0] = phase / r

        if orders > 0:
            coeff = (jkr - 1) * phase / r**3
            derivatives[1] = diff[0] * coeff
            derivatives[2] = diff[1] * coeff
            derivatives[3] = diff[2] * coeff

        if orders > 1:
            coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
            const = (jkr - 1) * phase / r**3
            derivatives[4] = diff[0]**2 * coeff + const
            derivatives[5] = diff[1]**2 * coeff + const
            derivatives[6] = diff[2]**2 * coeff + const
            derivatives[7] = diff[0] * diff[1] * coeff
            derivatives[8] = diff[0] * diff[2] * coeff
            derivatives[9] = diff[1] * diff[2] * coeff

        if orders > 2:
            const = (3 - 3 * jkr - kr**2) * phase / r**5
            coeff = ((jkr - 1) * (15 - kr**2) + 5 * kr**2) * phase / r**7
            derivatives[10] = diff[0] * (3 * const + diff[0]**2 * coeff)
            derivatives[11] = diff[1] * (3 * const + diff[1]**2 * coeff)
            derivatives[12] = diff[2] * (3 * const + diff[2]**2 * coeff)
            derivatives[13] = diff[1] * (const + diff[0]**2 * coeff)
            derivatives[14] = diff[2] * (const + diff[0]**2 * coeff)
            derivatives[15] = diff[0] * (const + diff[1]**2 * coeff)
            derivatives[16] = diff[2] * (const + diff[1]**2 * coeff)
            derivatives[17] = diff[0] * (const + diff[2]**2 * coeff)
            derivatives[18] = diff[1] * (const + diff[2]**2 * coeff)

        return derivatives

    def directivity_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        """ Calculates the spatial derivatives of the directivity

        The default implementation uses finite difference stencils to evaluate the
        derivatives. In principle this means that customised directivity models
        does not need to implement their own derivatives, but can do so for speed
        and precicion benefits.

        Parameters
        ----------
        source_position : numpy.ndarray
            The location of the transducer, as a 3 element array.
        source_normal : numpy.ndarray
            The look direction of the transducer, as a 3 element array.
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the derivatives. The last dimention must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape (M,...) where M is the number of spatial
            derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`, and the remaining
            dimentions are the same as the `receiver_position` input with the last dimention removed.

        """
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

        derivatives = np.empty((num_spatial_derivatives[orders],) + receiver_position.shape[:-1], dtype=np.complex128)
        h = 1 / self.k
        for derivative, (shifts, weights) in finite_difference_coefficients.items():
            derivatives[spatial_derivative_order.index(derivative)] = np.sum(self.directivity(source_position, source_normal, shifts * h + receiver_position[..., np.newaxis, :]) * weights, axis=-1) / h**len(derivative)
        return derivatives


class ReflectingTransducer:
    """ Metaclass for transducers with planar reflectors

    Parameters
    ----------
    ctype : class
        The class implementing the transducer model.
    plane_distance : float
        The distance between the array and the reflector, along the normal.
    plane_normal : array_like
        3 element vector with the plane normal.
    reflection_coefficient : float, complex
        Reflection coefficient to tune the magnitude and phase of the reflection.
    *args
        Passed to ctype initializer
    **kwargs
        Passed to ctype initializer

    Returns
    -------
    obj
        An object of a dynamically created class, inheriting ReflectingTransducer and ctype.

    """
    def __new__(cls, ctype, *args, **kwargs):
        obj = ctype.__new__(ctype)
        obj.__class__ = type('Reflecting{}'.format(ctype.__name__), (ReflectingTransducer, ctype), {})
        return obj

    def __init__(self, ctype, plane_distance, plane_normal=(0, 0, 1), reflection_coefficient=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plane_distance = plane_distance
        self.plane_normal = np.asarray(plane_normal, dtype='float64')
        self.plane_normal /= (self.plane_normal**2).sum()**0.5
        self.reflection_coefficient = reflection_coefficient

    def greens_function(self, source_position, source_normal, receiver_position):
        direct = super().greens_function(source_position, source_normal, receiver_position)
        mirror_position = source_position - 2 * self.plane_normal * ((source_position * self.plane_normal).sum() - self.plane_distance)
        mirror_normal = source_normal - 2 * self.plane_normal * (source_normal * self.plane_normal).sum()
        reflected = super().greens_function(mirror_position, mirror_normal, receiver_position)
        return direct + self.reflection_coefficient * reflected

    def spatial_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        direct = super().spatial_derivatives(source_position, source_normal, receiver_position, orders)
        mirror_position = source_position - 2 * self.plane_normal * ((source_position * self.plane_normal).sum() - self.plane_distance)
        mirror_normal = source_normal - 2 * self.plane_normal * (source_normal * self.plane_normal).sum()
        reflected = super().spatial_derivatives(mirror_position, mirror_normal, receiver_position, orders)
        return direct + self.reflection_coefficient * reflected


class PlaneWaveTransducer(TransducerModel):
    def greens_function(self, source_position, source_normal, receiver_position):
        return self.p0 * np.exp(1j * self.k * np.sum((receiver_position - source_position) * source_normal, axis=-1))

    def spatial_derivatives(self, source_position, source_normal, receiver_position, orders=3):
        source_normal = np.asarray(source_normal, dtype=np.float64)
        source_normal /= (source_normal**2).sum()**0.5
        derivatives = np.empty((num_spatial_derivatives[orders],) + receiver_position.shape[:-1], dtype=np.complex128)
        derivatives[0] = self.greens_function(source_position, source_normal, receiver_position)
        if orders > 0:
            derivatives[1] = 1j * self.k * source_normal[0] * derivatives[0]
            derivatives[2] = 1j * self.k * source_normal[1] * derivatives[0]
            derivatives[3] = 1j * self.k * source_normal[2] * derivatives[0]
        if orders > 1:
            derivatives[4] = 1j * self.k * source_normal[0] * derivatives[3]
            derivatives[5] = 1j * self.k * source_normal[1] * derivatives[2]
            derivatives[6] = 1j * self.k * source_normal[2] * derivatives[3]
            derivatives[7] = 1j * self.k * source_normal[1] * derivatives[1]
            derivatives[8] = 1j * self.k * source_normal[0] * derivatives[3]
            derivatives[9] = 1j * self.k * source_normal[2] * derivatives[2]
        if orders > 2:
            derivatives[10] = 1j * self.k * source_normal[0] * derivatives[4]
            derivatives[11] = 1j * self.k * source_normal[1] * derivatives[5]
            derivatives[12] = 1j * self.k * source_normal[2] * derivatives[6]
            derivatives[13] = 1j * self.k * source_normal[1] * derivatives[4]
            derivatives[14] = 1j * self.k * source_normal[2] * derivatives[4]
            derivatives[15] = 1j * self.k * source_normal[0] * derivatives[5]
            derivatives[16] = 1j * self.k * source_normal[2] * derivatives[5]
            derivatives[17] = 1j * self.k * source_normal[0] * derivatives[6]
            derivatives[18] = 1j * self.k * source_normal[1] * derivatives[6]
        return derivatives


class CircularPiston(TransducerModel):
    """ Circular piston transducer model

    Implementation of the circular piston directivity :math:`D(\\theta) = 2 J_1(ka\\sin\\theta) / (ka\\sin\\theta)`.

    Parameters
    ----------
    effective_radius : float
        The radius :math:`a` in the above.
    **kwargs
        See `TransducerModel`
    """

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
    """ Circular ring transducer model

    Implementation of the circular ring directivity :math:`D(\\theta) = J_0(ka\\sin\\theta)`.

    Parameters
    ----------
    effective_radius : float
        The radius :math:`a` in the above.
    **kwargs
        See `TransducerModel`
    """

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
        diff = np.moveaxis(receiver_position - source_position, -1, 0)  # Move the axis with coordinates to the from to line up with the derivatives
        dot = np.einsum('i...,i...', diff, source_normal)
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=0)**0.5
        n = source_normal
        norm = np.sum(n**2)**0.5
        cos = dot / r / norm
        sin = (1 - cos**2)**0.5
        ka = self.k * self.effective_radius
        ka_sin = ka * sin

        derivatives = np.empty((num_spatial_derivatives[orders],) + receiver_position.shape[:-1], dtype=np.complex128)
        J0 = j0(ka_sin)
        derivatives[0] = J0
        if orders > 0:
            r2 = r**2
            r3 = r**3
            cos_dx = (r2 * n[0] - diff[0] * dot) / r3 / norm
            cos_dy = (r2 * n[1] - diff[1] * dot) / r3 / norm
            cos_dz = (r2 * n[2] - diff[2] * dot) / r3 / norm

            with np.errstate(invalid='ignore'):
                J1_xi = np.where(sin == 0, 0.5, j1(ka_sin) / ka_sin)
            first_order_const = J1_xi * ka**2 * cos
            derivatives[1] = first_order_const * cos_dx
            derivatives[2] = first_order_const * cos_dy
            derivatives[3] = first_order_const * cos_dz

        if orders > 1:
            r5 = r2 * r3
            cos_dx2 = (3 * diff[0]**2 * dot - 2 * diff[0] * n[0] * r2 - dot * r2) / r5 / norm
            cos_dy2 = (3 * diff[1]**2 * dot - 2 * diff[1] * n[1] * r2 - dot * r2) / r5 / norm
            cos_dz2 = (3 * diff[2]**2 * dot - 2 * diff[2] * n[2] * r2 - dot * r2) / r5 / norm
            cos_dxdy = (3 * diff[0] * diff[1] * dot - r2 * (n[0] * diff[1] + n[1] * diff[0])) / r5 / norm
            cos_dxdz = (3 * diff[0] * diff[2] * dot - r2 * (n[0] * diff[2] + n[2] * diff[0])) / r5 / norm
            cos_dydz = (3 * diff[1] * diff[2] * dot - r2 * (n[1] * diff[2] + n[2] * diff[1])) / r5 / norm

            with np.errstate(invalid='ignore'):
                J2_xi2 = np.where(sin == 0, 0.125, (2 * J1_xi - J0) / ka_sin**2)
            second_order_const = J2_xi2 * ka**4 * cos**2 + J1_xi * ka**2
            derivatives[4] = second_order_const * cos_dx**2 + first_order_const * cos_dx2
            derivatives[5] = second_order_const * cos_dy**2 + first_order_const * cos_dy2
            derivatives[6] = second_order_const * cos_dz**2 + first_order_const * cos_dz2
            derivatives[7] = second_order_const * cos_dx * cos_dy + first_order_const * cos_dxdy
            derivatives[8] = second_order_const * cos_dx * cos_dz + first_order_const * cos_dxdz
            derivatives[9] = second_order_const * cos_dy * cos_dz + first_order_const * cos_dydz

        if orders > 2:
            r4 = r2**2
            r7 = r5 * r2
            cos_dx3 = (-15 * diff[0]**3 * dot + 9 * r2 * (diff[0]**2 * n[0] + diff[0] * dot) - 3 * r4 * n[0]) / r7 / norm
            cos_dy3 = (-15 * diff[1]**3 * dot + 9 * r2 * (diff[1]**2 * n[1] + diff[1] * dot) - 3 * r4 * n[1]) / r7 / norm
            cos_dz3 = (-15 * diff[2]**3 * dot + 9 * r2 * (diff[2]**2 * n[2] + diff[2] * dot) - 3 * r4 * n[2]) / r7 / norm
            cos_dx2dy = (-15 * diff[0]**2 * diff[1] * dot + 3 * r2 * (diff[0]**2 * n[1] + 2 * diff[0] * diff[1] * n[0] + diff[1] * dot) - r4 * n[1]) / r7 / norm
            cos_dx2dz = (-15 * diff[0]**2 * diff[2] * dot + 3 * r2 * (diff[0]**2 * n[2] + 2 * diff[0] * diff[2] * n[0] + diff[2] * dot) - r4 * n[2]) / r7 / norm
            cos_dy2dx = (-15 * diff[1]**2 * diff[0] * dot + 3 * r2 * (diff[1]**2 * n[0] + 2 * diff[1] * diff[0] * n[1] + diff[0] * dot) - r4 * n[0]) / r7 / norm
            cos_dy2dz = (-15 * diff[1]**2 * diff[2] * dot + 3 * r2 * (diff[1]**2 * n[2] + 2 * diff[1] * diff[2] * n[1] + diff[2] * dot) - r4 * n[2]) / r7 / norm
            cos_dz2dx = (-15 * diff[2]**2 * diff[0] * dot + 3 * r2 * (diff[2]**2 * n[0] + 2 * diff[2] * diff[0] * n[2] + diff[0] * dot) - r4 * n[0]) / r7 / norm
            cos_dz2dy = (-15 * diff[2]**2 * diff[1] * dot + 3 * r2 * (diff[2]**2 * n[1] + 2 * diff[2] * diff[1] * n[2] + diff[1] * dot) - r4 * n[1]) / r7 / norm

            with np.errstate(invalid='ignore'):
                J3_xi3 = np.where(sin == 0, 1 / 48, (4 * J2_xi2 - J1_xi) / ka_sin**2)
            third_order_const = J3_xi3 * ka**6 * cos**3 + 3 * J2_xi2 * ka**4 * cos
            derivatives[10] = third_order_const * cos_dx**3 + 3 * second_order_const * cos_dx2 * cos_dx + first_order_const * cos_dx3
            derivatives[11] = third_order_const * cos_dy**3 + 3 * second_order_const * cos_dy2 * cos_dy + first_order_const * cos_dy3
            derivatives[12] = third_order_const * cos_dz**3 + 3 * second_order_const * cos_dz2 * cos_dz + first_order_const * cos_dz3
            derivatives[13] = third_order_const * cos_dx**2 * cos_dy + second_order_const * (cos_dx2 * cos_dy + 2 * cos_dxdy * cos_dx) + first_order_const * cos_dx2dy
            derivatives[14] = third_order_const * cos_dx**2 * cos_dz + second_order_const * (cos_dx2 * cos_dz + 2 * cos_dxdz * cos_dx) + first_order_const * cos_dx2dz
            derivatives[15] = third_order_const * cos_dy**2 * cos_dx + second_order_const * (cos_dy2 * cos_dx + 2 * cos_dxdy * cos_dy) + first_order_const * cos_dy2dx
            derivatives[16] = third_order_const * cos_dy**2 * cos_dz + second_order_const * (cos_dy2 * cos_dz + 2 * cos_dydz * cos_dy) + first_order_const * cos_dy2dz
            derivatives[17] = third_order_const * cos_dz**2 * cos_dx + second_order_const * (cos_dz2 * cos_dx + 2 * cos_dxdz * cos_dz) + first_order_const * cos_dz2dx
            derivatives[18] = third_order_const * cos_dz**2 * cos_dy + second_order_const * (cos_dz2 * cos_dy + 2 * cos_dydz * cos_dz) + first_order_const * cos_dz2dy

        return derivatives


class TransducerArray:
    """ Class to handle transducer arrays

    Parameters
    ----------
    freq : float
        The frequency at which to emit
    transducer_model
        An object of `TransducerModel` or a subclass. If passed a class it will instantiate an object with default parameters.
    grid : (numpy.ndarray, numpy.ndarray)
        Tuple of ndarrays to define the transducer layout.
        The first emelent should be the transducer positions, shape Nx3.
        The second emelent should be the transducer normals, shape Nx3.
    transducer_size : float
        Fallback transducer size if no transducer model object is given, or if no grid is given.
    shape : int or (int, int)
        Fallback specificaiton if the transducer grid is not supplied. Assumes a rectangular grid.

    Attributes
    ----------
    phases : numpy.ndarray
        The phases of the transducer elements
    amplitudes : numpy.ndarray
        The amplitudes of the transduder elements
    complex_amplitudes : complex numpy.ndarray
        Transducer controls, complex form
    phases_amplitudes : numpy.ndarray
        Transducer controls, concatenated phase-amplitude form.
        This contains an array with two parts; first all phases, then all amplitudes.
        As a setter this accepts either the concatenated form, complex values,
        or just phases (amplitudes unchanged).
    num_transducers : int
        The number of transducers.
    k : float
        Wavenumber in air
    wavelength : float
        Wavelength in air
    omega : float
        Angular frequency
    freq : float
        Wave frequency

    """

    def __init__(self, freq=40e3, transducer_model=None,
                 grid=None, transducer_size=10e-3, shape=16,
                 ):
        self.transducer_size = transducer_size

        if transducer_model is None:
            self.transducer_model = TransducerModel(freq=freq)
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

    @property
    def complex_amplitudes(self):
        return self.amplitudes * np.exp(1j * self.phases)

    @complex_amplitudes.setter
    def complex_amplitudes(self, value):
        self.amplitudes = np.abs(value)
        self.phases = np.angle(value)

    @property
    def phases_amplitudes(self):
        return np.concatenate((self.phases, self.amplitudes))

    @phases_amplitudes.setter
    def phases_amplitudes(self, value):
        if len(value) == 2 * self.num_transducers:
            self.phases = value[:self.num_transducers]
            self.amplitudes = value[self.num_transducers:]
        elif len(value) == self.num_transducers:
            if np.iscomplexobj(value):
                self.complex_amplitudes = value
            else:
                self.phases = value
        else:
            raise ValueError('Cannot set {} phases and amplitudes with {} values!'.format(self.num_transducers, len(value)))

    def focus_phases(self, focus):
        """ Focuses the phases to create a focus point

        Parameters
        ----------
        focus : array_like
            Three element array with a location where to focus.

        Returns
        -------
        phases : numpy.ndarray
            Array with the phases for the transducer elements.

        """
        phase = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            phase[idx] = -np.sum((self.transducer_positions[idx, :] - focus)**2)**0.5 * self.k
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        return phase

    def twin_signature(self, position=(0, 0), angle=None):
        x = position[0]
        y = position[1]

        if angle is None:
            if np.allclose(x, 0):
                a = 0
                b = 1
            elif np.allclose(y, 0):
                a = 1
                b = 0
            else:
                a = 1 / y
                b = 1 / x
        else:
            cos = np.cos(angle)
            sin = np.sin(angle)
            if np.allclose(cos, 0):
                a = 1
                b = 0
            elif np.allclose(sin, 0):
                a = 0
                b = 1
            else:
                a = 1 / cos
                b = -1 / sin

        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if (self.transducer_positions[idx, 0] - x) * a + (self.transducer_positions[idx, 1] - y) * b > 0:
                signature[idx] = -np.pi / 2
            else:
                signature[idx] = np.pi / 2
        return signature

    def vortex_signature(self, position=(0, 0), angle=0):
        x = position[0]
        y = position[1]
        # TODO: Rotate, shift, and make sure that the calculation below actually works
        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            signature[idx] = np.arctan2(self.transducer_positions[idx, 1], self.transducer_positions[idx, 0])
        return signature

    def bottle_signature(self, position=(0, 0), radius=None):
        position = np.asarray(position)[:2]
        if radius is None:
            A = np.prod(self.shape) * self.transducer_size**2
            radius = (A / 2 / np.pi)**0.5

        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if np.sum((self.transducer_positions[idx, 0:2] - position)**2)**0.5 > radius:
                signature[idx] = np.pi
            else:
                signature[idx] = 0
        return signature

    def signature(self, focus, phases=None):
        if phases is None:
            phases = self.phases
        focus_phases = self.focus_phases(focus)
        return np.mod(phases - focus_phases + np.pi, 2 * np.pi) - np.pi

    def visualize_transducers(self, transducers='all', projection='xy', transducer_size=10e-3,
                              amplitudes=(True, 0, 1), phases=(True, -1, 1), phases_alpha=False,
                              amplitudes_colormap='viridis', phases_colormap='hsv',
                              labels=True, colorbar=True):
        ''' Visualizes the transducer grid and the amplitudes and phases

        Parameters
        ----------
        transducers : string or iterable
            Controlls which transducers should be visualized. Use an iterable
                for explicit controll. The strings 'all' and 'first-half', and
                'last-half' can also be used.
        projection : string
            Specifies how the transducer locations will be projected. One of:
                'xy', 'xz', 'yz', 'yx', 'zx', 'zy', '3d'
        amplitudes : bool, callable, or tuple
            Toggles if the amplitudes should be displayed.
                Pass a callable which will be applied to the amplitudes.
                Pass a tuple `(amplitudes, v_min, v_max)` with `amplitudes` as
                described, `v_min`, `v_max` sets the plot limits.
        phases : bool, callable, or tuple
            Toggles if the phases should be displayed.
                Pass a callable which will be applied to the phases.
                Defaults to normalize the phases by pi.
                Pass a tuple `(phases, v_min, v_max)` with `phases` as
                described, `v_min`, `v_max` sets the plot limits.
        phases_alpha : bool, callable, or tuple
            Toggles if the phases shuld use alpha values from the amplitudes.
                Pass a callable which will be applied to the amplitudes
                to calculate the alpha value.
                Default False, pass True to use the amplitude as alpha.
                Pass a tuple `(phases_alpha, v_min, v_max)` with `phases_alpha`
                as described, `v_min`, `v_max` sets the alpha limits.
        transducer_size : float
            The diameter of the transducers to visualize. Defaults to 10mm.
        amplitudes_colormap: string
            Which matplotlib colormap to use to the amplitude plot. Default 'viridis'.
        phases_colormap: string
            Which matplotlib colormap to use to the phase plot. Default 'hsv'.
        labels: bool
            Toggles if the transducers should be labled in the figure. Default True.
        colorbar: bool
            Toggles if a colorbar should be drawn. Default True.

        '''
        import matplotlib.pyplot as plt
        if transducers == 'all':
            transducers = range(self.num_transducers)
        if transducers == 'first_half':
            transducers = range(int(self.num_transducers / 2))
        if transducers == 'last_half':
            transducers = range(int(self.num_transducers / 2), self.num_transducers)

        # Prepare polygon shape creation
        radius = transducer_size / 2
        num_points = 50  # This is the points per half-circle
        theta = np.concatenate([np.linspace(0, np.pi, num_points), np.linspace(np.pi, 2 * np.pi, num_points)])
        cos, sin = np.cos(theta), np.sin(theta)
        if projection == '3d':
            axes = [0, 1, 2]
            def edge(t_idx):
                pos = self.transducer_positions[t_idx]
                norm = self.transducer_normals[t_idx]
                v1 = np.array([1., 1., 1.])
                v1[2] = -(v1[0] * norm[0] + v1[1] * norm[1]) / norm[2]
                v1 /= np.sqrt(np.sum(v1**2))
                v2 = np.cross(v1, norm)

                v1.shape = (-1, 1)
                v2.shape = (-1, 1)
                return (radius * (cos * v1 + sin * v2) + pos[:, np.newaxis]).T
        else:
            axes = [0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else 3 for ax in projection]

            def edge(t_idx):
                pos = self.transducer_positions[t_idx][axes]
                return pos + radius * np.stack([cos, sin], 1)
        # Calculate the actual polygons
        verts = [edge(t_idx) for t_idx in transducers]

        # Set the max and min of the scales
        try:
            phases, phase_min, phase_max = phases
        except TypeError:
            phase_min, phase_max = -1, 1
        try:
            amplitudes, amplitude_min, amplitude_max = amplitudes
        except TypeError:
            amplitude_min, amplitude_max = 0, 1
        try:
            phases_alpha, phase_alpha_min, phase_alpha_max = phases_alpha
        except TypeError:
            phase_alpha_min, phase_alpha_max = None, None
        phase_norm = plt.Normalize(phase_min, phase_max)
        amplitude_norm = plt.Normalize(amplitude_min, amplitude_max)
        phase_alpha_norm = plt.Normalize(phase_alpha_min, phase_alpha_max, clip=True)

        # Define default plotting scale
        if phases is True:
            def phases(phase): return phase / np.pi
        if amplitudes is True:
            def amplitudes(amplitude): return amplitude
        if phases_alpha is True:
            def phases_alpha(amplitude): return amplitude

        # Create the colors of the polygons
        two_plots = False
        if not amplitudes and not phases:
            colors = ['blue'] * len(verts)
            colorbar = False
        elif not amplitudes:
            colors = plt.get_cmap(phases_colormap)(phase_norm(phases(self.phases[transducers])))
            norm = phase_norm
            colormap = phases_colormap
            if phases_alpha:
                colors[:, 3] = phase_alpha_norm(phases_alpha(self.amplitudes[transducers]))
        elif not phases:
            colors = plt.get_cmap(amplitudes_colormap)(amplitude_norm(amplitudes(self.amplitudes[transducers])))
            norm = amplitude_norm
            colormap = amplitudes_colormap
        else:
            two_plots = True
            colors_phase = plt.get_cmap(phases_colormap)(phase_norm(phases(self.phases[transducers])))
            colors_amplitude = plt.get_cmap(amplitudes_colormap)(amplitude_norm(amplitudes(self.amplitudes[transducers])))
            if phases_alpha:
                colors_phase[:, 3] = phase_alpha_norm(phases_alpha(self.amplitudes[transducers]))

        if projection == '3d':
            # 3D plots
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            if two_plots:
                ax_amplitude = plt.subplot(1, 2, 1, projection='3d')
                ax_phase = plt.subplot(1, 2, 2, projection='3d')
                ax_amplitude.add_collection3d(Poly3DCollection(verts, facecolors=colors_amplitude))
                ax_phase.add_collection3d(Poly3DCollection(verts, facecolors=colors_phase))
                ax = [ax_amplitude, ax_phase]
            else:
                ax = plt.gca(projection='3d')
                ax.add_collection3d(Poly3DCollection(verts, facecolors=colors))
                ax = [ax]
            xlim = np.min(self.transducer_positions[transducers, 0]) - radius, np.max(self.transducer_positions[transducers, 0]) + radius
            ylim = np.min(self.transducer_positions[transducers, 1]) - radius, np.max(self.transducer_positions[transducers, 1]) + radius
            zlim = np.min(self.transducer_positions[transducers, 2]) - radius, np.max(self.transducer_positions[transducers, 2]) + radius
            for a in ax:
                a.set_xlim3d(xlim)
                a.set_ylim3d(ylim)
                a.set_zlim3d(zlim)
        else:
            # 2d plots, will not actually project transducer positons with cosiderations of the orientation.
            from matplotlib.collections import PolyCollection
            ax0_lim = np.min(self.transducer_positions[transducers, axes[0]]) - radius, np.max(self.transducer_positions[transducers, axes[0]]) + radius
            ax1_lim = np.min(self.transducer_positions[transducers, axes[1]]) - radius, np.max(self.transducer_positions[transducers, axes[1]]) + radius
            if two_plots:
                ax_amplitude = plt.subplot(1, 2, 1)
                ax_phase = plt.subplot(1, 2, 2)
                ax_amplitude.add_collection(PolyCollection(verts, facecolors=colors_amplitude))
                ax_phase.add_collection(PolyCollection(verts, facecolors=colors_phase))
                ax = [ax_amplitude, ax_phase]
            else:
                ax = plt.gca()
                ax.add_collection(PolyCollection(verts, facecolors=colors))
                ax = [ax]
            for a in ax:
                a.set_xlim(ax0_lim)
                a.set_ylim(ax1_lim)
                a.axis('scaled')
                a.grid(False)

            # Create colorbars, does not work for 3d plots
            if colorbar:
                if two_plots:
                    sm_amplitude = plt.cm.ScalarMappable(norm=amplitude_norm, cmap=amplitudes_colormap)
                    sm_amplitude.set_array([])
                    plt.colorbar(sm_amplitude, ax=ax_amplitude, orientation='horizontal')
                    sm_phase = plt.cm.ScalarMappable(norm=phase_norm, cmap=phases_colormap)
                    sm_phase.set_array([])
                    plt.colorbar(sm_phase, ax=ax_phase, orientation='horizontal')
                else:
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax[0], orientation='horizontal')

            # label the transducers, does not work for 3d plots
            if labels:
                for a in ax:
                    for t_idx in transducers:
                        pos = self.transducer_positions[t_idx][axes]
                        a.text(*pos, str(t_idx))
        return ax

    def calculate_pressure(self, point, transducer=None):
        """ Calculates the complex pressure amplitude created by the array.

        Parameters
        ----------
        point : numpy.ndarray or tuple
            Either a Nx3 ndarray with [x,y,z] as rows or a tuple with three matrices for x, y, z.
        transducer : int, optional
            Calculate only the pressure for the transducer with this index.
            If None (default) the sum from all transducers is calculated.

        Returns
        -------
        out : numpy.ndarray
            The calculated pressures, on the same form as the input with the last dimention removed
        """
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
            return p.reshape(shape)
        else:
            return p

    def spatial_derivatives(self, receiver_position, orders=3):
        """ Calculates the spatial derivatives for all the transducers

        Parameters
        ----------
        receiver_position : numpy.ndarray
            The location(s) at which to evaluate the derivatives. The last dimention must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape (M, N, ...) M is the number of spatial derivatives,
            where N is the numer of transducers, see `num_spatial_derivatives` and `spatial_derivative_order`,
            and the remaining dimentions are the same as the `receiver_position` input with the last dimention removed.
        """
        derivatives = np.empty((num_spatial_derivatives[orders], self.num_transducers) + receiver_position.shape[:-1], dtype=np.complex128)

        for idx in range(self.num_transducers):
            derivatives[:, idx] = self.transducer_model.spatial_derivatives(self.transducer_positions[idx], self.transducer_normals[idx], receiver_position, orders)
        return derivatives
