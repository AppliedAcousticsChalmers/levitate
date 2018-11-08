import numpy as np
from . import num_spatial_derivatives


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

    def __init__(self, transducer_positions, transducer_normals,
                 freq=40e3, transducer_model=None, transducer_size=10e-3, **kwargs
                 ):
        self.transducer_size = transducer_size

        if transducer_model is None:
            from .transducers import TransducerModel
            self.transducer_model = TransducerModel(freq=freq)
        elif type(transducer_model) is type:
            self.transducer_model = transducer_model(freq=freq)
        else:
            self.transducer_model = transducer_model

        # if not hasattr(shape, '__len__') or len(shape) == 1:
            # self.shape = (shape, shape)
        # else:
            # self.shape = shape
        # if grid is None:
            # self.transducer_positions, self.transducer_normals = rectangular_grid(self.shape, self.transducer_size)
        # else:
            # self.transducer_positions, self.transducer_normals = grid
        self.transducer_positions = transducer_positions
        self.num_transducers = self.transducer_positions.shape[0]
        if transducer_normals.ndim == 1:
            transducer_normals = np.tile(transducer_normals, (self.num_transducers, 1))
        self.transducer_normals = transducer_normals
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

    def signature(self, focus, phases=None):
        if phases is None:
            phases = self.phases
        focus_phases = self.focus_phases(focus)
        return np.mod(phases - focus_phases + np.pi, 2 * np.pi) - np.pi

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


class RectangularArray(TransducerArray):
    def __init__(self, shape=16, spread=10e-3, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        positions, normals = self.grid_generator(shape=shape, spread=spread, offset=offset, normal=normal, rotation=rotation, **kwargs)
        kwargs.setdefault('transducer_size', spread)
        super().__init__(positions, normals, **kwargs)

    @classmethod
    def grid_generator(cls, shape=None, spread=None, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
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
        if not hasattr(shape, '__len__') or len(shape) == 1:
            shape = (shape, shape)
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
            A = self.num_transducers * self.transducer_size**2
            radius = (A / 2 / np.pi)**0.5

        signature = np.empty(self.num_transducers)
        for idx in range(self.num_transducers):
            if np.sum((self.transducer_positions[idx, 0:2] - position)**2)**0.5 > radius:
                signature[idx] = np.pi
            else:
                signature[idx] = 0
        return signature


class DoublesidedArray:
    def __new__(cls, ctype, *args, **kwargs):
        obj = ctype.__new__(ctype)
        obj.__class__ = type('Doublesided{}'.format(ctype.__name__), (DoublesidedArray, ctype), {})
        return obj

    def __init__(self, ctype, separation, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        # positions, normals = self.doublesided_generator(separation, offset=offset, normal=normal, rotation=rotation, **kwargs)
        super().__init__(separation=separation, offset=offset, normal=normal, rotation=rotation, **kwargs)
        # TransducerArray.__init__(self, positions, normals, **kwargs)

    @classmethod
    def grid_generator(cls, separation=None, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
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

        pos_1, norm_1 = super().grid_generator(offset=offset - 0.5 * separation * normal, normal=normal, rotation=rotation, **kwargs)
        pos_2, norm_2 = super().grid_generator(offset=offset + 0.5 * separation * normal, normal=-normal, rotation=-rotation, **kwargs)
        return np.concatenate([pos_1, pos_2], axis=0), np.concatenate([norm_1, norm_2], axis=0)
