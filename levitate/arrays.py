"""Handling of transducer arrays.

The primary base class is the `TransducerArray` class, which contains the most
frequently used methods.
"""

import numpy as np
from . import num_spatial_derivatives
from .visualize import Visualizer
from .materials import Air


class TransducerArray:
    """Base class to handle transducer arrays.

    This class has no notion of the layout. If possible, try to use a more specific
    implementation instead.

    Parameters
    ----------
    transducer_positions : numpy.ndarray
        The positions of the transducer elements in the array, shape 3xN.
    transducer_normals : numpy.ndarray
        The normals of the transducer elements in the array, shape 3xN.
    transducer_model
        An object of `levitate.transducers.TransducerModel` or a subclass. If passed a class it will create a new instance.
    transducer_size : float
        Fallback transducer size if no transducer model object is given, or if no grid is given.
    transducer_kwargs : dict
        Extra keyword arguments used when instantiating a new transducer model.

    Attributes
    ----------
    phases : numpy.ndarray
        The phases of the transducer elements.
    amplitudes : numpy.ndarray
        The amplitudes of the transducer elements.
    complex_amplitudes : numpy.ndarray
        Transducer element controls on complex form.
    num_transducers : int
        The number of transducers used.
    transducer_positions : numpy.ndarray
        As above.
    transducer_normals : numpy.ndarray
        As above.
    transducer_model : `levitate.transducers.TransducerModel`
        An instance of a specific transducer model implementation.
    calculate : PersistentFieldEvaluator
        Use to perform cashed field calculations.
    freq : float
        Frequency of the transducer model.
    omega : float
        Angular frequency of the transducer model.
    k : float
        Wavenumber in air, corresponding to `freq`.
    wavelength : float
        Wavelength in air, corresponding to `freq`.
    """

    def __init__(self, transducer_positions, transducer_normals,
                 transducer_model=None, transducer_size=10e-3, transducer_kwargs=None,
                 medium=Air, **kwargs
                 ):
        self.transducer_size = transducer_size
        transducer_kwargs = transducer_kwargs or {}
        self.medium = medium
        transducer_kwargs['medium'] = self.medium

        if transducer_model is None:
            from .transducers import TransducerModel
            self.transducer_model = TransducerModel(**transducer_kwargs)
        elif type(transducer_model) is type:
            self.transducer_model = transducer_model(**transducer_kwargs)
        else:
            self.transducer_model = transducer_model

        self.calculate = self.PersistentFieldEvaluator(self)

        self.transducer_positions = transducer_positions
        self.num_transducers = self.transducer_positions.shape[1]
        if transducer_normals.ndim == 1:
            transducer_normals = np.tile(transducer_normals.reshape(3, 1), (1, self.num_transducers))
        self.transducer_normals = transducer_normals
        self.amplitudes = np.ones(self.num_transducers)
        self.phases = np.zeros(self.num_transducers)

        self.visualize = Visualizer(self)

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
        """Transducer element controls on complex form.

        The complex form of the transducer element controls is a convenience form.
        The returned value will be calculated from the normal phases and amplitudes.

        Note
        ----
            Do not try to set a single complex element as `array.complex_amplitudes[0] = 1 + 1j`.
            It will not change the underlining phases and amplitudes, only the temporary complex numpy array.
        """
        return self.amplitudes * np.exp(1j * self.phases)

    @complex_amplitudes.setter
    def complex_amplitudes(self, value):
        self.amplitudes = np.abs(value)
        self.phases = np.angle(value)

    def focus_phases(self, focus):
        """Focuses the phases to create a focus point.

        Parameters
        ----------
        focus : array_like
            Three element array with a location where to focus.

        Returns
        -------
        phases : numpy.ndarray
            Array with the phases for the transducer elements.

        """
        phase = -np.sum((self.transducer_positions - focus.reshape([3, 1]))**2, axis=0)**0.5 * self.k
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        return phase

    def signature(self, focus, phases=None):
        """Calculate the phase signature of the array.

        The signature of an array if the phase of the transducer elements
        when the phase required to focus all elements to a specific point
        has been removed.

        Parameters
        ----------
        focus : array_like
            Three element array with a location for where the signature is relative to.
        phases : numpy.ndarray, optional
            The phases of which to calculate the signature.
            Will default to the current phases in the array.

        Returns
        -------
        signature : numpy.ndarray
            The signature wrapped to the interval [-pi, pi].
        """
        if phases is None:
            phases = self.phases
        focus_phases = self.focus_phases(focus)
        return np.mod(phases - focus_phases + np.pi, 2 * np.pi) - np.pi

    def spatial_derivatives(self, positions, orders=3):
        """Calculate the spatial derivatives for all the transducers.

        Parameters
        ----------
        positions : numpy.ndarray
            The location(s) at which to evaluate the derivatives, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape (M, N, ...) where M is the number of spatial derivatives,
            and N is the number of transducers, see `num_spatial_derivatives` and `spatial_derivative_order`,
            and the remaining dimensions are the same as the `positions` input with the first dimension removed.
        """
        derivatives = np.empty((num_spatial_derivatives[orders], self.num_transducers) + positions.shape[1:], dtype=np.complex128)

        for idx in range(self.num_transducers):
            derivatives[:, idx] = self.transducer_model.spatial_derivatives(self.transducer_positions[:, idx], self.transducer_normals[:, idx], positions, orders)
        return derivatives

    class PersistentFieldEvaluator:
        """Implementation of cashed field calculations.

        Parameters
        ----------
        array : `TransducerArray`
            The array of which to calculate the fields.

        """

        from .algorithms import second_order_force as _force, second_order_stiffness as _stiffness

        def __init__(self, array):
            self.array = array
            self._last_positions = None
            self._spatial_derivatives = None
            self._existing_orders = -1

        def spatial_derivatives(self, positions, orders=3):
            """Cashed wrapper around `TransducerArray.spatial_derivatives`."""
            if (
                self._spatial_derivatives is not None and
                self._existing_orders >= orders and
                positions.shape == self._last_positions.shape and
                np.allclose(positions, self._last_positions)
            ):
                return self._spatial_derivatives

            self._spatial_derivatives = self.array.spatial_derivatives(positions, orders)
            self._existing_orders = orders
            self._last_positions = positions
            return self._spatial_derivatives

        def pressure(self, positions):
            """Calculate the pressure field.

            Parameters
            ----------
            positions : numpy.ndarray
                The location(s) at which to calculate the pressure, shape (3, ...).
                The first dimension must have length 3 and represent the coordinates of the points.

            Returns
            -------
            pressure : numpy.ndarray
                The complex pressure amplitudes, shape (...) as the positions.
            """
            return np.einsum('i..., i', self.spatial_derivatives(positions, orders=0)[0], self.array.complex_amplitudes)
            # return self._cost_functions.pressure(self.array, spatial_derivatives=self.spatial_derivatives(positions, orders=0))(self.array.phases, self.array.amplitudes)

        def velocity(self, positions):
            """Calculate the velocity field.

            Parameters
            ----------
            positions : numpy.ndarray
                The location(s) at which to calculate the velocity, shape (3, ...).
                The first dimension must have length 3 and represent the coordinates of the points.

            Returns
            -------
            velocity : numpy.ndarray
                The complex vector particle velocity, shape (3, ...) as the positions.
            """
            return np.einsum('ji..., i->j...', self.spatial_derivatives(positions, orders=1)[1:4], self.array.complex_amplitudes) / (1j * self.array.omega * self.array.medium.rho)
            # return self._cost_functions.velocity(self.array, spatial_derivatives=self.spatial_derivatives(positions, orders=1))(self.array.phases, self.array.amplitudes)

        def force(self, positions, **kwargs):
            """Calculate the force field.

            Parameters
            ----------
            positions : numpy.ndarray
                The location(s) at which to calculate the force, shape (3, ...).
                The first dimension must have length 3 and represent the coordinates of the points.

            Returns
            -------
            force : numpy.ndarray
                The vector radiation force, shape (3, ...) as the positions.
            """
            summed_derivs = np.einsum('ji..., i->j...', self.spatial_derivatives(positions, orders=2), self.array.complex_amplitudes)
            return TransducerArray.PersistentFieldEvaluator._force(self.array, **kwargs)[0](summed_derivs)

        def stiffness(self, positions, **kwargs):
            """Calculate the stiffness field.

            Parameters
            ----------
            positions : numpy.ndarray
                The location(s) at which to calculate the stiffness, shape (3, ...).
                The first dimension must have length 3 and represent the coordinates of the points.

            Returns
            -------
            force : numpy.ndarray
                The radiation stiffness, shape (...) as the positions.
            """
            summed_derivs = np.einsum('ji..., i->j...', self.spatial_derivatives(positions, orders=3), self.array.complex_amplitudes)
            return TransducerArray.PersistentFieldEvaluator._stiffness(self.array, **kwargs)[0](summed_derivs)


class RectangularArray(TransducerArray):
    """TransducerArray implementation for rectangular arrays.

    Defines the locations and normals of elements (transducers) in an array.
    For rotated arrays, the rotation is as follows:

        1) A grid of the correct layout is crated in the xy-plane
        2) The grid is rotated to the desired plane, as defined by the normal.
        3) The grid is rotated around the normal.

    The rotation to the desired plane is around the line where the desired
    plane intersects with the xy-plane.

    Parameters
    ----------
    shape : int or (int, int), default 16
        The number of transducer elements. Passing a single int will create a square array.
    spread : float, default 10e-3
        The distance between the array elements.
    offset : 3 element array_like, default (0, 0, 0)
        The location of the center of the array.
    normal : 3 element array_like, default (0, 0, 1)
        The normal of all elements in the array.
    rotation : float, default 0
        The in-plane rotation of the array around the normal.
    """

    def __init__(self, shape=16, spread=10e-3, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        positions, normals = self.grid_generator(shape=shape, spread=spread, offset=offset, normal=normal, rotation=rotation, **kwargs)
        kwargs.setdefault('transducer_size', spread)
        super().__init__(positions, normals, **kwargs)

    @classmethod
    def grid_generator(cls, shape=None, spread=None, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        """Create a grid with positions and normals.

        See `RectangularArray` for parameters and description.

        Returns
        -------
        positions : numpy.ndarray
            The positions of the array elements, shape 3xN.
        normals : numpy.ndarray
            The normals of the array elements, shape 3xN.
        """
        if not hasattr(shape, '__len__') or len(shape) == 1:
            shape = (shape, shape)
        normal = np.asarray(normal, dtype='float64')
        normal /= (normal**2).sum()**0.5
        x = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0]) * spread
        y = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1]) * spread

        X, Y, Z = np.meshgrid(x, y, 0)
        positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()))
        normals = np.tile(normal.reshape((3, 1)), (1, positions.shape[1]))

        if normal[0] != 0 or normal[1] != 0:
            # We need to rotate the grid to get the correct normal
            rotation_vector = np.cross(normal, (0, 0, 1))
            rotation_vector /= (rotation_vector**2).sum()**0.5
            cross_product_matrix = np.array([[0, rotation_vector[2], -rotation_vector[1]],
                                             [-rotation_vector[2], 0, rotation_vector[0]],
                                             [rotation_vector[1], -rotation_vector[0], 0]])
            cos = normal[2]
            sin = (1 - cos**2)**0.5
            rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(rotation_vector, rotation_vector))
        else:
            rotation_matrix = np.eye(3)
        if rotation != 0:
            cross_product_matrix = np.array([[0, normal[2], -normal[1]],
                                             [-normal[2], 0, normal[0]],
                                             [normal[1], -normal[0], 0]])
            cos = np.cos(-rotation)
            sin = np.sin(-rotation)
            rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(normal, normal)).dot(rotation_matrix)

        positions = rotation_matrix.dot(positions)
        positions += np.asarray(offset).reshape([3] + (positions.ndim - 1) * [1])
        return positions, normals

    def twin_signature(self, position=(0, 0), angle=None):
        """Get the twin trap signature.

        The twin trap signature should be added to focusing phases for a specific point
        in order to create a twin trap at that location. The twin signature shifts the
        phase of half of the elements by pi, splitting the array along a straight line.

        Parameters
        ----------
        position : array_like, default (0, 0)
            The center position for the signature, the line goes through this point.
        angle : float, optional
            The angle between the x-axis and the dividing line.
            Default is to create a line perpendicular to the line from the center of the array
            to `position`.

        Returns
        -------
        signature : numpy.ndarray
            The twin signature.

        Todo
        ----
        This is not at all working for arrays where the normal is not (0, 0, 1).
        """
        if angle is None:
            angle = np.arctan2(position[1], position[0]) + np.pi / 2
        signature = np.arctan2(self.transducer_positions[1] - position[1], self.transducer_positions[0] - position[0]) - angle
        signature = np.round(np.mod(signature / (2 * np.pi), 1))
        signature = (signature - 0.5) * np.pi
        return signature

    def vortex_signature(self, position=(0, 0), angle=0):
        """Get the vortex trap signature.

        The vortex trap signature should be added to focusing phases for a specific point
        in order to create a vortex trap at that location. The vortex signature phase shifts
        the elements in the array according to their angle in the coordinate plane.

        Parameters
        ----------
        position : array_like, default (0, 0)
            The center position for the signature.
        angle : float, default 0
            An angle which will be added to the rotation, in radians.

        Returns
        -------
        signature : numpy.ndarray
            The vortex signature.

        Todo
        ----
            This is not at all working for arrays where the normal is not (0, 0, 1).
        """
        return np.arctan2(self.transducer_positions[1] - position[1], self.transducer_positions[0] - position[0]) + angle

    def bottle_signature(self, position=(0, 0), radius=None):
        """Get the bottle trap signature.

        The bottle trap signature should be added to focusing phases for a specific point
        in order to create a bottle trap at that location. The bottle signature phase shifts
        the elements in the array according to their distance from the center, creating
        an inner zone and an outer zone of equal area with a relative shift of pi.

        Parameters
        ----------
        position : array_like, default (0, 0)
            The center position for the signature.
        radius : numeric, optional
            A custom radius to use for the division of transducers.
            The default is to use equal area partition based on the rectangular
            area occupied by each transducer. This gives the same number of transducers
            in the two groups for square arrays.

        Returns
        -------
        signature : numpy.ndarray
            The bottle signature.

        Todo
        ----
            This is not at all working for arrays where the normal is not (0, 0, 1).
        """
        position = np.asarray(position)[:2]
        if radius is None:
            A = self.num_transducers * self.transducer_size**2
            radius = (A / 2 / np.pi)**0.5
        return np.where(np.sum((self.transducer_positions[:2] - position[:, None])**2, axis=0) > radius**2, np.pi, 0)


class DoublesidedArray:
    """TransducerArray implementation for doublesided arrays.

    Creates a doublesided array based on mirroring a singlesided array.

    Parameters
    ----------
    ctype : Subclass of `TransducerArray`
        A class representing a singlesided array. Needs to implement `grid_generator`.
    separation : float
        The distance between the two halves, along the normal.
    offset : array_like, 3 elements
        The placement of the center between the two arrays.
    normal : array_like, 3 elements
        The normal of the first half.
    rotation : float, default 0
        The rotation around the normal of the first half.
    **kwargs
        Remaining arguments will be passed to the initializer for the singlesided array.
    """

    def __new__(cls, ctype, *args, **kwargs):
        """Create a new instance of the metaclass."""
        obj = ctype.__new__(ctype)
        obj.__class__ = type('Doublesided{}'.format(ctype.__name__), (DoublesidedArray, ctype), {})
        return obj

    def __init__(self, ctype, separation, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        # positions, normals = self.doublesided_generator(separation, offset=offset, normal=normal, rotation=rotation, **kwargs)
        super().__init__(separation=separation, offset=offset, normal=normal, rotation=rotation, **kwargs)
        # TransducerArray.__init__(self, positions, normals, **kwargs)

    @classmethod
    def grid_generator(cls, separation=None, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        """Create a double sided transducer grid.

        See `DoublesidedArray`.

        Returns
        -------
        positions : numpy.ndarray
            3xN array with the positions of the elements.
        normals : numpy.ndarray
            3xN array with the normals of the elements.
        """
        normal = np.asarray(normal, dtype='float64')
        normal /= (normal**2).sum()**0.5

        pos_1, norm_1 = super().grid_generator(offset=offset - 0.5 * separation * normal, normal=normal, rotation=rotation, **kwargs)
        pos_2, norm_2 = super().grid_generator(offset=offset + 0.5 * separation * normal, normal=-normal, rotation=-rotation, **kwargs)
        return np.concatenate([pos_1, pos_2], axis=1), np.concatenate([norm_1, norm_2], axis=1)

    def doublesided_signature(self):
        """Get the doublesided trap signature.

        The doublesided trap signature should be added to focusing phases for a specific point
        in order to create a trap at that location. The doublesided signature phase shifts
        the elements in one side of the array by pi.

        Returns
        -------
        signature : numpy.ndarray
            The doublesided signature.
        """
        return np.where(np.arange(self.num_transducers) < self.num_transducers // 2, 0, np.pi)


class DragonflyArray(RectangularArray):

    @classmethod
    def grid_generator(cls, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        from .hardware import dragonfly_grid
        positions, normals = dragonfly_grid

        if normal[0] != 0 or normal[1] != 0:
            # We need to rotate the grid to get the correct normal
            rotation_vector = np.cross(normal, (0, 0, 1))
            rotation_vector /= (rotation_vector**2).sum()**0.5
            cross_product_matrix = np.array([[0, rotation_vector[2], -rotation_vector[1]],
                                             [-rotation_vector[2], 0, rotation_vector[0]],
                                             [rotation_vector[1], -rotation_vector[0], 0]])
            cos = normal[2]
            sin = (1 - cos**2)**0.5
            rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(rotation_vector, rotation_vector))
        else:
            rotation_matrix = np.eye(3)
        if rotation != 0:
            cross_product_matrix = np.array([[0, normal[2], -normal[1]],
                                             [-normal[2], 0, normal[0]],
                                             [normal[1], -normal[0], 0]])
            cos = np.cos(-rotation)
            sin = np.sin(-rotation)
            rotation_matrix = (cos * np.eye(3) + sin * cross_product_matrix + (1 - cos) * np.outer(normal, normal)).dot(rotation_matrix)

        positions = rotation_matrix.dot(positions)
        positions += np.asarray(offset).reshape([3] + (positions.ndim - 1) * [1])
        return positions, normals
