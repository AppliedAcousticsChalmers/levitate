"""Handling of transducer arrays, grouping multiple transducer elements.

The main class is the `TransducerArray` class, but other classes exist to
simplify the creation of the transducer positions for common array geometries.

.. autosummary::
    :nosignatures:

    TransducerArray
    RectangularArray
    DoublesidedArray
    DragonflyArray

"""

import numpy as np
from .visualize import Visualizer


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
    transducer_model : TransducerModel
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

    _repr_fmt_spec = '{:%cls(transducer_model=%transducer_model_full, transducer_size=%transducer_size,\n\ttransducer_positions=%transducer_positions,\n\ttransducer_normals=%transducer_normals)}'
    _str_fmt_spec = '{:%cls(transducer_model=%transducer_model): %num_transducers transducers}'

    def __init__(self, transducer_positions, transducer_normals,
                 transducer_model=None, transducer_size=10e-3, transducer_kwargs=None,
                 medium=None, **kwargs
                 ):
        self.transducer_size = transducer_size
        transducer_kwargs = transducer_kwargs or {}
        self._extra_print_args = {}

        if transducer_model is None:
            from .transducers import PointSource
            self.transducer_model = PointSource(**transducer_kwargs)
        elif type(transducer_model) is type:
            self.transducer_model = transducer_model(**transducer_kwargs)
        else:
            self.transducer_model = transducer_model
        if medium is not None:
            self.medium = medium

        self.calculate = self.PersistentFieldEvaluator(self)

        self.transducer_positions = transducer_positions
        self.num_transducers = self.transducer_positions.shape[1]
        if transducer_normals.ndim == 1:
            transducer_normals = np.tile(transducer_normals.reshape(3, 1), (1, self.num_transducers))
        self.transducer_normals = transducer_normals
        self.amplitudes = np.ones(self.num_transducers)
        self.phases = np.zeros(self.num_transducers)

        self.visualize = Visualizer(self)

    def __format__(self, fmt_spec):
        s_out = fmt_spec
        s_out = s_out.replace('%cls', self.__class__.__name__).replace('%num_transducers', str(self.num_transducers))
        s_out = s_out.replace('%transducer_size', str(self.transducer_size))
        s_out = s_out.replace('%medium_full', repr(self.medium)).replace('%medium', str(self.medium))
        s_out = s_out.replace('%transducer_model_full', repr(self.transducer_model)).replace('%transducer_model', str(self.transducer_model))
        s_out = s_out.replace('%transducer_positions', repr(self.transducer_positions)).replace('%transducer_normals', repr(self.transducer_normals))
        for key, value in self._extra_print_args.items():
            s_out = s_out.replace('%' + key, str(value))
        return s_out

    def __eq__(self, other):
        return (
            isinstance(other, TransducerArray)
            and self.num_transducers == other.num_transducers
            and np.allclose(self.transducer_positions, other.transducer_positions)
            and np.allclose(self.transducer_normals, other.transducer_normals)
            and self.transducer_model == other.transducer_model
        )

    def __repr__(self):
        return self._repr_fmt_spec.format(self)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __str__(self):
        return self._str_fmt_spec.format(self)

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
    def medium(self):
        return self.transducer_model.medium

    @medium.setter
    def medium(self, val):
        self.transducer_model.medium = val

    @property
    def complex_amplitudes(self):
        """Transducer element controls on complex form.

        The complex form of the transducer element controls is a convenience form.
        The returned value will be calculated from the normal phases and amplitudes.

        Warning
        -------
        Do not try to set a single complex element as `array.complex_amplitudes[0] = 1 + 1j`.
        It will not change the underlying phases and amplitudes, only the temporary complex numpy array.

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

    def signature(self, position, phases=None, stype=None):
        """Calculate the phase signature of the array.

        The signature of an array if the phase of the transducer elements
        when the phase required to focus all elements to a specific point
        has been removed.

        Parameters
        ----------
        position : array_like
            Three element array with a position for where the signature is relative to.
        phases : numpy.ndarray, optional
            The phases of which to calculate the signature.
            Will default to the current phases in the array.

        Returns
        -------
        signature : numpy.ndarray
            The signature wrapped to the interval [-pi, pi].

        """
        if stype is not None:
            raise NotImplementedError("Unknown phase signature '{}' for array of type `{}`".format(stype, self.__class__.__name__))
        if phases is None:
            phases = self.phases
        focus_phases = self.focus_phases(position)
        return np.mod(phases - focus_phases + np.pi, 2 * np.pi) - np.pi

    def pressure_derivs(self, positions, orders=3):
        """Calculate derivatives of the pressure.

        Calculates the spatial derivatives of the pressure from all individual
        transducers in a Cartesian coordinate system.

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
        return self.transducer_model.pressure_derivs(self.transducer_positions, self.transducer_normals, positions, orders)

    def spherical_harmonics(self, positions, orders=0):
        """Spherical harmonics expansion of transducer sound fields.

        The sound fields generated by the individual transducers in the array are expanded
        in spherical harmonics around the positions specified. The coefficients are calculated
        using analytical translation of the transducer radiation patterns. This is a simplified
        calculation which will not account for the local directivity curve, only an overall
        scaling for each transducer-position combination.

        Parameters
        ----------
        positions : numpy.ndarray
            The location(s) at which to evaluate the derivatives, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int, default 0
            The maximum order to expand to.

        Return
        ------
        spherical_harmonics_coefficients : numpy.ndarray
            Array with the calculated expansion coefficients. The order of the coefficients
            are described in `~levitate.utils.SphericalHarmonicsIndexer`.
            Has shape (M, N, ...) where `M=len(SphericalHarmonicsIndexer(orders))`,
            `N` is the number of transducers in the array, and the remaining dimensions are
            the same as the `positions` input with the first dimension removed.

        """
        return self.transducer_model.spherical_harmonics(self.transducer_positions, self.transducer_normals, positions, orders)

    class PersistentFieldEvaluator:
        """Implementation of cashed field calculations.

        Parameters
        ----------
        array : `TransducerArray`
            The array of which to calculate the fields.

        """

        from .algorithms import RadiationForce as _force, RadiationForceStiffness as _stiffness

        def __init__(self, array):
            self.array = array
            self._last_positions = None
            self._pressure_derivs = None
            self._existing_orders = -1

        def pressure_derivs(self, positions, orders=3):
            """Cashed wrapper around `TransducerArray.pressure_derivs`."""
            if (
                self._pressure_derivs is not None
                and self._existing_orders >= orders
                and positions.shape == self._last_positions.shape
                and np.allclose(positions, self._last_positions)
            ):
                return self._pressure_derivs

            self._pressure_derivs = self.array.pressure_derivs(positions, orders)
            self._existing_orders = orders
            self._last_positions = positions.copy()  # In case the position is modified externally we need to keep a separate reference
            return self._pressure_derivs

        def pressure(self, positions, complex_amplitudes=None):
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
            complex_amplitudes = complex_amplitudes if complex_amplitudes is not None else self.array.complex_amplitudes
            return np.einsum('i..., i', self.pressure_derivs(positions, orders=0)[0], complex_amplitudes)
            # return self._cost_functions.pressure(self.array, pressure_derivs=self.pressure_derivs(positions, orders=0))(self.array.phases, self.array.amplitudes)

        def velocity(self, positions, complex_amplitudes=None):
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
            complex_amplitudes = complex_amplitudes if complex_amplitudes is not None else self.array.complex_amplitudes
            return np.einsum('ji..., i->j...', self.pressure_derivs(positions, orders=1)[1:4], complex_amplitudes) / (1j * self.array.omega * self.array.medium.rho)
            # return self._cost_functions.velocity(self.array, pressure_derivs=self.pressure_derivs(positions, orders=1))(self.array.phases, self.array.amplitudes)

        def force(self, positions, complex_amplitudes=None, **kwargs):
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
            complex_amplitudes = complex_amplitudes if complex_amplitudes is not None else self.array.complex_amplitudes
            summed_derivs = np.einsum('ji..., i->j...', self.pressure_derivs(positions, orders=2), complex_amplitudes)
            return TransducerArray.PersistentFieldEvaluator._force(self.array, **kwargs).values(summed_derivs)

        def stiffness(self, positions, complex_amplitudes=None, **kwargs):
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
            complex_amplitudes = complex_amplitudes if complex_amplitudes is not None else self.array.complex_amplitudes
            summed_derivs = np.einsum('ji..., i->j...', self.pressure_derivs(positions, orders=3), complex_amplitudes)
            return TransducerArray.PersistentFieldEvaluator._stiffness(self.array, **kwargs).values(summed_derivs)


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

    _str_fmt_spec = '{:%cls(transducer_model=%transducer_model, shape=%shape, spread=%spread, offset=%offset, normal=%normal, rotation=%rotation)}'

    def __init__(self, shape=16, spread=10e-3, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        extra_print_args = {'shape': shape, 'spread': spread, 'offset': offset, 'normal': normal, 'rotation': rotation}
        normal = np.asarray(normal, dtype='float64')
        normal /= (normal**2).sum()**0.5
        positions, normals = self._grid_generator(shape=shape, spread=spread, normal=normal, **kwargs)

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

        kwargs.setdefault('transducer_size', spread)
        kwargs.setdefault('transducer_positions', positions)
        kwargs.setdefault('transducer_normals', normals)
        super().__init__(**kwargs)
        self._extra_print_args.update(extra_print_args)

    @classmethod
    def _grid_generator(cls, shape=None, spread=None, normal=(0, 0, 1), **kwargs):
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
        x = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0]) * spread
        y = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1]) * spread

        X, Y, Z = np.meshgrid(x, y, 0)
        positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()))
        normals = np.tile(normal.reshape((3, 1)), (1, positions.shape[1]))
        return positions, normals

    def signature(self, position=None, stype=None, *args, **kwargs):
        """Calculate phase signatures of the array.

        The signature of an array if the phase of the transducer elements
        when the phase required to focus all elements to a specific point
        has been removed. If `stype` if set to one of the available
        signatures: 'twin', 'vortex', or 'bottle', the corresponding
        signature is returned.

        The signatures and the additional keyword parameters for them are:

        Current signature (`stype=None`)
            Calculates the current phase signature. See `TransducerArray.signature`

            phases (`numpy.ndarray`, optional)
                The phases of which to calculate the signature.
                Will default to the current phases in the array.

        Twin signature (`stype='twin'`)
            Calculates the twin trap signature which shifts the phase of half
            of the elements by pi, splitting the array along a straight line.

            angle (`float`, optional)
                The angle between the x-axis and the dividing line.
                Default is to create a line perpendicular to the line from the
                center of the array to `position`.

        Vortex signature (`stype='vortex'`)
            Calculates the vortex trap signature which phase shifts the
            elements in the array according to their angle in the coordinate
            plane.

            angle (`float`, optional)
                Additional angle to rotate the phase signature with.

        Bottle signature (`stype='bottle'`)
            Calculates the bottle trap signature which phase shifts the
            elements in the array according to their distance from the center,
            creating an inner zone and an outer zone of equal area with a
            relative shift of pi.

            radius (`float`, optional)
                A custom radius to use for the division of transducers.
                The default is to use equal area partition based on the
                rectangular area occupied by each transducer. This gives the
                same number of transducers in the two groups for square arrays.

        Parameters
        ----------
        position : array_like
            Three element array with a location for where the signature is relative to.
        stype : None, 'twin', 'bottle', 'vortex'. Default None
            Chooses which type of signature to calculate.

        Returns
        -------
        signature : numpy.ndarray
            The signature wrapped to the interval [-pi, pi].

        """
        if stype is None:
            return TransducerArray.signature(self, position, stype=stype, *args, **kwargs)
        position = position if position is not None else (0, 0, 0)
        if stype.lower().strip() == 'twin':
            angle = kwargs.get('angle', None)
            if angle is None:
                angle = np.arctan2(position[1], position[0]) + np.pi / 2
            signature = np.arctan2(self.transducer_positions[1] - position[1], self.transducer_positions[0] - position[0]) - angle
            signature = np.round(np.mod(signature / (2 * np.pi), 1))
            signature = (signature - 0.5) * np.pi
            return signature
        if stype.lower().strip() == 'vortex':
            angle = kwargs.get('angle', 0)
            return np.arctan2(self.transducer_positions[1] - position[1], self.transducer_positions[0] - position[0]) + angle
        if stype.lower().strip() == 'bottle':
            position = np.asarray(position)[:2]
            radius = kwargs.get('radius', (self.num_transducers / 2 / np.pi)**0.5 * self.transducer_size)
            return np.where(np.sum((self.transducer_positions[:2] - position[:, None])**2, axis=0) > radius**2, np.pi, 0)
        return super().signature(position, stype=stype, *args, **kwargs)


class DoublesidedArray(TransducerArray):
    """TransducerArray implementation for doublesided arrays.

    Creates a doublesided array based on mirroring a singlesided array.
    This can easily be used to create standard doublesided arrays by using
    the same normal for the mirroring as for the original array. If a different
    normal is used it is possible to create e.g. v-shaped arrays.

        1) The singlesided array is "centered" at the origin, where "center" is
           defined as the mean coordinate of the elements.
        2) The singlesided array is shifted with half of the separation in the
           opposite direction of the normal to create the "lower" half.
        3) The "upper" half is created by mirroring the "lower" half in the plane
           described by the normal.
        4) Both halves are offset with a specified vector.

    Note that only the orientation of the initial array matters, not the
    overall position.

    Parameters
    ----------
    array : Instance or (sub)class of `TransducerArray`.
        The singlesided object used to the creation of the doublesided array.
        Classes will be instantiated to generate the array, using all input
        arguments except `array`, `separation`, and `offset`.
    separation : float
        The distance between the two halves, along the normal.
    offset : array_like, 3 elements
        The placement of the center between the two arrays.
    normal : array_like, 3 elements
        The normal of the reflection plane.

    """

    _str_fmt_spec = '{:%cls(%array, separation=%separation, normal=%normal, offset=%offset)}'

    def __init__(self, array, separation, normal=(0, 0, 1), offset=(0, 0, 0), **kwargs):
        if type(array) is type:
            array = array(normal=normal, **kwargs)
        extra_print_args = {'separation': separation, 'normal': normal, 'offset': offset, 'array': str(array)}
        normal = np.asarray(normal, dtype='float64').copy()
        normal /= (normal**2).sum()**0.5
        offset = np.asarray(offset).copy()
        lower_positions = array.transducer_positions - 0.5 * separation * normal[:, None]
        lower_positions -= np.mean(array.transducer_positions, axis=1)[:, None]
        upper_positions = lower_positions - 2 * np.sum(lower_positions * normal[:, None], axis=0) * normal[:, None]
        lower_normals = array.transducer_normals.copy()
        normal_proj = np.sum(lower_normals * normal[:, None], axis=0) * normal[:, None]
        upper_normals = lower_normals - 2 * normal_proj
        super().__init__(
            transducer_positions=np.concatenate([lower_positions, upper_positions], axis=1) + offset[:, None],
            transducer_normals=np.concatenate([lower_normals, upper_normals], axis=1),
            transducer_model=array.transducer_model, transducer_size=array.transducer_size,
        )
        self._extra_print_args.update(extra_print_args)

        self._array_type = type(array)

    def signature(self, position=None, stype=None, *args, **kwargs):
        """Calculate phase signatures of the array.

        The signature of an array if the phase of the transducer elements
        when the phase required to focus all elements to a specific point
        has been removed. If `stype` if set to one of the available
        signatures the corresponding signature is returned. The signatures
        of the array used when creating the doublesided array are also available.

        The signatures and the additional keyword parameters for them are:

        Current signature (`stype=None`)
            Calculates the current phase signature. See `TransducerArray.signature`

            phases (`numpy.ndarray`, optional)
                The phases of which to calculate the signature.
                Will default to the current phases in the array.

        Doublesided signature (`stype='doublesided'`)
            Calculates the doublesided trap signature which shifts the phase
            of one side of the array half of the elements by pi.

        Parameters
        ----------
        position : array_like
            Three element array with a location for where the signature is relative to.
        stype : None, 'doublesided', etc. Default None
            Chooses which type of signature to calculate.

        Returns
        -------
        signature : numpy.ndarray
            The signature wrapped to the interval [-pi, pi].

        """
        if stype is None:
            return TransducerArray.signature(self, position, stype=stype, *args, **kwargs)
        if stype.lower().strip() == 'doublesided':
            return np.where(np.arange(self.num_transducers) < self.num_transducers // 2, 0, np.pi)
        try:
            return self._array_type.signature(self, position, stype=stype, *args, **kwargs)
        except TypeError as e:
            if str(e) != 'super(type, obj): obj must be an instance or subtype of type':
                raise
        return super().signature(self, position, stype=stype, *args, **kwargs)


class DragonflyArray(RectangularArray):
    """Rectangular array with Ultrahaptics Dragonfly U5 layout.

    This is a 16x16 element array where the order of the transducer elements
    are the same as the iteration order in the Ultrahaptics SDK. Otherwise
    behaves exactly like a `RectangularArray`.
    """

    _str_fmt_spec = '{:%cls(transducer_model=%transducer_model, offset=%offset, normal=%normal, rotation=%rotation)}'

    @classmethod
    def _grid_generator(cls, **kwargs):
        from .hardware import dragonfly_grid
        return dragonfly_grid
