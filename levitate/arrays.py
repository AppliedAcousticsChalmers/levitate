"""Handling of transducer arrays, grouping multiple transducer elements.

The main class is the `TransducerArray` class, but other classes exist to
simplify the creation of the transducer positions for common array geometries.

.. autosummary::
    :nosignatures:

    TransducerArray
    NormalTransducerArray
    RectangularArray
    DoublesidedArray
    DragonflyArray

"""

import numpy as np
from . import utils


class TransducerArray:
    """Base class to handle transducer arrays.

    This class has no notion of the layout. If possible, try to use a more specific
    implementation instead.

    Parameters
    ----------
    positions : numpy.ndarray
        The positions of the transducer elements in the array, shape 3xN.
    normals : numpy.ndarray
        The normals of the transducer elements in the array, shape 3xN.
    transducer
        An object of `levitate.transducers.TransducerModel` or a subclass. If passed a class it will create a new instance.
    transducer_size : float
        Fallback transducer size if no transducer model object is given, or if no grid is given.
    transducer_kwargs : dict
        Extra keyword arguments used when instantiating a new transducer model.

    Attributes
    ----------
    num_transducers : int
        The number of transducers used.
    positions : numpy.ndarray
        As above.
    normals : numpy.ndarray
        As above.
    transducer : TransducerModel
        An instance of a specific transducer model implementation.
    freq : float
        Frequency of the transducer model.
    omega : float
        Angular frequency of the transducer model.
    k : float
        Wavenumber in air, corresponding to `freq`.
    wavelength : float
        Wavelength in air, corresponding to `freq`.

    """

    _repr_fmt_spec = '{:%cls(transducer=%transducer_full, transducer_size=%transducer_size,\n\tpositions=%positions,\n\tnormals=%normals)}'
    _str_fmt_spec = '{:%cls(transducer=%transducer): %num_transducers transducers}'
    from .visualizers import ArrayVisualizer, ForceDiagram

    def __init__(self, positions, normals,
                 transducer=None, transducer_size=10e-3, transducer_kwargs=None,
                 medium=None, **kwargs
                 ):
        self.transducer_size = transducer_size
        transducer_kwargs = transducer_kwargs or {}
        self._extra_print_args = {}

        if transducer is None:
            from .transducers import PointSource
            self.transducer = PointSource(**transducer_kwargs)
        elif type(transducer) is type:
            self.transducer = transducer(**transducer_kwargs)
        else:
            self.transducer = transducer
        if medium is not None:
            self.medium = medium

        self.positions = positions
        self.normals = normals

        self.visualize = type(self).ArrayVisualizer(self, 'Transducers')
        self.force_diagram = type(self).ForceDiagram(self)

    def __format__(self, fmt_spec):
        s_out = fmt_spec
        s_out = s_out.replace('%cls', self.__class__.__name__).replace('%num_transducers', str(self.num_transducers))
        s_out = s_out.replace('%transducer_size', str(self.transducer_size))
        s_out = s_out.replace('%medium_full', repr(self.medium)).replace('%medium', str(self.medium))
        s_out = s_out.replace('%transducer_full', repr(self.transducer)).replace('%transducer', str(self.transducer))
        s_out = s_out.replace('%positions', repr(self.positions)).replace('%normals', repr(self.normals))
        for key, value in self._extra_print_args.items():
            s_out = s_out.replace('%' + key, str(value))
        return s_out

    def __eq__(self, other):
        return (
            isinstance(other, TransducerArray)
            and self.num_transducers == other.num_transducers
            and np.allclose(self.positions, other.positions)
            and np.allclose(self.normals, other.normals)
            and self.transducer == other.transducer
        )

    def __repr__(self):
        return self._repr_fmt_spec.format(self)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __str__(self):
        return self._str_fmt_spec.format(self)

    @property
    def k(self):
        return self.transducer.k

    @k.setter
    def k(self, value):
        self.transducer.k = value

    @property
    def omega(self):
        return self.transducer.omega

    @omega.setter
    def omega(self, value):
        self.transducer.omega = value

    @property
    def freq(self):
        return self.transducer.freq

    @freq.setter
    def freq(self, value):
        self.transducer.freq = value

    @property
    def wavelength(self):
        return self.transducer.wavelength

    @wavelength.setter
    def wavelength(self, value):
        self.transducer.wavelength = value

    @property
    def medium(self):
        return self.transducer.medium

    @medium.setter
    def medium(self, val):
        self.transducer.medium = val

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, val):
        val = np.asarray(val)
        if not val.shape[0] == 3:
            raise ValueError('Cannot set position to these values, the first axis must have length 3 and represent the [x,y,z] coordinates!')
        self._positions = val
        self._num_transducers = val.shape[1]

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, val):
        val = np.asarray(val)
        if not val.shape[0] == 3:
            raise ValueError('Cannot set normals to these values, the first axis must have length 3 and represent the [x,y,z] components!')
        if self.num_transducers == 0:
            raise ValueError('Set the array positions before setting the normals!')
        if val.ndim == 1:
            val = np.tile(val.reshape(3, 1), (1, self.num_transducers))
        elif val.shape[1] != self.num_transducers:
            raise ValueError('The array needs to have the same number of normals as transducers!')
        self._normals = val / np.sum(val**2, axis=0)**0.5

    @property
    def num_transducers(self):
        try:
            return self._num_transducers
        except AttributeError:
            return 0

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
        focus = np.asarray(focus)
        phase = -np.sum((self.positions - focus.reshape([3, 1]))**2, axis=0)**0.5 * self.k
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-pi, pi]
        return phase

    def signature(self, position, phases, stype=None):
        """Calculate the phase signature of the array.

        The signature of an array if the phase of the transducer elements
        when the phase required to focus all elements to a specific point
        has been removed.

        Parameters
        ----------
        position : array_like
            Three element array with a position for where the signature is relative to.
        phases : numpy.ndarray
            The phases of which to calculate the signature.

        Returns
        -------
        signature : numpy.ndarray
            The signature wrapped to the interval [-pi, pi].

        """
        if stype is not None:
            raise NotImplementedError("Unknown phase signature '{}' for array of type `{}`".format(stype, self.__class__.__name__))
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
        return self.transducer.pressure_derivs(self.positions, self.normals, positions, orders)

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
        return self.transducer.spherical_harmonics(self.positions, self.normals, positions, orders)

    def request(self, requests, position):
        """Evaluate a set of requests.

        This takes a mapping (e.g. dict) of requests, and evaluates them
        at a given position. This is independent of the current transducer state.
        If a certain quantity should be calculated with regards to the current
        transducer state, use a `FieldImplementation` from the `fields` module.

        Parameters
        ----------
        position: ndarray
            The position where to calculate the requirements needed, shape (3,...).
        requests : mapping, e.g. dict
            A mapping of the desired requests. The keys in the mapping should
            start with the desired output, and the value indicates some kind of
            parameter set. Possible requests listed below:

                pressure_derivs
                    A number of spatial derivatives of the pressure. Should contain the
                    maximum order of differentiation, see `pressure_derivs`.
                spherical_harmonics
                    Spherical harmonics coefficients for an expansion of the pressure.
                    Should contain the maximum order of expansion, see `spherical_harmonics`.

        Returns
        -------
        evaluated_requests : dict
            A dictionary of the set of calculated data, according to the requests.

        """
        position = np.asarray(position)
        parsed_requests = {}
        for key, value in requests.items():
            if key.find('pressure_derivs') > -1:
                parsed_requests['pressure_derivs'] = max(value, parsed_requests.get('pressure_derivs', -1))
            elif key.find('spherical_harmonics_gradient') > -1:
                parsed_requests['spherical_harmonics'] = max(value + 1, parsed_requests.get('spherical_harmonics', -1))
                parsed_requests['spherical_harmonics_gradient'] = max(value, parsed_requests.get('spherical_harmonics_gradient', -1))
            elif key.find('spherical_harmonics') > -1:
                parsed_requests['spherical_harmonics'] = max(value, parsed_requests.get('spherical_harmonics', -1))
            elif key != 'complex_transducer_amplitudes':
                raise ValueError("Unknown request from `TransducerArray`: '{}'".format(key))

        evaluated_requests = {}
        if 'pressure_derivs' in parsed_requests:
            evaluated_requests['pressure_derivs'] = self.pressure_derivs(position, orders=parsed_requests.pop('pressure_derivs'))
        if 'spherical_harmonics' in parsed_requests:
            evaluated_requests['spherical_harmonics'] = self.spherical_harmonics(position, orders=parsed_requests.pop('spherical_harmonics'))
        if 'spherical_harmonics_gradient' in parsed_requests:
            gradient_order = parsed_requests.pop('spherical_harmonics_gradient')
            sph_idx = utils.SphericalHarmonicsIndexer(gradient_order)

            def A(n, m):
                return ((n + m + 1) * (n + m + 2) / (2 * n + 1) / (2 * n + 3)) ** 0.5

            def B(n, m):
                return -((n + m + 1) * (n - m + 1) / (2 * n + 1) / (2 * n + 3)) ** 0.5

            S = evaluated_requests['spherical_harmonics']
            dS_dxpiy = np.zeros((len(sph_idx), self.num_transducers) + position.shape[1:], dtype=complex)
            dS_dxmiy = np.zeros((len(sph_idx), self.num_transducers) + position.shape[1:], dtype=complex)
            dS_dz = np.zeros((len(sph_idx), self.num_transducers) + position.shape[1:], dtype=complex)

            for idx, (n, m) in enumerate(sph_idx):
                dS_dxpiy[idx] = A(n, -m) * S[sph_idx(n + 1, m - 1)]
                dS_dxmiy[idx] = -A(n, m) * S[sph_idx(n + 1, m + 1)]
                dS_dz[idx] = -B(n, m) * S[sph_idx(n + 1, m)]
                try:
                    dS_dxpiy[idx] += A(n - 1, m - 1) * S[sph_idx(n - 1, m - 1)]
                except ValueError:
                    pass
                try:
                    dS_dxmiy[idx] -= A(n - 1, - m - 1) * S[sph_idx(n - 1, m + 1)]
                except ValueError:
                    pass
                try:
                    dS_dz[idx] += B(n - 1, m) * S[sph_idx(n - 1, m)]
                except ValueError:
                    pass

            dS_dx = 0.5 * (dS_dxpiy + dS_dxmiy)
            dS_dy = -0.5j * (dS_dxpiy - dS_dxmiy)

            dS = np.stack([dS_dx, dS_dy, dS_dz], axis=0) * self.k
            evaluated_requests['spherical_harmonics_gradient'] = dS

        if len(parsed_requests) > 0:
            raise ValueError('Unevaluated requests: {}'.format(parsed_requests))
        return evaluated_requests


class NormalTransducerArray(TransducerArray):
    """Transducer array with a clearly defined normal.

    This is mostly intended as a base class for other implementations.
    The advantage is that a simple arrangement can be created assuming a normal
    along the z-axis, which is then rotated and moved to the desired orientation.
    The positions and normals of the transducers should be input assuming that
    the overall normal for the array is along the z-axis. The positions and normals
    will be rotated around the origin to give the desired overall normal.
    This rotation will take place along the intersection line of the plane specificed
    by the desired normal, and the xy-plane.
    If rotation is desired, the positions are further rotated using the normal
    as the rotation axis. Finally an offset is applied to the entire array.

    Parameters
    ----------
    positions : numpy.ndarray
        The positions of the transducer elements in the array, shape 3xN.
    normals : numpy.ndarray
        The normals of the transducer elements in the array, shape 3xN (or 3 elements which will broadcast).
    offset : 3 element array_like, default (0, 0, 0)
        The location of the center of the array.
    normal : 3 element array_like, default (0, 0, 1)
        The normal of the overall array.
    rotation : float, default 0
        The in-plane rotation of the array around the normal.

    """

    def __init__(self, positions, normals, offset=(0, 0, 0), normal=(0, 0, 1), rotation=0, **kwargs):
        normal = np.asarray(normal, dtype=float)
        normal /= (normal**2).sum()**0.5
        self._overall_normal = normal
        self._overall_offset = offset
        self._overall_rotation = rotation
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
        positions[0] += offset[0]
        positions[1] += offset[1]
        positions[2] += offset[2]
        normals = rotation_matrix.dot(normals)

        kwargs.setdefault('positions', positions)
        kwargs.setdefault('normals', normals)
        super().__init__(**kwargs)
        self._extra_print_args.update(offset=offset, normal=normal, rotation=rotation)

    def signature(self, position=None, *args, stype=None, **kwargs):
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
            signature = np.arctan2(self.positions[1] - position[1], self.positions[0] - position[0]) - angle
            signature = np.round(np.mod(signature / (2 * np.pi), 1))
            signature = (signature - 0.5) * np.pi
            return signature
        if stype.lower().strip() == 'vortex':
            angle = kwargs.get('angle', 0)
            return np.arctan2(self.positions[1] - position[1], self.positions[0] - position[0]) + angle
        if stype.lower().strip() == 'bottle':
            position = np.asarray(position)[:2]
            radius = kwargs.get('radius', (self.num_transducers / 2 / np.pi)**0.5 * self.transducer_size)
            return np.where(np.sum((self.positions[:2] - position[:, None])**2, axis=0) > radius**2, np.pi, 0)
        return super().signature(position, stype=stype, *args, **kwargs)


class RectangularArray(NormalTransducerArray):
    """TransducerArray implementation for rectangular arrays.

    Defines the locations and normals of elements (transducers) in an array.
    See `NormaltransducerArray` for documentation of roration and transslation options.

    Parameters
    ----------
    shape : int or (int, int), default 16
        The number of transducer elements. Passing a single int will create a square array.
    spread : float, default 10e-3
        The distance between the array elements.

    """

    _str_fmt_spec = '{:%cls(transducer=%transducer, shape=%shape, spread=%spread, offset=%offset, normal=%normal, rotation=%rotation)}'

    def __init__(self, shape=16, spread=10e-3, **kwargs):
        if not hasattr(shape, '__len__') or len(shape) == 1:
            shape = (shape, shape)
        x = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0]) * spread
        y = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1]) * spread

        X, Y, Z = np.meshgrid(x, y, 0)
        positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()))

        kwargs.setdefault('transducer_size', spread)
        kwargs.setdefault('positions', positions)
        kwargs.setdefault('normals', [0, 0, 1])
        super().__init__(**kwargs)
        self._extra_print_args.update(shape=shape, spread=spread)


class SphericalCapArray(NormalTransducerArray):
    """Transducer array implementation for spherical caps.

    The transducers will be placed on a virtual spherical surface, i.e. on the same
    distance from a given point in space. Control the overall shape of the array
    with the `radius`, `rings`, and `spead` parameters.
    See `NormalTransdcerArray` for details on the overall placement of the array,
    e.g. rotations and offsets.

    There are many ways to pack transdcuers on a spherical surface.
    The 'distance' method will place the transducers on concentric rings where
    the distance between each ring is pre-determined. Each ring will have as
    many transducers as possible for the given ring size. This will typically
    pack the transducers densely, and the outer dimentions of the array is
    quite consistent.
    The 'count' method will use a pre-determined number of transducers in each ring,
    with 6 additional transducers for each successive ring. The inter-ring distance
    will be set to fit the requested number of transducers. This method will deliver
    a pre-determined number of transducers, but will not be as dense.
    If too many rings are requested, the 'count' method will fill a half-spere with
    transducers and then stop. The 'distance' method can fill the entire sphere with
    transducers.

    Parameters
    ----------
    radius : float
        The curvature of the spherical cap, i.e. how for away the focus is.
    rings : int
        Number of consecutive rings of transducers in the array.
    packing : str, default 'distance'
        Controlls which packing method is used. One of 'distance' or 'count', see above.
    spread : float, default 10e-3
        Controls the minimum spacing between individual transducers.
    """

    _str_fmt_spec = '{:%cls(transducer=%transducer, radius=%radius, rings=%rings, packing=%packing, offset=%offset, normal=%normal, rotation=%rotation)}'

    def __init__(self, radius, rings, spread=10e-3, packing='distance', **kwargs):
        focus = np.array([0, 0, 1]) * radius
        positions = []
        normals = []
        positions.append(np.array([0., 0., 0.]))
        normals.append(np.array([0., 0., 1.]))
        if packing == 'distance':
            for ring in range(1, rings + 1):
                inclination = np.pi - ring * 2 * np.arcsin(spread / 2 / radius)
                if inclination < 0:
                    # This means that we have filled the entire sphere.
                    break
                num_trans = int(np.sin(inclination) * radius * 2 * np.pi / spread)
                azimuth = np.arange(num_trans) / num_trans * 2 * np.pi
                position = radius * np.stack([
                    np.sin(inclination) * np.cos(azimuth),
                    np.sin(inclination) * np.sin(azimuth),
                    np.cos(inclination) * np.ones(num_trans)
                ], 1) + focus
                normal = focus - position
                positions.extend(position)
                normals.extend(normal)
        elif packing == 'count':
            for ring in range(1, rings + 1):
                azimuth = np.arange(6 * ring) / (6 * ring) * np.pi * 2
                axial_radius = spread / 2 / np.sin(np.pi / 6 / ring)
                if axial_radius > radius:
                    # We have filled the half-sphere, no possibility of fitting more transducers.
                    break
                height = radius - (radius**2 - axial_radius**2)**0.5
                position = np.stack([
                    axial_radius * np.cos(azimuth),
                    axial_radius * np.sin(azimuth),
                    np.ones_like(azimuth) * height], 1)
                normal = focus - position
                positions.extend(position)
                normals.extend(normal)
        kwargs.setdefault('transducer_size', spread)
        positions = np.stack(positions, 1)
        normals = np.stack(normals, 1)
        normals /= np.sum(normals**2, axis=0)**0.5
        super().__init__(positions=positions, normals=normals, **kwargs)
        self._extra_print_args.update(radius=radius, rings=rings, packing=packing, spread=spread)


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
        normal = np.asarray(normal, dtype=float).copy()
        normal /= (normal**2).sum()**0.5
        offset = np.asarray(offset).copy()
        lower_positions = array.positions - 0.5 * separation * normal[:, None]
        upper_positions = lower_positions - 2 * np.sum(lower_positions * normal[:, None], axis=0) * normal[:, None]
        lower_normals = array.normals.copy()
        normal_proj = np.sum(lower_normals * normal[:, None], axis=0) * normal[:, None]
        upper_normals = lower_normals - 2 * normal_proj
        super().__init__(
            positions=np.concatenate([lower_positions, upper_positions], axis=1) + offset[:, None],
            normals=np.concatenate([lower_normals, upper_normals], axis=1),
            transducer=array.transducer, transducer_size=array.transducer_size,
        )
        self._extra_print_args.update(extra_print_args)

        self._array_type = type(array)

    def signature(self, position=None, *args, stype=None, **kwargs):
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


class DragonflyArray(NormalTransducerArray):
    """Rectangular array with Ultrahaptics Dragonfly U5 layout.

    This is a 16x16 element array where the order of the transducer elements
    are the same as the iteration order in the Ultrahaptics SDK. Otherwise
    behaves exactly like a `RectangularArray`.
    """

    _str_fmt_spec = '{:%cls(transducer=%transducer, offset=%offset, normal=%normal, rotation=%rotation)}'

    def __init__(self, **kwargs):
        from .hardware import dragonfly_grid
        kwargs.update(transducer_size=10e-3, positions=dragonfly_grid[0], normals=dragonfly_grid[1])
        super().__init__(**kwargs)
