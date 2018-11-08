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
