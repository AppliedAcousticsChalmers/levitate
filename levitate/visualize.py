"""Visualization methods based on the plotly graphing library, and some connivance functions."""
import numpy as np
from .utils import SPL, SVL


class Visualizer:
    """Handle array visualizations.

    Two different style of plots are currently implemented: scalar field
    visualizations and transducer visualizations. The methods return
    plotly compatible dictionaries corresponding to a single trace.
    The calculations are buffered, so the first plots are always slower
    than successive plots. This also means that the memory usage is very large
    for high-resolution plots.

    Parameters
    ----------
    array : `TransducerArray`
        The transducer array to visualize.
    xlimits : 2 element array_like
        Override for the default x limits
    ylimits : 2 element array_like
        Override for the default y limits
    zlimits : 2 element array_like
        Override for the default z limits
    resolution : numeric, default 10
        Resolution of the plots, in elements per wavelength.
    constant_axis : tuple or str
        `(axis, value)` or `axis` where `axis` is in `['x', 'y', 'z']` and indicates
        which axis to keep constant in the plots. The value indicates at which value
        the slice is taken. Default to `('y', 0)`.

    """

    def __init__(self, array, xlimits=None, ylimits=None, zlimits=None, resolution=10, constant_axis=('y', 0)):
        self.array = array
        xlimits = xlimits or (np.min(array.transducer_positions[0]), np.max(array.transducer_positions[0]))
        ylimits = ylimits or (np.min(array.transducer_positions[1]), np.max(array.transducer_positions[1]))
        if 'Doublesided' not in type(array).__name__:
            # Singlesided array, one of the limits will give a zero range.
            # Assuming that the array is in the xy-plane, pointing up
            zlimits = zlimits or (np.min(array.transducer_positions[2]) + 1e-3, np.min(array.transducer_positions[2]) + 20 * array.wavelength)
        else:
            zlimits = zlimits or (np.min(array.transducer_positions[2]), np.max(array.transducer_positions[2]))

        self._xlimits = xlimits
        self._ylimits = ylimits
        self._zlimits = zlimits
        self._constant_axis = constant_axis
        self.resolution = resolution
        self.calculate = array.PersistentFieldEvaluator(array)

    @property
    def xlimits(self):
        return self._xlimits

    @xlimits.setter
    def xlimits(self, val):
        self._xlimits = val
        self._update_mesh()

    @property
    def ylimits(self):
        return self._ylimits

    @ylimits.setter
    def ylimits(self, val):
        self._ylimits = val
        self._update_mesh()

    @property
    def zlimits(self):
        return self._zlimits

    @zlimits.setter
    def zlimits(self, val):
        self._zlimits = val
        self._update_mesh()

    @property
    def constant_axis(self):
        try:
            axis, value = self._constant_axis
        except ValueError:
            axis, value = self._constant_axis, 0
        return axis, value

    @constant_axis.setter
    def constant_axis(self, val):
        try:
            axis = val[0]
            value = val[1]
        except TypeError:
            axis = val
            value = 0
        self._constant_axis = (axis, value)
        self._update_mesh()

    @property
    def resolution(self):
        return self.array.wavelength / self._resolution

    @resolution.setter
    def resolution(self, val):
        self._resolution = self.array.wavelength / val
        self._update_mesh()

    def _update_mesh(self):
        axis, value = self.constant_axis
        if axis is 'x':
            self._x, self._y, self._z = np.mgrid[value:value:1j, self.ylimits[0]:self.ylimits[1]:self._resolution, self.zlimits[0]:self.zlimits[1]:self._resolution]
        if axis is 'y':
            self._x, self._y, self._z = np.mgrid[self.xlimits[0]:self.xlimits[1]:self._resolution, value:value:1j, self.zlimits[0]:self.zlimits[1]:self._resolution]
        if axis is 'z':
            self._x, self._y, self._z = np.mgrid[self.xlimits[0]:self.xlimits[1]:self._resolution, self.ylimits[0]:self.ylimits[1]:self._resolution, value:value:1j]

    @property
    def _mesh(self):
        return np.stack([self._x, self._y, self._z])

    def scalar_field(self, calculator, **kwargs):
        """Evaluate and prepare a scalar field visualization.

        Parameters
        ----------
        calculator : callable or iterable of callables
            Callable which takes a mesh as input and returns data values for every point.
            If an iterable is passed, the callables will be called with the output from the
            previous callable, the first one called with the mesh.
        **kwargs
            Remaining keyword arguments will be added to the trace dictionary, replacing defaults.

        Returns
        -------
        trace : dict
            A plotly style dictionary with the trace for the field.

        """
        data = self._mesh
        try:
            for f in calculator:
                data = f(data)
        except TypeError:
            data = calculator(data)
        trace = dict(
            type='surface', surfacecolor=np.squeeze(data),
            x=np.squeeze(self._x),
            y=np.squeeze(self._y),
            z=np.squeeze(self._z),
        )
        trace.update(kwargs)
        return trace

    def pressure(self, min=130, max=170):
        """Visualize pressure field."""
        return self.scalar_field(calculator=[self.calculate.pressure, SPL],
                                 cmin=min, cmax=max, colorscale='Viridis', colorbar={'title': 'Sound pressure in dB re. 20 µPa'})

    def velocity(self, min=130, max=170):
        """Visualize velocity field."""
        return self.scalar_field(calculator=[self.calculate.velocity, SVL],
                                 cmin=min, cmax=max, colorscale='Viridis', colorbar={'title': 'Particle velocity in dB re. 50 nm/s'})

    def transducers(self, data=None, phases=None, amplitudes=None, signature_pos=None, trace_type='scatter3d'):
        """Create transducer visualization.

        A 3d scatter trace of the transducer elements in an array.
        Uses the color of the elements to show data.

        Parameters
        ----------
        array : `TransducerArray`
            The array to visualize.
        data : 'phase', 'amplitude', or numeric
            Which data to color the transducers with.

        Returns
        -------
        trace : dict
            A plotly style dictionary with the trace for the transducers.

        """
        if phases is not None or (type(data) is str and 'phase' in data):
            if phases is None:
                data = self.array.phases / np.pi
            else:
                data = phases / np.pi
            title = 'Transducer phase in rad/π'
            cmin = -1
            cmax = 1
            colorscale = [[0.0, 'hsv(0,255,255)'], [0.25, 'hsv(90,255,255)'], [0.5, 'hsv(180,255,255)'], [0.75, 'hsv(270,255,255)'], [1.0, 'hsv(360,255,255)']]

        elif amplitudes is not None or (type(data) is str and 'amp' in data):
            if amplitudes is None:
                data = self.array.amplitudes
            else:
                data = amplitudes
            title = 'Transducer amplitude'
            cmin = 0
            cmax = 1
            colorscale = 'Viridis'
        elif signature_pos is not None:
            title = 'Transducer phase signature in rad/π'
            if phases is None and data is not None:
                phases = data
            data = self.array.signature(signature_pos, phases) / np.pi
            cmin = -1
            cmax = 1
            colorscale = [[0.0, 'hsv(0,255,255)'], [0.25, 'hsv(90,255,255)'], [0.5, 'hsv(180,255,255)'], [0.75, 'hsv(270,255,255)'], [1.0, 'hsv(360,255,255)']]
        elif data is None:
            return self.transducers(data='phases', trace_type=trace_type)
        else:
            title = 'Transducer data'
            cmin = np.min(data)
            cmax = np.max(data)
            colorscale = 'Viridis'

        if trace_type == 'scatter3d':
            marker = dict(color=data, colorscale=colorscale, size=16, colorbar={'title': title, 'x': -0.02}, cmin=cmin, cmax=cmax)
            trace = dict(
                type='scatter3d', mode='markers',
                x=self.array.transducer_positions[0],
                y=self.array.transducer_positions[1],
                z=self.array.transducer_positions[2],
                marker=marker
            )

        if trace_type == 'mesh3d':
            # Get parameters for the shape
            upper_radius = self.array.transducer_size / 2
            lower_radius = upper_radius * 2 / 3
            height = upper_radius
            num_points = 50  # Points in each circle
            num_vertices = 2 * num_points + 2  # Vertices per transducer
            theta = np.arange(num_points) / num_points * 2 * np.pi
            cos = np.cos(theta)
            sin = np.sin(theta)

            # Create base index arrays
            up_center = [0] * num_points
            up_first = list(range(1, num_points + 1))
            up_second = list(range(2, num_points + 1)) + [1]
            down_center = [num_points + 1] * num_points
            down_first = list(range(num_points + 2, 2 * num_points + 2))
            down_second = list(range(num_points + 3, 2 * num_points + 2)) + [num_points + 2]
            # Lists for base index arrays
            i = []
            j = []
            k = []
            # Upper disk base indices
            i += up_center
            j += up_first
            k += up_second
            # Lower disk base indices
            i += down_center
            j += down_second
            k += down_first
            # Half side base indices
            i += up_first
            j += down_first
            k += up_second
            # Other half side indices
            i += up_second
            j += down_first
            k += down_second
            # Base indices as array
            base_indices = np.stack([i, j, k], axis=0)

            # Lists for storing the transducer meshes
            points = []
            indices = []
            vertex_color = []

            for t_idx in range(self.array.num_transducers):
                position = self.array.transducer_positions[:, t_idx]
                normal = self.array.transducer_normals[:, t_idx]

                # Find two vectors that sweep the circle
                if normal[2] != 0:
                    v1 = np.array([1., 1., 1.])
                    v1[2] = -(v1[0] * normal[0] + v1[1] * normal[1]) / normal[2]
                else:
                    v1 = np.array([0., 0., 1.])

                v1 /= np.sum(v1**2)**0.5
                v2 = np.cross(v1, normal)
                circle = cos * v1[:, None] + sin * v2[:, None]

                upper_circle = circle * upper_radius + position[:, None]
                lower_circle = circle * lower_radius + position[:, None] - height * normal[:, None]
                points.append(np.concatenate([position[:, None], upper_circle, position[:, None] - height * normal[:, None], lower_circle], axis=1))
                indices.append(base_indices + t_idx * num_vertices)
                vertex_color.append([data[t_idx]] * num_vertices)

            points = np.concatenate(points, axis=1)
            indices = np.concatenate(indices, axis=1)
            vertex_color = np.concatenate(vertex_color)
            trace = dict(
                type='mesh3d',
                x=points[0], y=points[1], z=points[2],
                i=indices[0], j=indices[1], k=indices[2],
                intensity=vertex_color, colorscale=colorscale,
                colorbar={'title': title, 'x': -0.02}, cmin=cmin, cmax=cmax,
            )
        return trace

    def find_trap(self, start_pos, tolerance=10e-6, time_interval=50, return_path=False, rho=25, radius=1e-3):
        r"""Find the approximate location of a levitation trap.

        Find an approximate position of a acoustic levitation trap close to a starting point.
        This is done by following the radiation force in the sound field using an differential
        equation solver. The differential equation is the unphysical equation
        :math:`d\vec x/dt  = \vec F(x,t)`, i.e. interpreting the force field as a velocity field.
        This works for finding the location of a trap and the field line from the starting position
        to the trap position, but it can not be seen as a proper kinematic simulation of the system.

        The solving of the above equation takes place until the whole time interval is covered,
        or the tolerance is met. The tolerance is evaluated using the assumption that the force
        is zero at the trap, evaluating the distance from the zero-force position using the force
        gradient.

        Parameters
        ----------
        start_pos : array_like, 3 elements
            The starting point for the solving.
        tolerance : numeric, default 10e-6
            The approximate tolerance of the solution, i.e. how close should
            the found position be to the true position, in meters.
        time_interval : numeric, default 10
            The unphysical time of the solution range in the differential equation above.
        return_path : bool or int, default False
            Controls if the path from the starting point to the found trap is returned.
            Set to an int to specify the number of points in the path.
        rho : numeric, default 25
            The density of the spherical bead.
        radius : numeric, default 1e-3
            The radius of the spherical bead.

        Returns
        -------
        trap_pos : numpy.ndarray
            The found trap position, or the path from the starting position to the trap position, see `return_path`.

        """
        from scipy.integrate import solve_ivp
        mg = rho * 4 * np.pi / 3 * radius**3 * 9.82
        evaluator = self.array.PersistentFieldEvaluator(self.array)

        def f(t, x):
            F = evaluator.force(x)
            F[2] -= mg
            return F

        def bead_close(t, x):
            dF = evaluator.stiffness(x)
            F = evaluator.force(x)
            F[2] -= mg
            distance = np.sum((F / dF)**2, axis=0)**0.5
            return np.clip(distance - tolerance, 0, None)
        bead_close.terminal = True
        outs = solve_ivp(f, (0, time_interval), np.asarray(start_pos), events=bead_close, vectorized=True, dense_output=return_path)
        if outs.message != 'A termination event occurred.':
            print('End criterion not met. Final path position might not be close to trap location.')
        if return_path:
            if return_path is True:
                return_path = 200
            return outs.sol(np.linspace(0, outs.sol.t_max, return_path))
        else:
            return outs.y[:, -1]


def selection_figure(*traces, additional_traces=None):
    num_varying_traces = len(traces)
    try:
        num_static_traces = len(additional_traces)
    except TypeError:
        num_static_traces = 0
    buttons = []
    data = []
    for idx, (trace, name) in enumerate(traces):
        buttons.append(dict(label=name, method='update', args=[{'visible': idx * [False] + [True] + (num_varying_traces - idx - 1) * [False] + num_static_traces * [True]}]))
        if idx > 0:
            trace['visible'] = False
        data.append(trace)

    try:
        data.extend(additional_traces)
    except TypeError:
        pass

    layout = dict(updatemenus=[dict(active=0, buttons=buttons)])
    fig = dict(data=data, layout=layout)
    return fig
