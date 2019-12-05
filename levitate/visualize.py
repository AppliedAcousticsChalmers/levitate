"""Visualization methods based on the plotly graphing library, and some convenience functions."""
import numpy as np
from .utils import SPL, SVL
try:
    import plotly
except ImportError:
    pass


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

    def __init__(self, array, xlimits=None, ylimits=None, zlimits=None, resolution=10, display_scale='mm'):
        self.array = array
        xlimits = xlimits or (np.min(array.transducer_positions[0]), np.max(array.transducer_positions[0]))
        ylimits = ylimits or (np.min(array.transducer_positions[1]), np.max(array.transducer_positions[1]))
        if 'Doublesided' not in type(array).__name__:
            # Singlesided array, one of the limits will give a zero range.
            # Assuming that the array is in the xy-plane, pointing up
            zlimits = zlimits or (np.min(array.transducer_positions[2]) + 1e-3, np.min(array.transducer_positions[2]) + 20 * array.wavelength)
        else:
            zlimits = zlimits or (np.min(array.transducer_positions[2]), np.max(array.transducer_positions[2]))

        if max(xlimits) > min(xlimits):
            ylimits = (max(ylimits) + min(ylimits)) / 2
        else:
            xlimits = (max(xlimits) + min(xlimits)) / 2

        self._xlimits = xlimits
        self._ylimits = ylimits
        self._zlimits = zlimits
        self.resolution = resolution
        self.calculate = array.PersistentFieldEvaluator(array)

        self.display_scale = display_scale

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
    def display_scale(self):
        if self._display_scale == 1e3:
            return 'km'
        if self._display_scale == 1:
            return 'm'
        if self._display_scale == 1e-1:
            return 'dm'
        if self._display_scale == 1e-2:
            return 'cm'
        if self._display_scale == 1e-3:
            return 'mm'
        if self._display_scale == 1e-6:
            return 'µm'
        if self._display_scale == 1e-9:
            return 'nm'
        if self._display_scale == self.array.wavelength:
            return 'λ'
        return '{:.2e} m'.format(self._display_scale)

    @display_scale.setter
    def display_scale(self, val):
        if val == 'km':
            self._display_scale = 1e3
        elif val == 'm':
            self._display_scale = 1
        elif val == 'dm':
            self._display_scale = 1e-1
        elif val == 'cm':
            self._display_scale = 1e-2
        elif val == 'mm':
            self._display_scale = 1e-3
        elif val == 'µm':
            self._display_scale = 1e-6
        elif val == 'nm':
            self._display_scale = 1e-9
        elif val == 'wavelengths' or val == 'λ':
            self._display_scale = self.array.wavelength
        else:
            self._display_scale = val

    @property
    def resolution(self):
        return self.array.wavelength / self._resolution

    @resolution.setter
    def resolution(self, val):
        self._resolution = self.array.wavelength / val
        self._update_mesh()

    def _update_mesh(self):
        xmin, xmax = np.min(self._xlimits), np.max(self._xlimits)
        ymin, ymax = np.min(self._ylimits), np.max(self._ylimits)
        zmin, zmax = np.min(self._zlimits), np.max(self._zlimits)
        nx = int((xmax - xmin) / self._resolution) + 1
        ny = int((ymax - ymin) / self._resolution) + 1
        nz = int((zmax - zmin) / self._resolution) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)

        self.mesh = np.stack(np.meshgrid(x, y, z, indexing='ij'))

    def field_slice(self, calculator, min=None, max=None, trace_type='surface', **kwargs):
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
        data = self.mesh
        try:
            for f in calculator:
                data = f(data)
        except TypeError:
            data = calculator(data)

        if trace_type == 'surface':
            trace = dict(
                type='surface', surfacecolor=np.squeeze(data),
                cmin=min, cmax=max,
                x=np.squeeze(self.mesh[0]) / self._display_scale,
                y=np.squeeze(self.mesh[1]) / self._display_scale,
                z=np.squeeze(self.mesh[2]) / self._display_scale,
            )
        if trace_type == 'heatmap':
            # We need to figure out which axis is constant.
            ax = [0, 1, 2]
            try:
                ax.remove(self.mesh.shape.index(1) - 1)
            except ValueError:
                raise ValueError('Cannot generate heatmap from 3d data')
            trace = dict(
                type='heatmap', z=np.squeeze(data),
                zmin=min, zmax=max, transpose=True,
                x=np.squeeze(self.mesh[ax[0]])[:, 0] / self._display_scale,
                y=np.squeeze(self.mesh[ax[1]])[0, :] / self._display_scale,
            )
        trace.update(kwargs)
        return trace

    def pressure(self, min=130, max=170, **kwargs):
        """Visualize pressure field."""
        return self.field_slice(
            calculator=[self.calculate.pressure, SPL],
            min=min, max=max, colorscale='Viridis',
            colorbar={'title': 'Sound pressure in dB re. 20 µPa'},
            **kwargs
        )

    def velocity(self, min=130, max=170, **kwargs):
        """Visualize velocity field."""
        return self.field_slice(
            calculator=[self.calculate.velocity, SVL],
            min=min, max=max, colorscale='Viridis',
            colorbar={'title': 'Particle velocity in dB re. 50 nm/s'},
            **kwargs
        )

    def transducers(self, data=None, phases=None, amplitudes=None, signature_pos=None, trace_type='mesh3d'):
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
                x=self.array.transducer_positions[0] / self._display_scale,
                y=self.array.transducer_positions[1] / self._display_scale,
                z=self.array.transducer_positions[2] / self._display_scale,
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

            points = np.concatenate(points, axis=1) / self._display_scale
            indices = np.concatenate(indices, axis=1)
            vertex_color = np.concatenate(vertex_color)
            trace = dict(
                type='mesh3d',
                x=points[0], y=points[1], z=points[2],
                i=indices[0], j=indices[1], k=indices[2],
                intensity=vertex_color, colorscale=colorscale,
                colorbar={'title': title, 'x': -0.02}, cmin=cmin, cmax=cmax,
            )
        try:
            return trace
        except UnboundLocalError:
            raise ValueError('Unknown trace type `{}` for transducer visualization'.format(trace_type))

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

    def force_diagram_traces(self, position, range=None, label=None, color=None,
                             radius_sphere=1e-3, force_calculator=None, scale_to_gravity=True, sphere_material=None,
                             _trace_id=[0], **kwargs):
        r"""Show the force spatial dependency.

        Calculates the radiation force on a bead along the three Cartesian axes centered at a position.
        This is visualized as a 3x3 matrix style plot, with the columns corresponding to movement along
        one of the Cartesian axes, and each row corresponds to a component of the force vector.

        This function only creates the definition of the traces, without any layout.
        Use the `force_diagram_layout` to generate a suitable layout. If needed, it is possible to add
        multiple traces at different points or different array settings etc., by calling this function
        multiple times and concatenating the lists.

        Parameters
        ----------
        position : array_like, 3 elements
            The center position around which to evaluate.
        range : numeric, optional
            Specifies how far away the force should be calculated along each axis.
            Defaults to one wavelength.
        label : str, optional
            The label of the traces in the plots.
        color: str, optional
            Manually specifying the color of the traces lines.
        radius_sphere : numeric, optional
            Specifies the radius of the spherical bead for force calculations.
            Default 1 mm.
        force_calculator : callable, optional
            The force calculator to use to calculate the force. If none is specified,
            a default calculator will be created using `SphericalHarmonicsForce` of order :math:`ka+3`.
        scale_to_gravity : bool, default True
            Toggles if the force is scaled to gravitational force or not.
        sphere_material : Material
            Specification of the sphere material. Used for the force calculator and gravitational force.

        Returns
        -------
        traces : list
            A list of dictionaries specifying the traces in a format compatible with plotly.

        """
        position = np.asarray(position)
        if sphere_material is None:
            from .materials import styrofoam as sphere_material
        if force_calculator is None:
            from .fields import SphericalHarmonicsForce
            ka = self.array.k * radius_sphere
            orders = int(ka) + 3  # Gives N>ka+2, i.e. two more orders than what we should need
            force_calculator = SphericalHarmonicsForce(array=self.array, radius_sphere=radius_sphere, orders=orders, sphere_material=sphere_material)

        range = range or self.array.wavelength
        n_pos = int(2 * range / self._resolution) + 1
        position_delta = np.linspace(-range, range, n_pos)
        calc_pos = np.tile(position[:, None, None], (1, 3, len(position_delta)))
        calc_pos[0, 0] += position_delta
        calc_pos[1, 1] += position_delta
        calc_pos[2, 2] += position_delta

        force = force_calculator(self.array.complex_amplitudes, calc_pos)
        if scale_to_gravity:
            mg = 9.82 * radius_sphere**3 * np.pi * 4 / 3 * sphere_material.rho
            force /= mg

        if color is None:
            cm = plotly.colors.qualitative.Plotly
            color = cm[_trace_id[0] % len(cm)]

        position_delta /= self._display_scale

        traces = []
        # Fx over x
        traces.append(dict(
            type='scatter', xaxis='x', yaxis='y',
            x=position_delta, y=force[0, 0],
            legendgroup=_trace_id[0], line={'color': color}, name=label,
        ))
        # Fy over x
        traces.append(dict(
            type='scatter', xaxis='x', yaxis='y2',
            x=position_delta, y=force[1, 0],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fz over x
        traces.append(dict(
            type='scatter', xaxis='x', yaxis='y3',
            x=position_delta, y=force[2, 0],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fx over y
        traces.append(dict(
            type='scatter', xaxis='x2', yaxis='y',
            x=position_delta, y=force[0, 1],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fy over y
        traces.append(dict(
            type='scatter', xaxis='x2', yaxis='y2',
            x=position_delta, y=force[1, 1],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fz over y
        traces.append(dict(
            type='scatter', xaxis='x2', yaxis='y3',
            x=position_delta, y=force[2, 1],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fx over z
        traces.append(dict(
            type='scatter', xaxis='x3', yaxis='y',
            x=position_delta, y=force[0, 2],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fy over z
        traces.append(dict(
            type='scatter', xaxis='x3', yaxis='y2',
            x=position_delta, y=force[1, 2],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))
        # Fz over z
        traces.append(dict(
            type='scatter', xaxis='x3', yaxis='y3',
            x=position_delta, y=force[2, 2],
            legendgroup=_trace_id[0], showlegend=False, line={'color': color}, name=label,
        ))

        _trace_id[0] += 1
        return traces

    def force_diagram_layout(self):
        r"""Create a dictionary specifying a layout suitable for force diagrams.

        Specification of a layout used in combination with `force_diagram_traces`
        to visualize the radiation force on a bead around a central point.

        Returns
        -------
        layout : dict
            A dictionary with specifications for the 3x3 matrix of axes
             to show forces around a central point.

        """
        layout_gap_ratio = 12
        total_pieces = layout_gap_ratio * 3 + 2
        width = layout_gap_ratio / total_pieces
        gap = 1 / total_pieces
        length_unit = self.display_scale
        layout = dict(
            xaxis=dict(title=r'$x\text{ in ' + length_unit + '}$', domain=[0, width], anchor='y3'),
            xaxis2=dict(title=r'$y\text{ in ' + length_unit + '}$', domain=[width + gap, 2 * width + gap], anchor='y3'),
            xaxis3=dict(title=r'$z\text{ in ' + length_unit + '}$', domain=[2 * width + 2 * gap, 1], anchor='y3'),
            yaxis=dict(title='$F_x$', domain=[2 * width + 2 * gap, 1]),
            yaxis2=dict(title='$F_y$', domain=[width + gap, 2 * width + gap]),
            yaxis3=dict(title='$F_z$', domain=[0, width]),
        )
        return layout


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
