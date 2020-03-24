"""Visualization classes based on the plotly graphing library."""
# DOCS: This entire module needs documentation...
import collections.abc
import numpy as np
try:
    from plotly.graph_objects import Figure
except ImportError:
    def Figure(data=None, layout=None, **kwargs):
        return dict(data=data, layout=layout, **kwargs)


def _string_formatter(string, format_type):
    if callable(format_type):
            return format_type(string)
    format_type = format_type.lower()
    if format_type == 'html':
        # **...**
        parts = string.split('**')
        parts[1::2] = map('<b>{}</b>'.format, parts[1::2])
        string = ''.join(parts)
        # _{...}
        parts = string.split('_{')
        parts[1:] = map(lambda s: '<sub>' + s.replace('}', '</sub>', 1), parts[1:])
        string = ''.join(parts)
        # _
        parts = string.split('_')
        parts[1:] = map(lambda s: '<sub>' + s[0] + '</sub>' + s[1:], parts[1:])
        string = ''.join(parts)
        # ^{...}
        parts = string.split('^{')
        parts[1:] = map(lambda s: '<sup>' + s.replace('}', '</sup>', 1), parts[1:])
        string = ''.join(parts)
        # ^
        parts = string.split('^')
        parts[1:] = map(lambda s: '<sup>' + s[0] + '</sup>' + s[1:], parts[1:])
        string = ''.join(parts)
        # $...$
        parts = string.split('$')
        parts[1::2] = map('<i>{}</i>'.format, parts[1::2])
        string = ''.join(parts)
        # multiplication dots
        string = string.replace('*', u'\u2219')
        return string
    if format_type == 'latex':
        string = string.replace('*', ' \\cdot ')
        string_parts = string.split('$')
        string_parts[::2] = map('\\text{{{}}}'.format, string_parts[::2])
        return '$' + ''.join(string_parts) + '$'
    return string


def _deepupdate(original, update):
    if not isinstance(original, collections.abc.Mapping):
        return update
    for key, value in update.items():
        if isinstance(value, collections.abc.Mapping):
            original[key] = _deepupdate(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class Visualizer(collections.abc.MutableSequence):
    template = 'plotly_white'
    string_format = 'html'

    def __init__(self, array, *traces, display_scale='mm'):
        self.array = array
        self.display_scale = display_scale
        self._traces = []
        self.extend(traces)

    def __getitem__(self, index):
        return self._traces[index]

    def __setitem__(self, index, value):
        value.visualizer = self
        self._traces[index] = value

    def __delitem__(self, index):
        del self._traces[index]

    def __len__(self):
        return len(self._traces)

    def insert(self, index, value):
        self._traces.insert(index, None)
        self[index] = value

    @property
    def layout(self):
        return dict(template=self.template, font=dict(family='CMU Serif, Latin Modern, Times New Roman'))

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


class ArrayVisualizer(Visualizer):
    """Visualizations of a trandcuer array.

    It is possible to set an item using either just a trace specifier,
    e.g. "Pressure", which create the appropriate trace with default arguments.
    If arguments are required or wanted, set the item to a tuple where the
    first element is the trace specifier, and subsequent elements are the
    arguments. If the last element in the tuple is a dictionary, it will
    be used as keyword arguments for the trace type.
    """

    def __init__(self, array, *args, **kwargs):
        super().__init__(array, *args, **kwargs)

    def __setitem__(self, idx, value):
        if not isinstance(value, Trace):
            # We did not get an actual trace instance
            if not isinstance(value, (tuple, list)):
                # We got just a trace specifier
                trace = value
                args = ()
                kwargs = {}
            else:
                # We got a tuple where the first element is the trace specifier
                trace = value[0]
                if type(value[-1]) is dict:
                    # The last element is a dict of keyword arguments
                    # Intermediate elements are plain arguments
                    kwargs = value[-1]
                    args = value[1:-1]
                else:
                    # No keyword arguments.
                    kwargs = {}
                    args = value[1:]

            if type(trace) is str:
                # The trace type is specified using a string.
                trace = trace.lower()
                if trace == 'pressure':
                    trace = PressureSlice
                elif trace == 'velocity':
                    trace = VelocitySlice
                elif trace == 'force':
                    trace = ForceCones
                elif trace.find('phase') >= 0:
                    trace = TransducerPhase
                elif trace.find('amp') >= 0:
                    trace = TransducerAmplitude
                elif trace.find('signature') >= 0:
                    trace = TransducerSignature
                elif trace.find('trans') >= 0:
                    trace = TransducerTrace
                elif trace.find('path') >= 0:
                    trace = TrapPath
                else:
                    raise NotImplementedError('No implemented trace type matching string "{}"'.format(trace))

            # Instantiate the class with the arguments.
            # Make sure that it uses the correct array.
            value = trace(self.array, *args, **kwargs)
        # Delegate the setting of the element to the superclass.
        # If we were handed a proper trace instance to begin with, we go directly here.
        super().__setitem__(idx, value)

        if isinstance(value, TransducerTrace):
            for trace in self:
                if type(trace) is TransducerTrace and trace is not value:
                    self.remove(trace)  # We don't need to keep the plain transducer trace around of we add transducer traces with data.

    @property
    def layout(self):
        return _deepupdate(
            super().layout, dict(scene=dict(
                aspectmode='data',
                xaxis=dict(title=_string_formatter('$x$ in ' + self.display_scale, self.string_format)),
                yaxis=dict(title=_string_formatter('$y$ in ' + self.display_scale, self.string_format)),
                zaxis=dict(title=_string_formatter('$z$ in ' + self.display_scale, self.string_format))
            )))

    def projection_layout(self, plane='xz', scale=None, layout=None):
        scale = 1 if scale is None else scale
        new = {'scene': {'aspectmode': 'manual', 'camera': {'projection': {'type': 'orthographic'}}}}
        if plane == 'xy':
            new['scene']['zaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=0, y=0, z=1)
            new['scene']['camera']['up'] = dict(x=0, y=1, z=0)
            new['scene']['aspectratio'] = dict(x=scale, y=scale, z=1)
        elif plane == 'yx':
            new['scene']['zaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=0, y=0, z=-1)
            new['scene']['camera']['up'] = dict(x=1, y=0, z=0)
            new['scene']['aspectratio'] = dict(x=scale, y=scale, z=1)
        elif plane == 'xz':
            new['scene']['yaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=0, y=-1, z=0)
            new['scene']['camera']['up'] = dict(x=0, y=0, z=1)
            new['scene']['aspectratio'] = dict(x=scale, y=1, z=scale)
        elif plane == 'zx':
            new['scene']['yaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=0, y=1, z=0)
            new['scene']['camera']['up'] = dict(x=1, y=0, z=0)
            new['scene']['aspectratio'] = dict(x=scale, y=1, z=scale)
        elif plane == 'yz':
            new['scene']['xaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=1, y=0, z=0)
            new['scene']['camera']['up'] = dict(x=0, y=0, z=1)
            new['scene']['aspectratio'] = dict(x=1, y=scale, z=scale)
        elif plane == 'zy':
            new['scene']['xaxis'] = {'visible': False}
            new['scene']['camera']['eye'] = dict(x=-1, y=0, z=0)
            new['scene']['camera']['up'] = dict(x=0, y=1, z=0)
            new['scene']['aspectratio'] = dict(x=1, y=scale, z=scale)
        else:
            raise ValueError('Unknown projection plane `{}`'.format(plane))
        # TODO: Handle plotly Layout objects, plotly Figures, and dicts corresponding to figures.
        # The difficulty is that Layout objects and Figures doesn't have a setdefault method.
        return _deepupdate({} if layout is None else layout, new)

    def __call__(self, *complex_transducer_amplitudes, **kwargs):
        traces = []
        transducer_trace_idx = []
        field_trace_idx = []
        if len(complex_transducer_amplitudes) == 0:
            complex_transducer_amplitudes = [None]
        elif len(complex_transducer_amplitudes) == 1:
            button_labels = ['{}']
        else:
            if 'labels' in kwargs:
                labels = kwargs.pop('labels')
                button_labels = [labels[idx] + ' | {}' for idx in range(len(complex_transducer_amplitudes))]
            else:
                label = kwargs.pop('label', 'State')
                button_labels = [label + ' {} | {{}}'.format(idx) for idx in range(len(complex_transducer_amplitudes))]

        trace_idx = 0
        for data_idx, data in enumerate(complex_transducer_amplitudes):
            for self_idx, trace in enumerate(self):
                if isinstance(trace, TransducerTrace):
                    if data_idx > 0 and type(trace) is TransducerTrace:
                        continue  # No reason to show a plain transducer visualization more than once.
                    transducer_trace_idx.append((data_idx, self_idx, trace_idx))
                elif isinstance(trace, FieldTrace):
                    field_trace_idx.append((data_idx, self_idx, trace_idx))
                traces.append(trace(data))
                trace_idx += 1
        del trace_idx

        updatemenus = []
        n_trans = len(transducer_trace_idx)
        if n_trans > 1:
            buttons = []
            active_on = [idx[2] for idx in transducer_trace_idx]
            for trans_idx, (data_idx, self_idx, trace_idx) in enumerate(transducer_trace_idx):
                buttons.append(dict(
                    method='restyle', label=button_labels[data_idx].format(self[self_idx].name),
                    args=[{'visible': trans_idx * [False] + [True] + (n_trans - trans_idx - 1) * [False]}, active_on],
                ))
                if trans_idx > 0:
                    traces[trace_idx]['visible'] = False
            updatemenus.append(dict(active=0, buttons=buttons, type='buttons', direction='down', x=-0.02, xanchor='right'))

        n_fields = len(field_trace_idx)
        if n_fields > 1:
            buttons = []
            active_on = [idx[2] for idx in field_trace_idx]
            for field_idx, (data_idx, self_idx, trace_idx) in enumerate(field_trace_idx):
                buttons.append(dict(
                    method='restyle', label=button_labels[data_idx].format(self[self_idx].name),
                    args=[{'visible': field_idx * [False] + [True] + (n_fields - field_idx - 1) * [False]}, active_on],
                ))
                if field_idx > 0:
                    traces[trace_idx]['visible'] = False
            updatemenus.append(dict(active=0, buttons=buttons, type='buttons', direction='down', x=1.02, xanchor='left'))
        layout = _deepupdate(self.layout, {'updatemenus': updatemenus})
        if 'projection' in kwargs:
            self.projection_layout(layout=layout, plane=kwargs.pop('projection'), scale=kwargs.pop('scale', None))
        return Figure(data=traces, layout=layout)


class Trace:
    label = ''
    name = ''

    def __init__(self, array):
        self.array = array

    @property
    def display_scale(self):
        try:
            return self.visualizer._display_scale
        except AttributeError:
            return 1

    @property
    def string_format(self):
        try:
            return self._stirng_format
        except AttributeError:
            try:
                return self.visualizer.string_format
            except AttributeError:
                return Visualizer.stirng_format

    @string_format.setter
    def string_format(self, val):
        self._string_format = val


class TrapPath(Trace):
    label = 'Trap path'
    name = 'Trap path'

    _default_args = dict(
        tolerance=0.1e-3,
        time_interval=10000,
        path_points=200,
    )

    def __init__(self, array, start_position, **kwargs):
        super().__init__(array)
        self.start_position = start_position
        self.kwargs = dict(self._default_args, **kwargs)

    def __call__(self, complex_transducer_amplitudes):
        from .utils import find_trap
        path = find_trap(array=self.array, start_position=self.start_position,
                         complex_transducer_amplitudes=complex_transducer_amplitudes,
                         **self.kwargs)
        path /= self.display_scale
        return dict(
            type='scatter3d',
            x=path[0], y=path[1], z=path[2],
        )


class MeshTrace(Trace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mesh_is_outdated = True

    def _generate_mesh(self):
        raise NotImplementedError('Subclasses of `MeshTrace` has to implement `_generate_mesh`')

    def __update_mesh(self, force=False):
        if force or self.__mesh_is_outdated:
            self.mesh = self._generate_mesh()

    def __call__(self):
        self.__update_mesh()

    @property
    def mesh(self):
        self.__update_mesh()
        return self.__mesh

    @mesh.setter
    def mesh(self, value):
        self.__mesh = value
        self.__mesh_is_outdated = False

    class meshproperty:
        def __set_name__(self, obj, name):
            self.name = '_' + name

        def __init__(self, fpre=None, fpost=None):
            self.fpre = fpre or (lambda self, value: value)
            self.fpost = fpost or (lambda self, value: value)
            self.__doc__ = fpre.__doc__

        def __get__(self, obj, obj_type=None):
            return self.fpost(obj, getattr(obj, self.name))

        def __set__(self, obj, value):
            setattr(obj, self.name, self.fpre(obj, value))
            obj._MeshTrace__mesh_is_outdated = True

        def __delete__(self, obj):
            delattr(obj, self.name)

        def preprocessor(self, fpre):
            return type(self)(fpre, self.fpost)

        def postprocessor(self, fpost):
            return type(self)(self.fpre, fpost)


class FieldTrace(MeshTrace):
    colorscale = 'Viridis'

    def __init__(self, array, field=None, **field_kwargs):
        super().__init__(array)
        if field is not None:
            if isinstance(field, type):
                self.field = field(array, **field_kwargs)
            else:
                self.field = field
        elif hasattr(self, '_field_class'):
            self.field = self._field_class(array, **field_kwargs)
        else:
            raise ValueError('Class {} has no field class, and no field was supplied!'.format(self.__class__.__name__))

    def __call__(self, complex_transducer_amplitudes=None):
        super().__call__()
        return self.postprocessing(self.field(self.preprocessing(complex_transducer_amplitudes)))

    def preprocessing(self, complex_transducer_amplitudes):
        return complex_transducer_amplitudes

    def postprocessing(self, field_data):
        return field_data

    @property
    def mesh(self):
        return super().mesh

    @mesh.setter
    def mesh(self, value):
        # This accesses the setter implemented in the superclass.
        # It's slightly annoying that there is no really convenient way to
        # get the setter of a property in the superclass.
        super(FieldTrace, type(self)).mesh.fset(self, value)
        # Rebind the field to the new mesh.
        # Doing this in a setter makes sure that if the user changes the mesh manually,
        # the field will still match the mesh.
        self.field = self.field @ self.mesh


class TransducerTrace(MeshTrace):
    name = 'Transducers'
    radius_ratio = 3 / 2
    height = 5e-3
    num_vertices = 10
    colorscale = 'Greys'
    showscale = False
    cmin = 0
    cmax = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def data_map(self, complex_transducer_amplitudes=None):
        return np.zeros(self.array.num_transducers)

    def __call__(self, complex_transducer_amplitudes=None):
        super().__call__()
        coordinates, indices = self.mesh
        viz_data = self.data_map(complex_transducer_amplitudes)
        intensity = np.repeat(viz_data, coordinates.shape[1] // len(viz_data))

        return dict(
            type='mesh3d',
            x=coordinates[0] / self.display_scale, y=coordinates[1] / self.display_scale, z=coordinates[2] / self.display_scale,
            i=indices[0], j=indices[1], k=indices[2],
            intensity=intensity,
            colorscale=self.colorscale, showscale=self.showscale,
            colorbar={'title': {'text': _string_formatter(self.label, self.string_format), 'side': 'right'}, 'x': -0.02, 'xanchor': 'left'},
            cmin=self.cmin, cmax=self.cmax,
        )

    def _generate_mesh(self):
        N = self.num_vertices
        # Each transducer coordinates will have first N point in the upper circle, then N points in the lower circle
        i = []
        j = []
        k = []
        # Upper disk
        i += [0] * (N - 2)
        j += list(range(1, N - 1))
        k += list(range(2, N))
        # Lower disk
        i += [N] * (N - 2)
        j += list(range(N + 1, 2 * N - 1))
        k += list(range(N + 2, 2 * N))
        # Half of the side
        i += list(range(0, N))
        j += list(range(N, 2 * N))
        k += list(range(1, N)) + [0]
        # Other half of the side
        i += list(range(N, 2 * N))
        j += list(range(N + 1, 2 * N)) + [N]
        k += list(range(1, N)) + [0]

        base_indices = np.stack([i, j, k], axis=0)

        upper_radius = self.array.transducer_size / 2
        lower_radius = upper_radius / self.radius_ratio
        theta = np.arange(N) / N * np.pi * 2
        cos = np.cos(theta)
        sin = np.sin(theta)

        coordinates = []
        indices = []
        for t_idx in range(self.array.num_transducers):
            pos = self.array.positions[:, t_idx]
            n = self.array.normals[:, t_idx]
            n_max = np.argmax(np.abs(n))
            v1 = np.array([1., 1., 1.])
            v1[n_max] = -(np.sum(n) - n[n_max]) / n[n_max]
            v2 = np.cross(v1, n)
            v1 /= np.sum(v1**2)**0.5
            v2 /= np.sum(v2**2)**0.5

            circle = cos * v1[:, None] + sin * v2[:, None]
            upper_circle = circle * upper_radius + pos[:, None]
            lower_circle = circle * lower_radius + pos[:, None] - n[:, None] * self.height
            coordinates.append(np.concatenate([upper_circle, lower_circle], axis=1))
            indices.append(base_indices + t_idx * 2 * N)

        indices = np.concatenate(indices, axis=1)
        coordinates = np.concatenate(coordinates, axis=1)
        return (coordinates, indices)


class TransducerPhase(TransducerTrace):
    cmin = -1
    cmax = 1
    colorscale = 'Phase'
    name = 'Phase'
    label = 'Transducer phase in rad/π'
    showscale = True

    def data_map(self, complex_transducer_amplitudes):
        return np.angle(complex_transducer_amplitudes) / np.pi


class TransducerSignature(TransducerPhase):
    name = 'Signature at '
    label = 'Transducer signature in rad/π'

    def __init__(self, array, position, *args, **kwargs):
        super().__init__(array, *args, **kwargs)
        self.position = position
        self.name = self.name + str(position)

    def data_map(self, complex_transducer_amplitudes):
        return self.array.signature(position=self.position, phases=np.angle(complex_transducer_amplitudes)) / np.pi


class TransducerAmplitude(TransducerTrace):
    cmin = 0
    cmax = 1
    colorscale = 'Viridis'
    name = 'Amplitude'
    label = 'Transducer normalized amplitude'
    showscale = True

    def data_map(self, complex_transducer_amplitudes):
        return np.abs(complex_transducer_amplitudes)


class ScalarFieldSlice(FieldTrace):
    cmin = None
    cmax = None

    def __init__(self, array, intersect=None, normal=None, xlimits=None, ylimits=None, zlimits=None, resolution=10, **kwargs):
        super().__init__(array, **kwargs)

        self.resolution = resolution
        self.normal = normal if normal is not None else (0, 1, 0)
        self.intersect = intersect if intersect is not None else (0, 0, 0)

        xlimits = xlimits or (np.min(array.positions[0]), np.max(array.positions[0]))
        ylimits = ylimits or (np.min(array.positions[1]), np.max(array.positions[1]))
        if 'Doublesided' not in type(array).__name__:
            # Singlesided array, one of the limits will give a zero range.
            # Assuming that the array is in the xy-plane, pointing up
            zlimits = zlimits or (np.min(array.positions[2]) + 1e-3, np.min(array.positions[2]) + 20 * array.wavelength)
        else:
            zlimits = zlimits or (np.min(array.positions[2]), np.max(array.positions[2]))
        xlimits = (min(xlimits), max(xlimits))
        ylimits = (min(ylimits), max(ylimits))
        zlimits = (min(zlimits), max(zlimits))
        if xlimits[0] == xlimits[1]:
            xlimits = (xlimits[0], xlimits[1] + 20 * array.wavelength)
        if ylimits[0] == ylimits[1]:
            ylimits = (ylimits[0], ylimits[1] + 20 * array.wavelength)
        if zlimits[0] == zlimits[1]:
            zlimits = (zlimits[0], zlimits[1] + 20 * array.wavelength)

        self.xlimits = xlimits
        self.ylimits = ylimits
        self.zlimits = zlimits

    @MeshTrace.meshproperty
    def normal(self, value):
        value = np.asarray(value, dtype=float)
        return value / np.sum(value**2)**0.5

    @MeshTrace.meshproperty
    def intersect(self, value):
        return np.asarray(value, dtype=float)

    @MeshTrace.meshproperty
    def resolution(self, val):
        return self.array.wavelength / val

    @resolution.postprocessor
    def resolution(self, val):
        return self.array.wavelength / val

    xlimits = MeshTrace.meshproperty()
    ylimits = MeshTrace.meshproperty()
    zlimits = MeshTrace.meshproperty()

    def _generate_mesh(self):
        # Find two vectors that span the plane
        v1 = np.array([1., 1., 1.])
        n_max = np.argmax(np.abs(self.normal))
        v1[n_max] = -(np.sum(self.normal) - self.normal[n_max]) / self.normal[n_max]
        v2 = np.cross(self.normal, v1)
        v1 /= np.sum(v1**2)**0.5
        v2 /= np.sum(v2**2)**0.5

        v1 *= self._resolution
        v2 *= self._resolution
        # v1 and v2 are in the plane, orthogonal, and with the correct length to get our target resolution

        # Get the bounding box, shifted so that the intersect point is at (0,0,0)
        xmin, xmax = self.xlimits
        ymin, ymax = self.ylimits
        zmin, zmax = self.zlimits

        xmin = xmin - self.intersect[0]
        xmax = xmax - self.intersect[0]
        ymin = ymin - self.intersect[1]
        ymax = ymax - self.intersect[1]
        zmin = zmin - self.intersect[2]
        zmax = zmax - self.intersect[2]

        # Intersection of the bounding box and the plane
        nx, ny, nz = self.normal
        edge_intersections = []
        if nx != 0:
            edge_intersections.extend([np.array([-(ny * yl + nz * zl) / nx, yl, zl]) for yl in (ymin, ymax) for zl in (zmin, zmax)])
        if ny != 0:
            edge_intersections.extend([np.array([xl, -(nx * xl + nz * zl) / ny, zl]) for xl in (xmin, xmax) for zl in (zmin, zmax)])
        if nz != 0:
            edge_intersections.extend([np.array([xl, yl, -(nx * xl + ny * xl) / nz]) for xl in (xmin, xmax) for yl in (ymin, ymax)])

        # Find how much of each of the vectors is needed to get to the bounding box.
        v1_min = v1_max = v2_min = v2_max = 0
        V = np.stack([v1, v2], axis=1)
        for pi in edge_intersections:
            (w1, w2), _, _, _ = np.linalg.lstsq(V, pi, rcond=None)
            v1_min = min(v1_min, w1)
            v1_max = max(v1_max, w1)
            v2_min = min(v2_min, w2)
            v2_max = max(v2_max, w2)

        # Round outwards to integer values
        v1_min = np.floor(v1_min).astype(int)
        v2_min = np.floor(v2_min).astype(int)
        v1_max = np.ceil(v1_max).astype(int)
        v2_max = np.ceil(v2_max).astype(int)

        w1 = np.arange(v1_min, v1_max + 1).reshape((1, -1, 1))
        w2 = np.arange(v2_min, v2_max + 1).reshape((1, 1, -1))

        # The mesh is creates by stepping v1 and v2 with the correct number of times
        # The mesh is then filtered to only keep the values within the bounding box.
        mesh = v1.reshape((3, 1, 1)) * w1 + v2.reshape((3, 1, 1)) * w2
        idx = (
            (xmin <= mesh[0]) & (mesh[0] <= xmax)
            & (ymin <= mesh[1]) & (mesh[1] <= ymax)
            & (zmin <= mesh[2]) & (mesh[2] <= zmax)
        )

        # Reshift the mesh to the actual coordinates so that it intersects the intersect point.
        return mesh[:, idx] + self.intersect.reshape((3, 1))

    def __call__(self, complex_transducer_amplitudes=None):
        return dict(
            type='mesh3d', intensity=super().__call__(complex_transducer_amplitudes),
            cmin=self.cmin, cmax=self.cmax, colorscale=self.colorscale,
            colorbar={'title': {'text': _string_formatter(self.label, self.string_format), 'side': 'right'}, 'x': 1.02, 'xanchor': 'right'},
            x=np.squeeze(self.mesh[0]) / self.display_scale,
            y=np.squeeze(self.mesh[1]) / self.display_scale,
            z=np.squeeze(self.mesh[2]) / self.display_scale,
            delaunayaxis='xyz'[np.argmax(np.abs(self.normal))],
        )


class PressureSlice(ScalarFieldSlice):
    name = 'Pressure'
    label = 'Sound pressure in dB re. 20 µPa'
    from .fields import Pressure as _field_class
    from .utils import SPL
    postprocessing = staticmethod(SPL)
    cmin = 130
    cmax = 170


class VelocitySlice(ScalarFieldSlice):
    name = 'Velocity'
    label = 'Particle velocity in dB re. 50 nm/s'
    from .fields import Velocity as _field_class
    from .utils import SVL
    postprocessing = staticmethod(SVL)
    cmin = 130
    cmax = 170


class VectorFieldCones(FieldTrace):
    opacity = 0.5
    cone_length = 0.75
    cone_vertices = 10
    cone_ratio = 3
    cmin = None
    cmax = None

    def __init__(self, array, center, resolution=5,
                 xrange=None, yrange=None, zrange=None, **kwargs):
        super().__init__(array, **kwargs)

        self.center = center
        self.resolution = resolution
        self.xrange = xrange if xrange is not None else self.array.wavelength
        self.yrange = yrange if yrange is not None else self.array.wavelength
        self.zrange = zrange if zrange is not None else self.array.wavelength

    @MeshTrace.meshproperty
    def resolution(self, val):
        return self.array.wavelength / val

    @resolution.postprocessor
    def resolution(self, val):
        return self.array.wavelength / val

    center = MeshTrace.meshproperty()
    xrange = MeshTrace.meshproperty()
    yrange = MeshTrace.meshproperty()
    zrange = MeshTrace.meshproperty()

    def _generate_mesh(self):
        nx = int(2 * np.ceil(self.xrange / self._resolution) + 1)
        ny = int(2 * np.ceil(self.yrange / self._resolution) + 1)
        nz = int(2 * np.ceil(self.zrange / self._resolution) + 1)

        x = np.linspace(-self.xrange, self.xrange, nx) + self.center[0]
        y = np.linspace(-self.yrange, self.yrange, ny) + self.center[1]
        z = np.linspace(-self.zrange, self.zrange, nz) + self.center[2]

        return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0).reshape((3, -1))

    def __call__(self, complex_transducer_amplitudes=None):
        field_data = super().__call__(complex_transducer_amplitudes)
        vertex_indices, vertex_coordinates, vertex_intensities = self._generate_vertices(field_data)
        return dict(
            type='mesh3d',
            x=vertex_coordinates[0], y=vertex_coordinates[1], z=vertex_coordinates[2],
            i=vertex_indices[0], j=vertex_indices[1], k=vertex_indices[2],
            intensity=vertex_intensities, opacity=self.opacity, colorscale=self.colorscale, cmin=self.cmin, cmax=self.cmax,
            colorbar={'title': {'text': _string_formatter(self.label, self.string_format), 'side': 'right'}, 'x': 1.02, 'xanchor': 'right'},
        )

    def _generate_vertices(self, field_data):
        centers = self.mesh
        intensity = np.sum(np.abs(field_data)**2, axis=0)**0.5
        normals = field_data / intensity

        # n_vertices is the number of vertices in the base of the cone
        n_vertices = self.cone_vertices
        i = [0] * n_vertices
        j = list(range(1, n_vertices + 1))
        k = list(range(2, n_vertices + 1)) + [1]
        i += [1] * (n_vertices - 2)
        j += list(range(2, n_vertices))
        k += list(range(3, n_vertices + 1))
        base_indices = np.stack([i, j, k], axis=0)

        theta = np.arange(n_vertices) / n_vertices * 2 * np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)

        vertex_indices = []
        vertex_coordinates = []
        vertex_intensities = []
        back_length = self._resolution * self.cone_length / 3
        forward_length = self._resolution * self.cone_length / 3 * 2
        radius = 0.5 * self._resolution * self.cone_length / self.cone_ratio

        for cone_idx in range(len(intensity)):
            # Find two vectors that sweep the surface of the base
            c = centers[:, cone_idx]
            n = normals[:, cone_idx]
            n_max = np.argmax(np.abs(n))
            v1 = np.array([1., 1., 1.])
            v1[n_max] = -(np.sum(n) - n[n_max]) / n[n_max]
            v2 = np.cross(v1, n)
            v1 /= np.sum(v1**2)**0.5
            v2 /= np.sum(v2**2)**0.5

            circle = cos * v1[:, None] + sin * v2[:, None]
            circle = circle * radius - n[:, None] * back_length

            vertex_indices.append(base_indices + (n_vertices + 1) * cone_idx)
            vertex_intensities.append([intensity[cone_idx]] * (n_vertices + 1))
            vertex_coordinates.append(np.concatenate([n[:, None] * forward_length, circle], axis=1) + c[:, None])

        vertex_indices = np.concatenate(vertex_indices, axis=1)
        vertex_intensities = np.concatenate(vertex_intensities, axis=0)
        vertex_coordinates = np.concatenate(vertex_coordinates, axis=1) / self.display_scale
        return vertex_indices, vertex_coordinates, vertex_intensities


class ForceCones(VectorFieldCones):
    name = 'Force'

    def __init__(self, *args, add_gravity=True, scale_to_gravity=True, field=None, **kwargs):
        if field is None:
            if 'radius' in kwargs:
                from .fields import SphericalHarmonicsForce as field
            else:
                from .fields import RadiationForce as field
        super().__init__(*args, field=field, **kwargs)
        self.add_gravity = add_gravity
        self.scale_to_gravity = scale_to_gravity

    def postprocessing(self, field_data):
        if self.add_gravity:
            field_data[2] -= self.field.field.mg
        if self.scale_to_gravity:
            field_data /= self.field.field.mg
        return field_data

    @property
    def label(self):
        try:
            if self.scale_to_gravity:
                return "|$F$|$/mg$"
        except AttributeError:
            pass
        return 'Force magnitude in N'


class ForceDiagram(Visualizer):
    def __init__(self, *args, scale_to_gravity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_to_gravity = scale_to_gravity

    def __setitem__(self, index, value):
        if type(value) is not ForceDiagram.ForceTrace:
            if ((isinstance(value, (tuple, list))
                    and all([isinstance(v, (float, int)) for v in value]))
               or (isinstance(value, np.ndarray))):
                # Value is either a list/tuple
                #   with elements which are all flots or ints
                # or a numpy array.
                # I.e. the value is the center position.
                center = value
                args = ()
                kwargs = {}
            elif type(value[-1]) is dict:
                center = value[0]
                args = value[1:-1]
                kwargs = value[-1]
            else:
                center = value[0]
                args = value[1:]
            value = self.ForceTrace(self.array, center, *args, **kwargs)
        super().__setitem__(index, value)

    def __call__(self, *complex_transducer_amplitudes, **kwargs):
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        all_traces = []
        if len(complex_transducer_amplitudes) > 1 and len(self) > 1:
            # We could possibly use markers to distinguish between the sets of complex amplitudes,
            # and colors to distinguing between forces (or the other way around).
            # There is a "maxdisplayed" property for markers, which is probably useful
            # to avoid clutter in the plots. I have not found any list of available marker shapers,
            # so that probably has to be hardcoded here.
            raise NotImplementedError('Showing force diagrams for mutiple array settings with multiple forces is not available!')
        elif len(complex_transducer_amplitudes) > 1:
            field = self[0]
            for idx, data in enumerate(complex_transducer_amplitudes):
                this_state_traces = field(
                    data, line=kwargs['line'][idx] if 'line' in kwargs else dict(color=colors[idx % len(colors)]),
                    name=_string_formatter(kwargs['name'][idx] if 'name' in kwargs else 'Data {}'.format(idx), self.string_format),
                )
                all_traces.extend(this_state_traces)
        elif len(self) > 1:
            data = complex_transducer_amplitudes[0]
            for idx, field in enumerate(self):
                this_field_traces = field(
                    data, line=kwargs['line'][idx] if 'line' in kwargs else dict(color=colors[idx % len(colors)]),
                    name=_string_formatter(kwargs['name'][idx] if 'name' in kwargs else field.name if field.name is not '' else 'Force {}'.format(idx), self.string_format),
                )
                all_traces.extend(this_field_traces)
        else:
            data = complex_transducer_amplitudes[0]
            field = self[0]
            all_traces = field(data, line=kwargs['line'] if 'line' in kwargs else dict(color=colors[0]))
            all_traces[0]['showlegend'] = False

        return Figure(data=all_traces, layout=self.layout)

    @property
    def layout(self):
        layout_gap_ratio = 12
        total_pieces = layout_gap_ratio * 3 + 2
        width = layout_gap_ratio / total_pieces
        gap = 1 / total_pieces
        length_unit = ' in {}'.format(self.display_scale)
        force_unit = '$/mg$' if self.scale_to_gravity else ' in N'
        return _deepupdate(
            super().layout, dict(
                xaxis=dict(title=_string_formatter('$x$' + length_unit, self.string_format), domain=[0, width], anchor='y3'),
                xaxis2=dict(title=_string_formatter('$y$' + length_unit, self.string_format), domain=[width + gap, 2 * width + gap], anchor='y3'),
                xaxis3=dict(title=_string_formatter('$z$' + length_unit, self.string_format), domain=[2 * width + 2 * gap, 1], anchor='y3'),
                yaxis=dict(title=_string_formatter('$F_x$' + force_unit, self.string_format), domain=[2 * width + 2 * gap, 1]),
                yaxis2=dict(title=_string_formatter('$F_y$' + force_unit, self.string_format), domain=[width + gap, 2 * width + gap]),
                yaxis3=dict(title=_string_formatter('$F_z$' + force_unit, self.string_format), domain=[0, width]),
            ))

    class ForceTrace(FieldTrace):
        _trace_objects = 0

        def __init__(self, array, center, *args,
                     resolution=25, xrange=None, yrange=None, zrange=None,
                     field=None, name=None, **kwargs):
            if field is None:
                if 'radius' in kwargs:
                    from .fields import SphericalHarmonicsForce as field
                else:
                    from .fields import RadiationForce as field
            super().__init__(array, *args, field=field, **kwargs)
            self._trace_id = self._trace_objects
            self._trace_objects += 1
            self._trace_calls = 0
            if name is not None:
                self.name = name

            self.center = center
            self.resolution = resolution
            self.xrange = xrange if xrange is not None else self.array.wavelength
            self.yrange = yrange if yrange is not None else self.array.wavelength
            self.zrange = zrange if zrange is not None else self.array.wavelength

        @property
        def scale_to_gravity(self):
            try:
                return self.visualizer.scale_to_gravity
            except AttributeError:
                return False

        @MeshTrace.meshproperty
        def resolution(self, val):
            return self.array.wavelength / val

        @resolution.postprocessor
        def resolution(self, val):
            return self.array.wavelength / val

        center = MeshTrace.meshproperty()
        xrange = MeshTrace.meshproperty()
        yrange = MeshTrace.meshproperty()
        zrange = MeshTrace.meshproperty()

        def _generate_mesh(self):
            nx = int(2 * np.ceil(self.xrange / self._resolution) + 1)
            ny = int(2 * np.ceil(self.yrange / self._resolution) + 1)
            nz = int(2 * np.ceil(self.zrange / self._resolution) + 1)

            self._xidx = np.arange(0, nx)
            self._yidx = np.arange(nx, nx + ny)
            self._zidx = np.arange(nx + ny, nx + ny + nz)

            x = np.linspace(-self.xrange, self.xrange, nx) + self.center[0]
            y = np.linspace(-self.yrange, self.yrange, ny) + self.center[1]
            z = np.linspace(-self.zrange, self.zrange, nz) + self.center[2]

            X = np.stack([x, np.repeat(self.center[1], nx), np.repeat(self.center[2], nx)], axis=0)
            Y = np.stack([np.repeat(self.center[0], ny), y, np.repeat(self.center[2], ny)], axis=0)
            Z = np.stack([np.repeat(self.center[0], nz), np.repeat(self.center[1], nz), z], axis=0)

            return np.concatenate([X, Y, Z], axis=1)

        def __call__(self, complex_transducer_amplitudes, **kwargs):
            force = super().__call__(complex_transducer_amplitudes)
            if self.scale_to_gravity:
                force /= self.field.field.mg

            dx = (self.mesh[0, self._xidx] - self.center[0]) / self.display_scale
            dy = (self.mesh[1, self._yidx] - self.center[1]) / self.display_scale
            dz = (self.mesh[2, self._zidx] - self.center[2]) / self.display_scale

            unique_id = 'obj{}_call{}'.format(self._trace_id, self._trace_calls)
            self._trace_calls += 1

            traces = []
            # Fx over x
            traces.append(dict(
                type='scatter', xaxis='x', yaxis='y', x=dx, y=force[0, self._xidx],
                legendgroup=unique_id, showlegend=True, **kwargs,
            ))
            # Fy over x
            traces.append(dict(
                type='scatter', xaxis='x', yaxis='y2', x=dx, y=force[1, self._xidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fz over x
            traces.append(dict(
                type='scatter', xaxis='x', yaxis='y3', x=dx, y=force[2, self._xidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fx over y
            traces.append(dict(
                type='scatter', xaxis='x2', yaxis='y', x=dy, y=force[0, self._yidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fy over y
            traces.append(dict(
                type='scatter', xaxis='x2', yaxis='y2', x=dy, y=force[1, self._yidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fz over y
            traces.append(dict(
                type='scatter', xaxis='x2', yaxis='y3', x=dy, y=force[2, self._yidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fx over z
            traces.append(dict(
                type='scatter', xaxis='x3', yaxis='y', x=dz, y=force[0, self._zidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fy over z
            traces.append(dict(
                type='scatter', xaxis='x3', yaxis='y2', x=dz, y=force[1, self._zidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            # Fz over z
            traces.append(dict(
                type='scatter', xaxis='x3', yaxis='y3', x=dz, y=force[2, self._zidx],
                legendgroup=unique_id, showlegend=False, **kwargs,
            ))
            return traces
