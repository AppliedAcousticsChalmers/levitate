"""Visualization methods based on the plotly graphing library, and some convenience functions."""
import collections.abc
import numpy as np
from .utils import SPL, SVL
import plotly.graph_objects as go
import plotly.colors


"""Notes:
Write individual classes for the types of fields to visualize: one class
for visualizing slices of scalar fields, e.g. pressure of velocity,
and a different class to visualize e.g. the force diagrams, or the force vector field.

You should consider creating your own visualizer for force fields, in a similar way
that you visualize the transducers. Mesh3d plots can have opacity, which might be nice
for force vector plots.

Create some kind of main class to which the individual field visualizes are appended.
When called, the main class should return a plotly figure.
The individual field classes should probably return plotly traces.
The figure should have all the information already there, so the annoying calls to `selection_figure`
should not be needed.

We want to visualize
3d:
Should have a drop down menu to select what is shown as the field and on the transducers.
We could allow for showing multiple traces at once. It is not that
difficult to just deselect one of the traces when selecting one of the other traces.
We might want to allow naming the traces, defaulting to something useful.
Should visualizations using multiple sets of complex amplitudes be possible?
I'm thinking no, just show them in separate figures if needed.
- Scalar field slices (specify a slice plane, perhaps using the normal and a distance?) -> surface
- Array geometry + transducer data -> mesh3d
- Vector fields (specify a meshing method or a given mesh) -> mesh3d
- Additional objects, e.g. spherical markers or a given radius. -> mesh3d

2d:
We want to have essentially the same kind of options here, only that the render method needs to be different.
There is no longer any need to have fancy ways of switching between the plots. This is not primarily for interactive use
but for publication ready plots. Perhaps this is not really needed that much?
Only for certain common plots? Write some scripts or something? Don't add this to the package?
- Scalar field slices -> heatmap (not needed?)
- Vector fields -> quiver, super annoying to use, perhaps write a script that does this instead and don't have it as part of the package


Force diagram. This is the only really useful 2d plot. It is not possible to integrate with any other plots, and it's annoying to
recreate every time. This should probably be kept somewhere, but perhaps not inside the package if the 3d force visualization turns out nice.

"""


class Visualizer(collections.abc.MutableSequence):
    template = 'plotly_white'

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
        return dict(template=self.template)

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
    def __init__(self, array, **kwargs):
        super().__init__(array, **kwargs)

    @property
    def layout(self):
        return dict(super().layout, scene=dict(aspectmode='data'))

    def __call__(self, complex_transducer_amplitudes):
        traces = []
        transducer_trace_idx = []
        field_trace_idx = []
        for trace_idx, trace in enumerate(self):
            if isinstance(trace, TransducerTrace):
                transducer_trace_idx.append(trace_idx)
            elif isinstance(trace, FieldTrace):
                field_trace_idx.append(trace_idx)
            traces.append(trace(complex_transducer_amplitudes))

        updatemenus = []
        n_trans = len(transducer_trace_idx)
        if n_trans > 1:
            buttons = []
            for trans_idx, trace_idx in enumerate(transducer_trace_idx):
                buttons.append(dict(
                    method='restyle', label=self[trace_idx].name,
                    args=[{'visible': trans_idx * [False] + [True] + (n_trans - trans_idx - 1) * [False]}, transducer_trace_idx],
                ))
                if trans_idx > 0:
                    traces[trace_idx]['visible'] = False
            updatemenus.append(dict(active=0, buttons=buttons, type='buttons', direction='down', x=-0.02, xanchor='right'))

        n_fields = len(field_trace_idx)
        if n_fields > 1:
            buttons = []
            for field_idx, trace_idx in enumerate(field_trace_idx):
                buttons.append(dict(
                    method='restyle', label=self[trace_idx].name,
                    args=[{'visible': field_idx * [False] + [True] + (n_fields - field_idx - 1) * [False]}, field_trace_idx],
                ))
                if field_idx > 0:
                    traces[trace_idx]['visible'] = False
            updatemenus.append(dict(active=0, buttons=buttons, type='buttons', direction='down', x=1.02, xanchor='left'))
        layout = dict(self.layout, updatemenus=updatemenus)
        return go.Figure(data=traces, layout=layout)


class MeshTrace:
    label = ''
    name = ''

    @property
    def display_scale(self):
        try:
            return self.visualizer._display_scale
        except AttributeError:
            return 1

    def _update_mesh(self):
        raise NotImplementedError('Subclasses of `MeshTrace` has to implement `_update_mesh`')

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
            obj._update_mesh()

        def __delete__(self, obj):
            delattr(obj, self.name)

        def preprocessor(self, fpre):
            return type(self)(fpre, self.fpost)

        def postprocessor(self, fpost):
            return type(self)(self.fpre, fpost)


class FieldTrace(MeshTrace):
    colorscale = 'Viridis'
    preprocessors = []
    postprocessors = []

    def __init__(self, array, field=None, **field_kwargs):
        self.array = array
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
        for pp in self.preprocessors:
            complex_transducer_amplitudes = pp(self, complex_transducer_amplitudes)
        data = self.field(complex_transducer_amplitudes)
        for pp in self.postprocessors:
            data = pp(self, data)
        return data

    @property
    def mesh(self):
        return self.__mesh

    @mesh.setter
    def mesh(self, value):
        # Rebind the field to the new mesh.
        # Doing this in a setter makes sure that if the user changes the mesh manually,
        # the field will still match the mesh.
        self.__mesh = value
        self.field = self.field @ value


class TransducerTrace(FieldTrace):
    name = 'Transducers'
    preprocessors = [lambda self, values=None: np.zeros(self.array.num_transducers)]

    radius_ratio = 3 / 2
    height = 5e-3
    num_vertices = 10
    colorscale = 'Greys'
    showscale = False
    cmin = 0
    cmax = 0

    class _field_class:
        def __init__(self, array):
            self.array = array

        def __call__(self, data):
            return np.repeat(data, self.num_coordinates // len(data))

        def __matmul__(self, other):
            coordinates, indices = other
            self.num_faces = indices.shape[1]
            self.num_coordinates = coordinates.shape[1]
            self.num_vertices_per_side = self.num_coordinates // 2
            return self

    def __init__(self, *args, visualize=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vizualise = visualize
        self._update_mesh()

    def __call__(self, complex_transducer_amplitudes=None):
        coordinates, indices = self.mesh

        return dict(
            type='mesh3d',
            x=coordinates[0] / self.display_scale, y=coordinates[1] / self.display_scale, z=coordinates[2] / self.display_scale,
            i=indices[0], j=indices[1], k=indices[2],
            intensity=super().__call__(complex_transducer_amplitudes),
            colorscale=self.colorscale, showscale=self.showscale,
            colorbar={'title': {'text': self.label, 'side': 'right'}, 'x': -0.02, 'xanchor': 'left'},
            cmin=self.cmin, cmax=self.cmax,
        )

    def _update_mesh(self):
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
            n_max = np.argmax(n)
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
        self.mesh = (coordinates, indices)


class TransducerPhase(TransducerTrace):
    preprocessors = [lambda self, data: np.angle(data) / np.pi]
    cmin = -1
    cmax = 1
    colorscale = 'Phase'
    name = 'Phase'
    label = 'Transducer phase in rad/π'
    showscale = True


class TransducerSignature(TransducerPhase):
    name = 'Signature at '
    label = 'Transducer signature in rad/π'

    preprocessors = [
        lambda self, data: np.angle(data),
        lambda self, data: self.array.signature(position=self.position, phases=data),
        lambda self, data: data / np.pi,
    ]

    def __init__(self, *args, position=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = position
        self.name = self.name + str(position)


class TransducerAmplitude(TransducerTrace):
    preprocessors = [lambda self, data: np.abs(data)]
    cmin = 0
    cmax = 1
    colorscale = 'Viridis'
    name = 'Amplitude'
    label = 'Transducer normalized amplitude'
    showscale = True


class ScalarFieldSlice(FieldTrace):
    cmin = None
    cmax = None

    def __init__(self, array, xlimits=None, ylimits=None, zlimits=None, normal=None, intersect=None, resolution=10, **kwargs):
        super().__init__(array, **kwargs)
        self._in_init = True

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

        self._in_init = False
        self._update_mesh()

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

    def _update_mesh(self):
        if self._in_init:
            return
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
        self.mesh = mesh[:, idx] + self.intersect.reshape((3, 1))

    def __call__(self, complex_transducer_amplitudes):
        return dict(
            type='mesh3d', intensity=super().__call__(complex_transducer_amplitudes),
            cmin=self.cmin, cmax=self.cmax, colorscale=self.colorscale,
            colorbar={'title': {'text': self.label, 'side': 'right'}, 'x': 1.02, 'xanchor': 'right'},
            x=np.squeeze(self.mesh[0]) / self.display_scale,
            y=np.squeeze(self.mesh[1]) / self.display_scale,
            z=np.squeeze(self.mesh[2]) / self.display_scale,
            delaunayaxis='xyz'[np.argmax(self.normal)],
        )


class PressureSlice(ScalarFieldSlice):
    name = 'Pressure'
    label = 'Sound pressure in dB re. 20 µPa'
    postprocessors = [lambda self, values: SPL(values)]
    from .fields import Pressure as _field_class
    cmin = 130
    cmax = 170


class VelocitySlice(ScalarFieldSlice):
    name = 'Velocity'
    label = 'Particle velocity in dB re. 50 nm/s'
    postprocessors = [lambda self, values: SVL(values)]
    from .fields import Velocity as _field_class
    cmin = 130
    cmax = 170


class VectorFieldCones(FieldTrace):
    opacity = 0.5
    cone_length = 0.8
    cone_vertices = 10
    cone_ratio = 3

    def __init__(self, array, center, resolution=5,
                 xrange=None, yrange=None, zrange=None, **kwargs):
        super().__init__(array, **kwargs)
        self._in_init = True
        self.center = center
        self.resolution = resolution
        self.xrange = xrange if xrange is not None else self.array.wavelength
        self.yrange = yrange if yrange is not None else self.array.wavelength
        self.zrange = zrange if zrange is not None else self.array.wavelength
        self._in_init = False
        self._update_mesh()

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

    def _update_mesh(self):
        if self._in_init:
            return
        nx = int(2 * np.ceil(self.xrange / self._resolution) + 1)
        ny = int(2 * np.ceil(self.yrange / self._resolution) + 1)
        nz = int(2 * np.ceil(self.zrange / self._resolution) + 1)

        x = np.linspace(-self.xrange, self.xrange, nx) + self.center[0]
        y = np.linspace(-self.yrange, self.yrange, ny) + self.center[1]
        z = np.linspace(-self.zrange, self.zrange, nz) + self.center[2]

        self.mesh = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0).reshape((3, -1))

    def __call__(self, complex_transducer_amplitudes):
        field_data = super().__call__(complex_transducer_amplitudes)
        vertex_indices, vertex_coordinates, vertex_intensities = self._generate_vertices(field_data)
        return dict(
            type='mesh3d',
            x=vertex_coordinates[0], y=vertex_coordinates[1], z=vertex_coordinates[2],
            i=vertex_indices[0], j=vertex_indices[1], k=vertex_indices[2],
            intensity=vertex_intensities, opacity=self.opacity, colorscale=self.colorscale,
            colorbar={'title': {'text': self.label, 'side': 'right'}, 'x': 1.02, 'xanchor': 'right'},
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


class RadiationForceCones(VectorFieldCones):
    from .fields import RadiationForce as _field_class
    postprocessors = []
    label = 'Force magnitude in N'
    name = 'Force'

    def __init__(self, *args, add_gravity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_gravity = add_gravity

    def _add_gravity(self, values):
        if self.add_gravity:
            values[2] -= self.field.field.mg
        return values

    postprocessors.append(_add_gravity)


class SphericalHarmonicsForceCones(RadiationForceCones):
    from .fields import SphericalHarmonicsForce as _field_class


class ForceDiagram(Visualizer):
    def __init__(self, *args, scale_to_gravity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_to_gravity = scale_to_gravity

    def __call__(self, *complex_transducer_amplitudes, **kwargs):
        colors = plotly.colors.qualitative.Plotly

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
                    name=kwargs['name'][idx] if 'name' in kwargs else 'Data {}'.format(idx),
                )
                all_traces.extend(this_state_traces)
        elif len(self) > 1:
            data = complex_transducer_amplitudes[0]
            for idx, field in enumerate(self):
                this_field_traces = field(
                    data, line=kwargs['line'][idx] if 'line' in kwargs else dict(color=colors[idx % len(colors)]),
                    name=kwargs['name'][idx] if 'name' in kwargs else field.name if field.name is not '' else 'Force {}'.format(idx),
                )
                all_traces.extend(this_field_traces)
        else:
            data = complex_transducer_amplitudes[0]
            field = self[0]
            all_traces = field(data, line=kwargs['line'] if 'line' in kwargs else dict(color=colors[0]))
            all_traces[0]['showlegend'] = False

        return go.Figure(data=all_traces, layout=self.layout)

    @property
    def layout(self):
        layout_gap_ratio = 12
        total_pieces = layout_gap_ratio * 3 + 2
        width = layout_gap_ratio / total_pieces
        gap = 1 / total_pieces
        length_unit = r'\text{{ in {}}}'.format(self.display_scale)
        force_unit = '/mg' if self.scale_to_gravity else r'\text{ in N}'
        return dict(
            super().layout,
            xaxis=dict(title='$x' + length_unit + '$', domain=[0, width], anchor='y3'),
            xaxis2=dict(title='$y' + length_unit + '$', domain=[width + gap, 2 * width + gap], anchor='y3'),
            xaxis3=dict(title='$z' + length_unit + '$', domain=[2 * width + 2 * gap, 1], anchor='y3'),
            yaxis=dict(title='$F_x' + force_unit + '$', domain=[2 * width + 2 * gap, 1]),
            yaxis2=dict(title='$F_y' + force_unit + '$', domain=[width + gap, 2 * width + gap]),
            yaxis3=dict(title='$F_z' + force_unit + '$', domain=[0, width]),
        )

    class ForceTrace(FieldTrace):
        _trace_objects = 0

        def __init__(self, array, center, *args,
                     resolution=25, xrange=None, yrange=None, zrange=None,
                     force_calculator=None, name=None, **kwargs):
            if force_calculator is None:
                if 'radius' in kwargs:
                    from .fields import SphericalHarmonicsForce as force_calculator
                else:
                    from .fields import RadiationForce as force_calculator
            super().__init__(array, *args, field=force_calculator, **kwargs)
            self._trace_id = self._trace_objects
            self._trace_objects += 1
            self._trace_calls = 0
            if name is not None:
                self.name = name

            self._in_init = True
            self.center = center
            self.resolution = resolution
            self.xrange = xrange if xrange is not None else self.array.wavelength
            self.yrange = yrange if yrange is not None else self.array.wavelength
            self.zrange = zrange if zrange is not None else self.array.wavelength
            self._in_init = False
            self._update_mesh()

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

        def _update_mesh(self):
            if self._in_init:
                return
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

            self.mesh = np.concatenate([X, Y, Z], axis=1)

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
