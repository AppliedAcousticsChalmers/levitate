"""Visualization methods based on the plotly graphing library, and some convenience functions."""
import collections.abc
import numpy as np
from .utils import SPL, SVL
try:
    import plotly
except ImportError:
    pass


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
        value.visualizer = self
        self._traces.insert(index, value)

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

    def __call__(self, complex_transducer_amplitudes):
        pass
        # Get the traces from the included fields, make sure they have access to the "visualizer" property
        # Set the layout so that you can toggle the traces
        # Create transducer visualizations, make sure to have raw phases, amplitudes, real, imaginary, and signature as options for the transducers


class Trace:
    colorscale = 'Viridis'

    def __init__(self, array, field=None, **field_kwargs):
        self.array = array
        if field is not None:
            if type(field) is type:
                self.field = field(array, **field_kwargs)
            else:
                self.field = field
        elif hasattr(self, '_field_class'):
            self.field = self._field_class(array, **field_kwargs)

    @property
    def display_scale(self):
        try:
            return self.visualizer._display_scale
        except AttributeError:
            return 1

    class meshproperty:
        def __init__(self, fpre=None, fpost=None):
            self.fpre = fpre or (lambda self, value: value)
            self.fpost = fpost or (lambda self, value: value)
            self.value = None
            self.__doc__ = fpre.__doc__

        def __get__(self, obj, obj_type=None):
            return self.fpost(obj, self.value)

        def __set__(self, obj, value):
            self.value = self.fpre(obj, value)
            obj._update_mesh()

        def __delete__(self, obj):
            self.value = None

        def preprocessor(self, fpre):
            return type(self)(fpre, self.fpost)

        def postprocessor(self, fpost):
            return type(self)(self.fpre, fpost)

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

    def _update_mesh(self):
        raise NotImplementedError('Subclasses of `Trace` has to implement `_update_mesh`')


class ScalarFieldSlice(Trace):
    label = ''
    cmin = None
    cmax = None

    preprocesssors = []
    postprocessors = []

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

    @Trace.meshproperty
    def normal(self, value):
        value = np.asarray(value, dtype=float)
        return value / np.sum(value**2)**0.5

    @Trace.meshproperty
    def intersect(self, value):
        return np.asarray(value, dtype=float)

    @Trace.meshproperty
    def resolution(self, val):
        self._resolution = self.array.wavelength / val
        return val

    xlimits = Trace.meshproperty()
    ylimits = Trace.meshproperty()
    zlimits = Trace.meshproperty()

    def _update_mesh(self):
        if self._in_init:
            return
        # We need to create a mesh which is parallel to e.g. xy to start with, then map that to the desired tilted plane
        # A proper rotate and shift should preserve the distances between points, so that's not a problem
        # It will be somewhat difficult to figure out the limits of the original plane depending on the limits we want for out final mesh
        # Perhaps we should find two vectors in the plane and add integer numbers of the vectors together until we reach the end of out mesh domain.
        # It might be faster to mesh too much first and then remove the points outside out target domain?
        nx, ny, nz = self.normal
        if nz == 0:
            v1 = np.array([0, 0, 1], dtype=float)
        else:
            v1 = np.array([1, 1, -(nx + ny) / nz], dtype=float)
            v1 /= np.sum(v1**2)**0.5
        v2 = np.cross(self.normal, v1)
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
        edge_intersections = []
        if nx != 0:
            edge_intersections.extend([np.array([-(ny * yl + nz * zl) / ny, yl, zl]) for yl in (ymin, ymax) for zl in (zmin, zmax)])
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
        for pp in self.preprocesssors:
            complex_transducer_amplitudes = pp(complex_transducer_amplitudes)
        field_data = self.field(complex_transducer_amplitudes)
        for pp in self.postprocessors:
            field_data = pp(field_data)

        return dict(
            type='mesh3d', intensity=np.squeeze(field_data),
            cmin=self.cmin, cmax=self.cmax, colorscale=self.colorscale,
            colorbar={'title': self.label},
            x=np.squeeze(self.mesh[0]) / self.display_scale,
            y=np.squeeze(self.mesh[1]) / self.display_scale,
            z=np.squeeze(self.mesh[2]) / self.display_scale,
            delaunayaxis='xyz'[np.argmax(self.normal)],
        )


class PressureSlice(ScalarFieldSlice):
    name = 'Pressure'
    label = 'Sound pressure in dB re. 20 µPa'
    postprocessors = [SPL]
    from .fields import Pressure as _field_class
    cmin = 130
    cmax = 170


class VelocitySlice(ScalarFieldSlice):
    name = 'Velocity'
    label = 'Particle velocity in dB re. 50 nm/s'
    postprocessors = [SVL]
    from .fields import Velocity as _field_class
    cmin = 130
    cmax = 170
