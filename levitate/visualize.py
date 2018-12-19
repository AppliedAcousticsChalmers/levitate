"""Visualization methods based on the plotly graphing library, and some connivance functions."""
import numpy as np
# import plotly.graph_objs as go
from .materials import Air


def dB(x, power=False):
    r"""Convert ratio to decibels.

    Converting a ratio to decibels depends on whether the ratio is a ratio
    of amplitudes or a ratio of powers. For amplitudes the decibel value is
    :math:`20\log(|x|)`, while for power ratios the value is :math:`10\log(|x|)`
    where :math:`\log` is the base 10 logarithm.

    Parameters
    ----------
    x : numeric
        Linear amplitude or radio, can be complex.
    power : bool, default False
        Toggles if the ration is proportional to power.

    Returns
    -------
    Lx : numeric
        The decibel value.
    """
    if power:
        return 10 * np.log10(np.abs(x))
    else:
        return 20 * np.log10(np.abs(x))


def SPL(p):
    """Convert sound pressure to sound pressure level.

    Uses the standard reference value for airborne acoustics: 20µPa.
    Note that the input is the complex rms amplitude.

    Parameters
    ----------
    p : numeric, complex
        The complex sound pressure rms amplitude.

    Returns
    -------
    SPL : numeric
        The sound pressure level
    """
    return dB(p / 20e-6)


def SVL(u):
    """Convert sound particle velocity to sound velocity level.

    Uses the standard reference value for airborne acoustics: 20µPa
    and the material properties of air from the materials module.
    Note that the input is the complex rms amplitude.

    If the first axis of the velocity input has length 3, it will be assumed to
    be the three Cartesian components of the velocity.

    Parameters
    ----------
    u : numeric, complex
        The complex sound velocity rms amplitude, or the vector velocity.

    Returns
    -------
    SVL : numeric
        The sound velocity level
    """
    u = np.asarray(u)
    try:
        if u.shape[0] == 3:
            u = np.sum(u**2, 0)**0.5
    except IndexError:
        pass
    return SPL(u * Air.c * Air.rho)


def find_trap(array, start_pos, tolerance=10e-6, time_interval=50, return_path=False, rho=25, radius=1e-3):
    r"""Find the approximate location of a levitation trap.

    Find an approximate position of a acoustic levitation trap close to a starting point.
    This is done by following the radiation force in the sound field using an differential
    equation solver. The differential equation is the unphysical equation :math:`d\vec x/dt  = \vec F(x,t)`,
    i.e. interpreting the force field as a velocity field.
    This works for finding the location of a trap and the field line from the starting position
    to the trap position, but it can not be seen as a proper kinematic simulation of the system.

    The solving of the above equation takes place until the whole time interval is covered,
    or the tolerance is met. The tolerance is evaluated using the assumption that the force
    is zero at the trap, evaluating the distance from the zero-force position using the force
    gradient.

    Parameters
    ----------
    array : `TransducerArray`
        The array creating the sound field.
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
    evaluator = array.PersistentFieldEvaluator(array)

    def f(t, x):
        F = evaluator.force(x.T)
        F[2] -= mg
        return F

    def bead_close(t, x):
        dF = evaluator.stiffness(x.T)
        F = evaluator.force(x.T)
        F[2] -= mg
        distance = np.sum((F / dF)**2, axis=0)**0.5
        return np.clip(distance - tolerance, 0, None)
    bead_close.terminal = True
    outs = solve_ivp(f, (0, time_interval), np.asarray(start_pos), events=bead_close, vectorized=True, dense_output=return_path)
    if return_path:
        if return_path is True:
            return_path = 200
        return outs.sol(np.linspace(0, outs.sol.t_max, return_path))
    else:
        return outs.y[:, -1]


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
        xlimits = xlimits or (np.min(array.transducer_positions[:, 0]), np.max(array.transducer_positions[:, 0]))
        ylimits = ylimits or (np.min(array.transducer_positions[:, 1]), np.max(array.transducer_positions[:, 1]))
        if 'Doublesided' not in type(array).__name__:
            # Singlesided array, one of the limits will give a zero range.
            # Assuming that the array is in the xy-plane, pointing up
            zlimits = zlimits or (np.min(array.transducer_positions[:, 2]) + 1e-3, 20 * array.wavelength)
        else:
            zlimits = zlimits or (np.min(array.transducer_positions[:, 2]), np.max(array.transducer_positions[:, 2]))

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
        x = np.mgrid[self.xlimits[0]:self.xlimits[1]:self._resolution]
        y = np.mgrid[self.ylimits[0]:self.ylimits[1]:self._resolution]
        z = np.mgrid[self.zlimits[0]:self.zlimits[1]:self._resolution]
        if axis is 'x':
            self._x, self._y, self._z = np.mgrid[value:value:1j, self.ylimits[0]:self.ylimits[1]:self._resolution, self.zlimits[0]:self.zlimits[1]:self._resolution]
        if axis is 'y':
            self._x, self._y, self._z = np.mgrid[self.xlimits[0]:self.xlimits[1]:self._resolution, value:value:1j, self.zlimits[0]:self.zlimits[1]:self._resolution]
        if axis is 'z':
            self._x, self._y, self._z = np.mgrid[self.xlimits[0]:self.xlimits[1]:self._resolution, self.ylimits[0]:self.ylimits[1]:self._resolution, value:value:1j]

    @property
    def _mesh(self):
        return np.stack([self._x, self._y, self._z], axis=-1)

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
        try:
            data = self._mesh
            for f in calculator:
                data = f(data)
        except TypeError:
            data = calculator(self._mesh)
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
                                 cmin=min, cmax=max, colorscale='Viridis')

    def velocity(self, min=130, max=170):
        """Visualize velocity field."""
        return self.scalar_field(calculator=[self.calculate.velocity, SVL],
                                 cmin=min, cmax=max, colorscale='Viridis')

    def transducers(self, data='phases'):
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
        if type(data) == str:
            if 'phase' in data:
                title = 'Transducer phase in rad/π'
                data = self.array.phases / np.pi
                cmin = -1
                cmax = 1
                colorscale = [[0.0, 'hsv(0,255,255)'], [0.25, 'hsv(90,255,255)'], [0.5, 'hsv(180,255,255)'], [0.75, 'hsv(270,255,255)'], [1.0, 'hsv(360,255,255)']]
            elif 'amp' in data:
                title = 'Transducer amplitude'
                data = self.array.amplitudes
                cmin = 0
                cmax = 1
                colorscale = 'Viridis'
        else:
            title = 'Transducer data'
            cmin = np.min(data)
            cmax = np.max(data)
            colorscale = 'Viridis'

        marker = dict(color=data, colorscale=colorscale, size=16, colorbar={'title': title, 'x': -0.02}, cmin=cmin, cmax=cmax)
        return dict(
            type='scatter3d', mode='markers',
            x=self.array.transducer_positions[:, 0],
            y=self.array.transducer_positions[:, 1],
            z=self.array.transducer_positions[:, 2],
            marker=marker
        )