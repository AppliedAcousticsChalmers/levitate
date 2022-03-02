"""Some tools for analysis of sound fields and levitation traps."""

import numpy as np
import scipy.integrate
from . import fields, materials


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
    L : numeric
        The decibel value.

    """
    if power:
        return 10 * np.log10(np.abs(x))
    else:
        return 20 * np.log10(np.abs(x))


def SPL(p):
    """Convert sound pressure to sound pressure level.

    Uses the standard reference value for airborne acoustics: 20 µPa.
    Note that the input is the pressure amplitude, not the RMS value.

    Parameters
    ----------
    p : numeric, complex
        The complex sound pressure amplitude.

    Returns
    -------
    SPL : numeric
        The sound pressure level

    """
    return dB(p / (20e-6 * 2**0.5))


def SVL(u):
    """Convert sound particle velocity to sound velocity level.

    Uses the standard reference value for airborne acoustics: 50 nm/s,
    which is approximately 20 µPa / c_0 / rho_0
    Note that the input the velocity amplitude(s), not the RMS values.

    If the first axis of the velocity input has length 3, it will be assumed to
    be the three Cartesian components of the velocity.

    Parameters
    ----------
    u : numeric, complex
        The complex sound velocity amplitude, or the vector velocity.

    Returns
    -------
    SVL : numeric
        The sound velocity level

    """
    u = np.asarray(u)
    try:
        if u.shape[0] == 3:
            u = np.sum(np.abs(u)**2, 0)**0.5
    except IndexError:
        pass
    return dB(u / (50e-9 * 2**0.5))


def find_trap(array, start_position, complex_transducer_amplitudes, tolerance=10e-6, time_interval=50, path_points=1, **kwargs):
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
    array : TrasducerArray
        The transducer array to use for the solving.
    start_position : array_like, 3 elements
        The starting point for the solving.
    complex_transducer_amplitudes: complex array like
        The complex transducer amplitudes to use for the solving.
    tolerance : numeric, default 10e-6
        The approximate tolerance of the solution, i.e. how close should
        the found position be to the true position, in meters.
    time_interval : numeric, default 50
        The unphysical time of the solution range in the differential equation above.
    path_points : int, default 1
        Sets the number of points to return the path at.
        A single evaluation point will only return the found position of the trap.

    Returns
    -------
    trap_pos : numpy.ndarray
        The found trap position, or the path from the starting position to the trap position.

    """
    if 'radius' in kwargs:
        Force = fields.SphericalHarmonicsForce
        ForceGradient = fields.SphericalHarmonicsForceGradient
    else:
        Force = fields.RadiationForce
        ForceGradient = fields.RadiationForceGradient
    evaluator = fields.stack(Force(array, **kwargs), ForceGradient(array, **kwargs))
    mg = evaluator.fields[0].field.mg

    def f(t, x):
        F = evaluator(complex_transducer_amplitudes, x)[0]
        F[2] -= mg
        return F

    def bead_close(t, x):
        F, dF = evaluator(complex_transducer_amplitudes, x)
        F[2] -= mg
        dx = np.linalg.lstsq(dF, F, rcond=None)[0]
        distance = np.sum(dx**2, axis=0)**0.5
        return np.clip(distance - tolerance, 0, None)
    bead_close.terminal = True
    outs = scipy.integrate.solve_ivp(f, (0, time_interval), np.asarray(start_position), events=bead_close, vectorized=True, dense_output=path_points > 1)
    if outs.message != 'A termination event occurred.':
        print('End criterion not met. Final path position might not be close to trap location.')
    if path_points > 1:
        return outs.sol(np.linspace(0, outs.sol.t_max, path_points))
    else:
        return outs.y[:, -1]


class KineticSimulation:
    """Performs kinetic simulations for levitated spherical objects.

    Initialize with the relevant parameters. Call the object with a state
    and a position to start the simulation.
    After a simulation, the object stores attributes for the results.

    If the simulation is not started at the center of something resembling a trap,
    the energy tracking will not work properly.

    Attributes
    ----------
    t : ndarray, shape (T,)
        Time vector for the simulated positions.
    position : ndarray, shape (3, T)
        Simulated positions.
    velocity : ndarray, shape (3, T)
        Simulated velocities.
    kinetic_energy : ndarray, shape (T,)
        Simulated kinetic energy.
    potential_energy : ndarray, shape (T,)
        Approximate potential energy. Calculated from a linear approximation at the starting position.
    total_energy : ndarray, shape (T,)
        Sum of kinetic and potential energy.
    """
    def __init__(self, array, t_end=1,
                 radius=1e-3, material=materials.styrofoam,
                 force=None, force_gradient=None,
                 **solver_kwargs
                 ):
        self.array = array
        self.t_end = t_end

        self.radius = radius
        self.mass = 4 / 3 * np.pi * radius**3 * material.rho
        self.weight = self.mass * 9.82

        if force is None or force_gradient is None:
            if (force, force_gradient) != (None, None):
                raise TypeError('Cannot supply only one of `force` and `force_gradient`')
            if radius * array.k < 0.1:
                force = fields.RadiationForce(array, radius=radius, material=material)
                force_gradient = fields.RadiationForceGradient(array, radius=radius, material=material)
            else:
                force = fields.SphericalHarmonicsForce(array, radius=radius, material=material)
                force_gradient = fields.SphericalHarmonicsForceGradient(array, radius=radius, material=material)
        self._radiation_force = force
        self._radiation_force_and_gradient = fields.stack([force, force_gradient])

        self.solver_kwargs = solver_kwargs

    def _differential(self, t, x):
        F = self._radiation_force(self._state, x[:3]) + self._drag_force(x[3:6])
        F[2] -= self.weight
        acceleration = F / self.mass

        return np.concatenate([x[3:6], acceleration], axis=0)

    def _drag_force(self, velocity):
        velocity_magnitude = np.sum(velocity**2, axis=0)**0.5
        if np.allclose(velocity_magnitude, 0):
            return np.zeros_like(velocity)

        R = 2 * self.radius * velocity_magnitude / self.array.medium.kinematic_viscosity
        Cd = 24 / R * (1 + 0.15 * R ** 0.618) + 0.407 / (1 + 8701 / R)
        F_drag = -np.pi / 2 * self.radius**2 * self.array.medium.rho * Cd * velocity * velocity_magnitude
        return F_drag

    def _kinetic_energy(self, velocity):
        return np.sum(velocity**2, axis=0) * self.mass / 2

    def _potential_energy(self, position):
        offset = position - self._initial_position.reshape((3,) + (-1,) * (np.ndim(position) - 1))
        return np.einsum('i..., ij, j... -> ...', offset, -self._initial_force_gradient, offset).squeeze() / 2

    def _diverged(self, t, x):
        distance_from_start = np.sum((x[:3] - self._initial_position)**2)**0.5
        if distance_from_start > 100e-3:
            return 0
        return 1

    _diverged.terminal = True

    def _converged(self, t, x):
        energy = self._kinetic_energy(x[3:6]) + self._potential_energy(x[:3])
        return np.clip(energy - self._initial_energy / 10, 0, None)

    def __call__(self, state, initial_position):
        self._state = state
        self._initial_position = np.asarray(initial_position)

        F, dF = self._radiation_force_and_gradient(state, self._initial_position)
        eigenvals, eigenvecs = np.linalg.eig(dF)
        resonance_omega = (np.abs(eigenvals) / self.mass)**0.5
        shortest_period = np.min(2 * np.pi / resonance_omega)
        initial_speed = np.min(resonance_omega) / self.array.k
        initial_direction = np.linalg.solve(dF, [1, 1, 1])
        initial_direction = np.abs(np.diag(dF))**0.5
        self._initial_velocity = initial_speed * initial_direction / np.sum(initial_direction**2)**0.5

        self._initial_force_gradient = dF
        self._initial_energy = self._kinetic_energy(self._initial_velocity)

        x0 = np.concatenate([self._initial_position, self._initial_velocity], axis=0)
        output = scipy.integrate.solve_ivp(
            self._differential, (0, self.t_end), x0,
            events=[self._diverged, self._converged],
            dense_output=True, vectorized=True, max_step=shortest_period / 6,
            t_eval=np.linspace(0, self.t_end, np.math.ceil(self.t_end / shortest_period * 24)),
            **self.solver_kwargs
        )
        self.solver_output = output

        self.t = output.t
        self.position = output.y[:3]
        self.velocity = output.y[3:]
        self.kinetic_energy = self._kinetic_energy(self.velocity)
        self.potential_energy = self._potential_energy(self.position)
        self.total_energy = self.kinetic_energy + self.potential_energy

        return self

