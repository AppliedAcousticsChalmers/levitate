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
    and the material properties of air in the materials module.
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
    equation solver. The differential equation is question is the unphysical equation
    :math:`d\vec x/dt  = \vec F(x,t)`, i.e. interpreting the force field as a velocity field.
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
