"""Some tools for analysis of sound fields and levitation traps."""

import numpy as np


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
    from scipy.integrate import solve_ivp
    from numpy.linalg import lstsq
    if 'radius' in kwargs:
        from .fields import SphericalHarmonicsForce as Force, SphericalHarmonicsForceGradient as ForceGradient
    else:
        from .fields import RadiationForce as Force, RadiationForceGradient as ForceGradient
    from .fields import stack
    evaluator = stack(Force(array, **kwargs), ForceGradient(array, **kwargs))
    mg = evaluator.fields[0].field.mg

    def f(t, x):
        F = evaluator(complex_transducer_amplitudes, x)[0]
        F[2] -= mg
        return F

    def bead_close(t, x):
        F, dF = evaluator(complex_transducer_amplitudes, x)
        F[2] -= mg
        dx = lstsq(dF, F, rcond=None)[0]
        distance = np.sum(dx**2, axis=0)**0.5
        return np.clip(distance - tolerance, 0, None)
    bead_close.terminal = True
    outs = solve_ivp(f, (0, time_interval), np.asarray(start_position), events=bead_close, vectorized=True, dense_output=path_points > 1)
    if outs.message != 'A termination event occurred.':
        print('End criterion not met. Final path position might not be close to trap location.')
    if path_points > 1:
        return outs.sol(np.linspace(0, outs.sol.t_max, path_points))
    else:
        return outs.y[:, -1]
