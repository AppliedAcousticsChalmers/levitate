"""Handling of individual transducers and their directivities.

This module contains classes describing how individual transducer elements radiate sound,
e.g. waveforms and directivities.
This is also where the various spatial properties, e.g. derivatives, are implemented.
Most calculations in this module are fully vectorized, so the models can calculate
sound fields for any number of source positions and receiver positions at once.

.. autosummary::
    :nosignatures:

    TransducerModel
    PointSource
    PlaneWaveTransducer
    CircularPiston
    CircularRing
    TransducerReflector
"""

import numpy as np
import logging
from scipy.special import j0, j1, jv, jnp_zeros, jvp
from scipy.special import spherical_jn, spherical_yn, sph_harm
from .materials import air
from . import utils

logger = logging.getLogger(__name__)


class TransducerModel:
    """Base class for ultrasonic single frequency transducers.

    Parameters
    ----------
    freq : float, default 40 kHz
        The resonant frequency of the transducer.
    p0 : float, default 6 Pa
        The sound pressure created at maximum amplitude at 1m distance, in Pa.
        Note: This is not an rms value!
    medium : Material
        The medium in which the array is operating.
    physical_size : float, default 10e-3
        The physical dimentions of the transducer. Mainly used for visualization
        and some geometrical assumptions.

    Attributes
    ----------
    k : float
        Wavenumber in the medium.
    wavelength : float
        Wavelength in the medium.
    omega : float
        Angular frequency.
    freq : float
        Wave frequency.

    """

    _repr_fmt_spec = '{:%cls(freq=%freq, p0=%p0, medium=%mediumfull, physical_size=%physical_size)}'
    _str_fmt_spec = '{:%cls(freq=%freq, p0=%p0, medium=%medium)}'

    def __init__(self, freq=40e3, p0=6, medium=air, physical_size=10e-3):
        self.medium = medium
        self.freq = freq
        self.p0 = p0
        self.physical_size = physical_size
        # The murata transducers are measured to 85 dB SPL at 1 V at 1 m, which corresponds to ~6 Pa at 20 V
        # The datasheet specifies 120 dB SPL @ 0.3 m, which corresponds to ~6 Pa @ 1 m

    def __format__(self, fmt_spec):
        return fmt_spec.replace('%cls', self.__class__.__name__).replace('%freq', str(self.freq)).replace('%p0', str(self.p0)).replace('%mediumfull', repr(self.medium)).replace('%medium', str(self.medium)).replace('%physical_size', str(self.physical_size))

    def __str__(self):
        return self._str_fmt_spec.format(self)

    def __repr__(self):
        return self._repr_fmt_spec.format(self)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and np.allclose(self.p0, other.p0)
            and np.allclose(self.omega, other.omega)
            and np.allclose(self.k, other.k)
            and self.medium == other.medium
            and self.physical_size == other.physical_size
        )

    @property
    def k(self):
        return self.omega / self.medium.c

    @k.setter
    def k(self, value):
        self._omega = value * self.medium.c

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    @property
    def freq(self):
        return self.omega / 2 / np.pi

    @freq.setter
    def freq(self, value):
        self.omega = value * 2 * np.pi

    @property
    def wavelength(self):
        return 2 * np.pi / self.k

    @wavelength.setter
    def wavelength(self, value):
        self.k = 2 * np.pi / value

    def pressure(self, source_positions, source_normals, receiver_positions, **kwargs):
        """Calculate the complex sound pressure from the transducer.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.

        Returns
        -------
        out : numpy.ndarray
            The pressure at the locations, shape `source_positions.shape[1:] + receiver_positions.shape[1:]`.

        """
        return self.pressure_derivs(source_positions=source_positions, source_normals=source_normals, receiver_positions=receiver_positions, orders=0, **kwargs)[0]

    def pressure_derivs(self, source_positions, source_normals, receiver_positions, orders=3, **kwargs):
        """Calculate the spatial derivatives of the greens function.

        Calculates Cartesian spatial derivatives of the pressure Green's function. Should be implemented by concrete subclasses.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        raise NotImplementedError('Transducer model of type `{}` has not implemented cartesian pressure derivatives'.format(self.__class__.__name__))


class PointSource(TransducerModel):
    r"""Point source transducers.

    A point source is in this context defines as a spherically spreading wave,
    optionally with a directivity. On its own this class defines a monopole,
    but subclasses are free to change the directivity to other shapes.

    The spherical spreading is defined as

    .. math:: G(r) = {e^{ikr} \over r}

    where :math:`r` is the distance from the source, and :math:`k` is the wavenumber of the wave.
    """

    def directivity(self, source_positions, source_normals, receiver_positions):
        """Evaluate transducer directivity.

        Subclasses will preferably implement this to create new directivity models.
        Default implementation is omnidirectional sources.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.

        Returns
        -------
        out : numpy.ndarray
            The amplitude (and phase) of the directivity, shape `source_positions.shape[1:] + receiver_positions.shape[1:]`.

        """
        return np.ones(np.asarray(source_positions).shape[1:2] + np.asarray(receiver_positions).shape[1:])

    def pressure_derivs(self, source_positions, source_normals, receiver_positions, orders=3, **kwargs):
        """Calculate the spatial derivatives of the greens function.

        This is the combination of the derivative of the spherical spreading, and
        the derivatives of the directivity, including source strength.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        receiver_positions = np.asarray(receiver_positions)
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')
        wavefront_derivatives = self.wavefront_derivatives(source_positions, receiver_positions, orders)
        if type(self) == PointSource:
            return wavefront_derivatives * self.p0
        directivity_derivatives = self.directivity_derivatives(source_positions, source_normals, receiver_positions, orders)

        derivatives = np.empty(wavefront_derivatives.shape, dtype=np.complex128)
        derivatives[0] = wavefront_derivatives[0] * directivity_derivatives[0]

        if orders > 0:
            derivatives[1] = wavefront_derivatives[0] * directivity_derivatives[1] + directivity_derivatives[0] * wavefront_derivatives[1]
            derivatives[2] = wavefront_derivatives[0] * directivity_derivatives[2] + directivity_derivatives[0] * wavefront_derivatives[2]
            derivatives[3] = wavefront_derivatives[0] * directivity_derivatives[3] + directivity_derivatives[0] * wavefront_derivatives[3]

        if orders > 1:
            derivatives[4] = wavefront_derivatives[0] * directivity_derivatives[4] + directivity_derivatives[0] * wavefront_derivatives[4] + 2 * directivity_derivatives[1] * wavefront_derivatives[1]
            derivatives[5] = wavefront_derivatives[0] * directivity_derivatives[5] + directivity_derivatives[0] * wavefront_derivatives[5] + 2 * directivity_derivatives[2] * wavefront_derivatives[2]
            derivatives[6] = wavefront_derivatives[0] * directivity_derivatives[6] + directivity_derivatives[0] * wavefront_derivatives[6] + 2 * directivity_derivatives[3] * wavefront_derivatives[3]
            derivatives[7] = wavefront_derivatives[0] * directivity_derivatives[7] + directivity_derivatives[0] * wavefront_derivatives[7] + wavefront_derivatives[1] * directivity_derivatives[2] + directivity_derivatives[1] * wavefront_derivatives[2]
            derivatives[8] = wavefront_derivatives[0] * directivity_derivatives[8] + directivity_derivatives[0] * wavefront_derivatives[8] + wavefront_derivatives[1] * directivity_derivatives[3] + directivity_derivatives[1] * wavefront_derivatives[3]
            derivatives[9] = wavefront_derivatives[0] * directivity_derivatives[9] + directivity_derivatives[0] * wavefront_derivatives[9] + wavefront_derivatives[2] * directivity_derivatives[3] + directivity_derivatives[2] * wavefront_derivatives[3]

        if orders > 2:
            derivatives[10] = wavefront_derivatives[0] * directivity_derivatives[10] + directivity_derivatives[0] * wavefront_derivatives[10] + 3 * (directivity_derivatives[4] * wavefront_derivatives[1] + wavefront_derivatives[4] * directivity_derivatives[1])
            derivatives[11] = wavefront_derivatives[0] * directivity_derivatives[11] + directivity_derivatives[0] * wavefront_derivatives[11] + 3 * (directivity_derivatives[5] * wavefront_derivatives[2] + wavefront_derivatives[5] * directivity_derivatives[2])
            derivatives[12] = wavefront_derivatives[0] * directivity_derivatives[12] + directivity_derivatives[0] * wavefront_derivatives[12] + 3 * (directivity_derivatives[6] * wavefront_derivatives[3] + wavefront_derivatives[6] * directivity_derivatives[3])
            derivatives[13] = wavefront_derivatives[0] * directivity_derivatives[13] + directivity_derivatives[0] * wavefront_derivatives[13] + wavefront_derivatives[2] * directivity_derivatives[4] + directivity_derivatives[2] * wavefront_derivatives[4] + 2 * (wavefront_derivatives[1] * directivity_derivatives[7] + directivity_derivatives[1] * wavefront_derivatives[7])
            derivatives[14] = wavefront_derivatives[0] * directivity_derivatives[14] + directivity_derivatives[0] * wavefront_derivatives[14] + wavefront_derivatives[3] * directivity_derivatives[4] + directivity_derivatives[3] * wavefront_derivatives[4] + 2 * (wavefront_derivatives[1] * directivity_derivatives[8] + directivity_derivatives[1] * wavefront_derivatives[8])
            derivatives[15] = wavefront_derivatives[0] * directivity_derivatives[15] + directivity_derivatives[0] * wavefront_derivatives[15] + wavefront_derivatives[1] * directivity_derivatives[5] + directivity_derivatives[1] * wavefront_derivatives[5] + 2 * (wavefront_derivatives[2] * directivity_derivatives[7] + directivity_derivatives[2] * wavefront_derivatives[7])
            derivatives[16] = wavefront_derivatives[0] * directivity_derivatives[16] + directivity_derivatives[0] * wavefront_derivatives[16] + wavefront_derivatives[3] * directivity_derivatives[5] + directivity_derivatives[3] * wavefront_derivatives[5] + 2 * (wavefront_derivatives[2] * directivity_derivatives[9] + directivity_derivatives[2] * wavefront_derivatives[9])
            derivatives[17] = wavefront_derivatives[0] * directivity_derivatives[17] + directivity_derivatives[0] * wavefront_derivatives[17] + wavefront_derivatives[1] * directivity_derivatives[6] + directivity_derivatives[1] * wavefront_derivatives[6] + 2 * (wavefront_derivatives[3] * directivity_derivatives[8] + directivity_derivatives[3] * wavefront_derivatives[8])
            derivatives[18] = wavefront_derivatives[0] * directivity_derivatives[18] + directivity_derivatives[0] * wavefront_derivatives[18] + wavefront_derivatives[2] * directivity_derivatives[6] + directivity_derivatives[2] * wavefront_derivatives[6] + 2 * (wavefront_derivatives[3] * directivity_derivatives[9] + directivity_derivatives[3] * wavefront_derivatives[9])
            derivatives[19] = wavefront_derivatives[0] * directivity_derivatives[19] + wavefront_derivatives[19] * directivity_derivatives[0] + wavefront_derivatives[1] * directivity_derivatives[9] + wavefront_derivatives[2] * directivity_derivatives[8] + wavefront_derivatives[3] * directivity_derivatives[7] + directivity_derivatives[1] * wavefront_derivatives[9] + directivity_derivatives[2] * wavefront_derivatives[8] + directivity_derivatives[3] * wavefront_derivatives[7]

        derivatives *= self.p0
        return derivatives

    def wavefront_derivatives(self, source_positions, receiver_positions, orders=3):
        """Calculate the spatial derivatives of the spherical spreading.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : ndarray
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        source_positions = np.asarray(source_positions)
        receiver_positions = np.asarray(receiver_positions)
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        r = np.sum(diff**2, axis=0)**0.5
        kr = self.k * r
        jkr = 1j * kr
        phase = np.exp(jkr)

        derivatives = np.empty((utils.num_pressure_derivs[orders],) + r.shape, dtype=np.complex128)
        derivatives[0] = phase / r

        if orders > 0:
            coeff = (jkr - 1) * phase / r**3
            derivatives[1] = diff[0] * coeff
            derivatives[2] = diff[1] * coeff
            derivatives[3] = diff[2] * coeff

        if orders > 1:
            coeff = (3 - kr**2 - 3 * jkr) * phase / r**5
            const = (jkr - 1) * phase / r**3
            derivatives[4] = diff[0]**2 * coeff + const
            derivatives[5] = diff[1]**2 * coeff + const
            derivatives[6] = diff[2]**2 * coeff + const
            derivatives[7] = diff[0] * diff[1] * coeff
            derivatives[8] = diff[0] * diff[2] * coeff
            derivatives[9] = diff[1] * diff[2] * coeff

        if orders > 2:
            const = (3 - 3 * jkr - kr**2) * phase / r**5
            coeff = (-15 + 15 * jkr + 6 * kr**2 - 1j * kr**3) * phase / r**7
            derivatives[10] = diff[0] * (3 * const + diff[0]**2 * coeff)
            derivatives[11] = diff[1] * (3 * const + diff[1]**2 * coeff)
            derivatives[12] = diff[2] * (3 * const + diff[2]**2 * coeff)
            derivatives[13] = diff[1] * (const + diff[0]**2 * coeff)
            derivatives[14] = diff[2] * (const + diff[0]**2 * coeff)
            derivatives[15] = diff[0] * (const + diff[1]**2 * coeff)
            derivatives[16] = diff[2] * (const + diff[1]**2 * coeff)
            derivatives[17] = diff[0] * (const + diff[2]**2 * coeff)
            derivatives[18] = diff[1] * (const + diff[2]**2 * coeff)
            derivatives[19] = diff[0] * diff[1] * diff[2] * coeff

        return derivatives

    def directivity_derivatives(self, source_positions, source_normals, receiver_positions, orders=3):
        """Calculate the spatial derivatives of the directivity.

        The default implementation uses finite difference stencils to evaluate the
        derivatives. In principle this means that customized directivity models
        does not need to implement their own derivatives, but can do so for speed
        and precision benefits.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')
        finite_difference_coefficients = {'': (np.array([[0, 0, 0]]).T, np.array([1]))}
        if orders > 0:
            finite_difference_coefficients['x'] = (np.array([[1, 0, 0], [-1, 0, 0]]).T, np.array([0.5, -0.5]))
            finite_difference_coefficients['y'] = (np.array([[0, 1, 0], [0, -1, 0]]).T, np.array([0.5, -0.5]))
            finite_difference_coefficients['z'] = (np.array([[0, 0, 1], [0, 0, -1]]).T, np.array([0.5, -0.5]))
        if orders > 1:
            finite_difference_coefficients['xx'] = (np.array([[1, 0, 0], [0, 0, 0], [-1, 0, 0]]).T, np.array([1, -2, 1]))  # Alt -- (np.array([[2, 0, 0], [0, 0, 0], [-2, 0, 0]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['yy'] = (np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]).T, np.array([1, -2, 1]))  # Alt-- (np.array([[0, 2, 0], [0, 0, 0], [0, -2, 0]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['zz'] = (np.array([[0, 0, 1], [0, 0, 0], [0, 0, -1]]).T, np.array([1, -2, 1]))  # Alt -- (np.array([[0, 0, 2], [0, 0, 0], [0, 0, -2]]), [0.25, -0.5, 0.25])
            finite_difference_coefficients['xy'] = (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]).T, np.array([0.25, 0.25, -0.25, -0.25]))
            finite_difference_coefficients['xz'] = (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1]]).T, np.array([0.25, 0.25, -0.25, -0.25]))
            finite_difference_coefficients['yz'] = (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1]]).T, np.array([0.25, 0.25, -0.25, -0.25]))
        if orders > 2:
            finite_difference_coefficients['xxx'] = (np.array([[2, 0, 0], [-2, 0, 0], [1, 0, 0], [-1, 0, 0]]).T, np.array([0.5, -0.5, -1, 1]))  # Alt -- (np.array([[3, 0, 0], [-3, 0, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['yyy'] = (np.array([[0, 2, 0], [0, -2, 0], [0, 1, 0], [0, -1, 0]]).T, np.array([0.5, -0.5, -1, 1]))  # Alt -- (np.array([[0, 3, 0], [0, -3, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['zzz'] = (np.array([[0, 0, 2], [0, 0, -2], [0, 0, 1], [0, 0, -1]]).T, np.array([0.5, -0.5, -1, 1]))  # Alt -- (np.array([[0, 0, 3], [0, 0, -3], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.375, 0.375])
            finite_difference_coefficients['xxy'] = (np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [0, 1, 0], [0, -1, 0]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[2, 1, 0], [-2, -1, 0], [2, -1, 0], [-2, 1, 0], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['xxz'] = (np.array([[1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [0, 0, 1], [0, 0, -1]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[2, 0, 1], [-2, 0, -1], [2, 0, -1], [-2, 0, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['yyx'] = (np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 0, 0], [-1, 0, 0]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[1, 2, 0], [-1, -2, 0], [-1, 2, 0], [1, -2, 0], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['yyz'] = (np.array([[0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 0, 1], [0, 0, -1]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[0, 2, 1], [0, -2, -1], [0, 2, -1], [0, -2, 1], [0, 0, 1], [0, 0, -1]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['zzx'] = (np.array([[1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 0], [-1, 0, 0]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[1, 0, 2], [-1, 0, -2], [-1, 0, 2], [1, 0, -2], [1, 0, 0], [-1, 0, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['zzy'] = (np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 0], [0, -1, 0]]).T, np.array([0.5, -0.5, -0.5, 0.5, -1, 1]))  # Alt -- (np.array([[0, 1, 2], [0, -1, -2], [0, -1, 2], [0, 1, -2], [0, 1, 0], [0, -1, 0]]), [0.125, -0.125, -0.125, 0.125, -0.25, 0.25])
            finite_difference_coefficients['xyz'] = (np.array([[1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [1, -1, 1], [-1, -1, 1], [1, 1, -1]]).T, np.array([1, -1, 1, -1, 1, -1, 1, -1]) * 0.125)

        derivatives = np.empty((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[1:], dtype=np.complex128)
        h = 1 / self.k
        # For all derivatives needed:
        for derivative, (shifts, weights) in finite_difference_coefficients.items():
            # Create the finite difference grid for all positions simultaneously by inserting a new axis for them (axis 1).
            # positions.shape = (3, n_difference_points, n_receiver_points)
            positions = shifts.reshape([3, -1] + (receiver_positions.ndim - 1) * [1]) * h + receiver_positions[:, np.newaxis, ...]
            # Calcualte the directivity at all positions at once, and weight them with the correct weights
            # weighted_values.shape = (n_difference_points, n_receiver_points)
            weighted_values = self.directivity(source_positions, source_normals, positions) * weights.reshape((source_positions.ndim - 1) * [1] + [-1] + (receiver_positions.ndim - 1) * [1])
            # sum the finite weighted points and store in the correct position in the output array.
            derivatives[utils.pressure_derivs_order.index(derivative)] = np.sum(weighted_values, axis=(source_positions.ndim - 1)) / h**len(derivative)
        return derivatives

    def spherical_harmonics(self, source_positions, source_normals, receiver_positions, orders=0, **kwargs):
        """Expand sound field in spherical harmonics.

        Performs a spherical harmonics expansion of the sound field created from the transducer model.
        The expansion is centered at the receiver position(s), and calculated by translating spherical
        wavefronts from the source position(s).

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of spherical harmonics coefficients to calculate.

        Returns
        -------
        coefficients : numpy.ndarray
            Array with the calculated expansion coefficients. Has the shape
            `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`,
            where `M=len(SphericalHarmonicsIndexer(orders))`, see `~levitate.utils.SphericalHarmonicsIndexer`
            for details on the structure of the coefficients.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')

        # We need the vector from the receiver to the source, since we are calculating an expansion centered
        # at the receiving point.
        diff = source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,)) - receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:])
        r = np.sum(diff**2, axis=0)**0.5
        kr = self.k * r
        colatitude = np.arccos(diff[2] / r)
        azimuth = np.arctan2(diff[1], diff[0])
        # Calculate the spherical hankel function of the second kind
        # See Williams Eq 8.22:
        # exp(jk|r-r'|) / (4pi |r-r'|) = jk sum_n j_n(k r_min) h_n(k r_max) sum_m Y_n^m (theta', phi')^* Y_n^m (theta, phi)
        # See Ahrens 2.37a with Errata for the 4pi
        # exp(-jk|r-r'|) / (4pi |r-r'|) = -jk sum_n j_n(k r_min) h^(2)_n(k r_max) sum_m Y_n^-m (theta', phi') Y_n^m (theta, phi)

        sph_idx = utils.SphericalHarmonicsIndexer(orders)
        coefficients = np.empty((len(sph_idx),) + source_positions.shape[1:2] + receiver_positions.shape[1:], dtype=np.complex128)
        for n in sph_idx.orders:
            hankel_func = spherical_jn(n, kr) + 1j * spherical_yn(n, kr)
            for m in sph_idx.modes:
                coefficients[sph_idx(n, m)] = hankel_func * np.conj(sph_harm(m, n, azimuth, colatitude))
        directivity = self.directivity(source_positions, source_normals, receiver_positions)
        return self.p0 * 4 * np.pi * 1j * self.k * directivity * coefficients


class TransducerReflector(TransducerModel):
    """Class for transducers with planar reflectors.

    This class can be used to add reflectors to all transducer models.
    This uses the image source method, so only infinite planar reflectors are
    possible.

    Parameters
    ----------
    transducer : `TrnsducerModel` instance or (sub)class
        The base transducer to reflect. If passed a class it will be instantiated
        with the remaining arguments not used by the reflector.
    plane_intersect : array_like, default (0, 0, 0)
        A point which the reflection plane intersects.
    plane_normal : array_like, default (0,0,1)
        3 element vector with the plane normal.
    reflection_coefficient : complex float, default 1
        Reflection coefficient to tune the magnitude and phase of the reflection.

    Returns
    -------
    transducer
        The transducer model with reflections.

    """

    _repr_fmt_spec = '{:%cls(transducer=%transducer_full, plane_intersect=%plane_intersect, plane_normal=%plane_normal, reflection_coefficient=%reflection_coefficient)}'
    _str_fmt_spec = '{:%cls(transducer=%transducer, plane_intersect=%plane_intersect, plane_normal=%plane_normal, reflection_coefficient=%reflection_coefficient)}'

    def __init__(self, transducer, plane_intersect=(0, 0, 0), plane_normal=(0, 0, 1), reflection_coefficient=1, *args, **kwargs):
        if type(transducer) is type:
            transducer = transducer(*args, **kwargs)
        self._transducer = transducer
        self.plane_intersect = np.asarray(plane_intersect, dtype=float)
        self.plane_normal = np.asarray(plane_normal, dtype=float)
        self.plane_normal /= (self.plane_normal**2).sum()**0.5
        self.reflection_coefficient = reflection_coefficient

    def __format__(self, fmt_str):
        s_out = fmt_str.replace('%transducer_full', repr(self._transducer)).replace('%transducer', str(self._transducer))
        s_out = s_out.replace('%plane_intersect', str(self.plane_intersect)).replace('%plane_normal', str(tuple(self.plane_normal)))
        s_out = s_out.replace('%reflection_coefficient', str(self.reflection_coefficient))
        return super().__format__(s_out)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self._transducer == other._transducer
            and np.allclose(self.plane_intersect, other.plane_intersect)
            and np.allclose(self.plane_normal, other.plane_normal)
            and np.allclose(self.reflection_coefficient, other.reflection_coefficient)
        )

    @property
    def omega(self):
        return self._transducer.omega

    @omega.setter
    def omega(self, val):
        self._transducer.omega = val

    @property
    def k(self):
        return self._transducer.k

    @k.setter
    def k(self, val):
        self._transducer.k = val

    @property
    def medium(self):
        return self._transducer.medium

    @medium.setter
    def medium(self, val):
        self._transducer.medium = val

    @property
    def p0(self):
        return self._transducer.p0

    @p0.setter
    def p0(self, val):
        self._transducer.p0 = val

    @property
    def physical_size(self):
        return self._transducer.physical_size

    @physical_size.setter
    def physical_size(self, val):
        self._transducer.physical_size = val

    def pressure_derivs(self, source_positions, source_normals, receiver_positions, *args, **kwargs):
        """Calculate the spatial derivatives of the greens function.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        return self._evaluate_with_reflector(self._transducer.pressure_derivs, source_positions, source_normals, receiver_positions, *args, **kwargs)

    def spherical_harmonics(self, source_positions, source_normals, receiver_positions, *args, **kwargs):
        """Evaluate the spherical harmonics expansion at a point.

        Mirrors the sources in the reflection plane and calculates the superposition of the expansions
        from the combined sources.
        For the full documentation of the parameters and output format, see the documentation of the
        spherical harmonics method of the underlying transducer model.

        """
        return self._evaluate_with_reflector(self._transducer.spherical_harmonics, source_positions, source_normals, receiver_positions, *args, **kwargs)

    def _evaluate_with_reflector(self, func, source_positions, source_normals, receiver_positions, *args, **kwargs):
        """Evaluate a function using a mirror source model.

        Calculates the positions and normals of the mirror sources. Evaluates the function
        using both the real sources and the mirrored sources. Adds the two results, considering
        some arbitrary complex reflections coefficient.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        plane_normal = self.plane_normal.reshape((3,) + (1,) * (source_positions.ndim - 1))
        plane_distance = np.sum(self.plane_normal * self.plane_intersect)
        mirror_position = source_positions - 2 * plane_normal * ((source_positions * plane_normal).sum(axis=0) - plane_distance)
        mirror_normal = source_normals - 2 * plane_normal * (source_normals * plane_normal).sum(axis=0)

        direct = func(source_positions, source_normals, receiver_positions, *args, **kwargs)
        reflected = func(mirror_position, mirror_normal, receiver_positions, *args, **kwargs)

        source_side = np.sign((source_positions * plane_normal).sum(axis=0) - plane_distance).reshape(source_positions.shape[1:] + (1,) * (receiver_positions.ndim - 1))
        receiver_side = np.sign(np.einsum('i...,i', receiver_positions, self.plane_normal) - plane_distance)
        # `source_side` and `receiver_side` are zero if the source or receiver is inside the plane.
        # `source_side * receiver_side` will be -1 if they are on different sides, 0 if any of them is in the plane, and 1 otherwise.
        # We should return 0 if the source and receiver is on different sides, otherwise we return the calculated expression.
        # The below expression maps (-1, 0, 1) to (0, 1, 1).
        same_side = np.sign(source_side * receiver_side + 1)

        return (direct + self.reflection_coefficient * reflected) * same_side


class PlaneWaveTransducer(TransducerModel):
    """Class representing planar waves.

    This is not representing a physical transducer per se, but a traveling
    plane wave.
    """

    def pressure_derivs(self, source_positions, source_normals, receiver_positions, orders=3, **kwargs):
        """Calculate the spatial derivatives of the greens function.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the shape (`(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        source_positions = np.asarray(source_positions)
        receiver_positions = np.asarray(receiver_positions)
        source_normals = np.asarray(source_normals, dtype=np.float64)
        source_normals /= (source_normals**2).sum(axis=0)**0.5
        source_normals = source_normals.reshape(source_normals.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        x_dot_n = np.einsum('i..., i...', diff, source_normals)

        derivatives = np.empty((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[1:], dtype=np.complex128)
        derivatives[0] = self.p0 * np.exp(1j * self.k * x_dot_n)

        if orders > 0:
            derivatives[1] = 1j * self.k * source_normals[0] * derivatives[0]
            derivatives[2] = 1j * self.k * source_normals[1] * derivatives[0]
            derivatives[3] = 1j * self.k * source_normals[2] * derivatives[0]
        if orders > 1:
            derivatives[4] = 1j * self.k * source_normals[0] * derivatives[3]
            derivatives[5] = 1j * self.k * source_normals[1] * derivatives[2]
            derivatives[6] = 1j * self.k * source_normals[2] * derivatives[3]
            derivatives[7] = 1j * self.k * source_normals[1] * derivatives[1]
            derivatives[8] = 1j * self.k * source_normals[0] * derivatives[3]
            derivatives[9] = 1j * self.k * source_normals[2] * derivatives[2]
        if orders > 2:
            derivatives[10] = 1j * self.k * source_normals[0] * derivatives[4]
            derivatives[11] = 1j * self.k * source_normals[1] * derivatives[5]
            derivatives[12] = 1j * self.k * source_normals[2] * derivatives[6]
            derivatives[13] = 1j * self.k * source_normals[1] * derivatives[4]
            derivatives[14] = 1j * self.k * source_normals[2] * derivatives[4]
            derivatives[15] = 1j * self.k * source_normals[0] * derivatives[5]
            derivatives[16] = 1j * self.k * source_normals[2] * derivatives[5]
            derivatives[17] = 1j * self.k * source_normals[0] * derivatives[6]
            derivatives[18] = 1j * self.k * source_normals[1] * derivatives[6]
            derivatives[19] = 1j * self.k * source_normals[2] * derivatives[7]
        return derivatives


class CircularPiston(PointSource):
    r"""Circular piston transducer model.

    Implementation of the circular piston directivity :math:`D(\theta) = 2 {J_1(ka\sin\theta) \over ka\sin\theta}`.

    Parameters
    ----------
    effective_radius : float
        The radius :math:`a` in the above.
    **kwargs
        See `TransducerModel`

    Note
    ----
    This class has no implementation of analytic jacobians yet, and is much slower to use than other models.

    """

    _repr_fmt_spec = '{:%cls(freq=%freq, p0=%p0, effective_radius=%effective_radius, medium=%mediumfull)}'
    _str_fmt_spec = '{:%cls(freq=%freq, p0=%p0, effective_radius=%effective_radius, medium=%medium)}'

    def __init__(self, effective_radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effective_radius = effective_radius

    def __format__(self, fmt_spec):
        return super().__format__(fmt_spec).replace('%effective_radius', str(self.effective_radius))

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.effective_radius, other.effective_radius)

    def directivity(self, source_positions, source_normals, receiver_positions):
        r"""Evaluate transducer directivity.

        Returns :math:`D(\theta) = 2 J_1(ka\sin\theta) / (ka\sin\theta)`
        where :math:`a` is the `effective_radius` of the transducer,
        :math:`k` is the wavenumber of the transducer (`k`),
        :math:`\theta` is the angle between the transducer normal
        and the vector from the transducer to the receiving point,
        and and :math:`J_1` is the first order Bessel function.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.

        Returns
        -------
        out : numpy.ndarray
            The amplitude (and phase) of the directivity, shape `source_positions.shape[1:] + receiver_positions.shape[1:]`.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dots = np.einsum('i...,i...', diff, source_normals)
        norm1 = np.einsum('i...,i...', source_normals, source_normals)**0.5
        norm2 = np.einsum('i...,i...', diff, diff)**0.5
        cos_angle = np.clip(dots / norm2 / norm1, -1, 1)  # Clip needed because numrical precicion sometimes give a value slightly outside the reasonable range.
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.effective_radius

        denom = ka * sin_angle
        numer = j1(denom)
        with np.errstate(invalid='ignore'):
            return np.where(denom == 0, 1, 2 * numer / denom)


class CircularRing(PointSource):
    r"""Circular ring transducer model.

    Implementation of the circular ring directivity :math:`D(\theta) = J_0(ka\sin\theta)`.

    Parameters
    ----------
    effective_radius : float
        The radius :math:`a` in the above.
    **kwargs
        See `TransducerModel`

    """

    _repr_fmt_spec = '{:%cls(freq=%freq, p0=%p0, effective_radius=%effective_radius, medium=%mediumfull)}'
    _str_fmt_spec = '{:%cls(freq=%freq, p0=%p0, effective_radius=%effective_radius, medium=%medium)}'

    def __init__(self, effective_radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effective_radius = effective_radius

    def __format__(self, fmt_spec):
        return super().__format__(fmt_spec).replace('%effective_radius', str(self.effective_radius))

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.effective_radius, other.effective_radius)

    def directivity(self, source_positions, source_normals, receiver_positions):
        r"""Evaluate transducer directivity.

        Returns :math:`D(\theta) = J_0(ka\sin\theta)` where
        :math:`a` is the `effective_radius` of the transducer,
        :math:`k` is the wavenumber of the transducer (`k`),
        :math:`\theta` is the angle between the transducer normal
        and the vector from the transducer to the receiving point,
        and :math:`J_0` is the zeroth order Bessel function.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.

        Returns
        -------
        out : numpy.ndarray
            The amplitude (and phase) of the directivity, shape `source_positions.shape[1:] + receiver_positions.shape[1:]`.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dots = np.einsum('i...,i...', diff, source_normals)
        norm1 = np.einsum('i...,i...', source_normals, source_normals)**0.5
        norm2 = np.einsum('i...,i...', diff, diff)**0.5
        cos_angle = np.clip(dots / norm2 / norm1, -1, 1)  # Clip needed because numrical precicion sometimes give a value slightly outside the reasonable range.
        sin_angle = (1 - cos_angle**2)**0.5
        ka = self.k * self.effective_radius
        return j0(ka * sin_angle)

    def directivity_derivatives(self, source_positions, source_normals, receiver_positions, orders=3):
        """Calculate the spatial derivatives of the directivity.

        Explicit implementation of the derivatives of the directivity, based
        on analytical differentiation.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        source_normals : numpy.ndarray
            The look direction of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.
        orders : int
            How many orders of derivatives to calculate. Currently three orders are supported.

        Returns
        -------
        derivatives : numpy.ndarray
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        source_positions = np.asarray(source_positions)
        source_normals = np.asarray(source_normals)
        receiver_positions = np.asarray(receiver_positions)
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dot = np.einsum('i...,i...', diff, source_normals)
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=0)**0.5
        n = source_normals
        norm = np.einsum('i...,i...', n, n)**0.5
        cos = np.clip(dot / r / norm, -1, 1)  # Clip needed because numrical precicion sometimes give a value slightly outside the reasonable range.
        sin = (1 - cos**2)**0.5
        ka = self.k * self.effective_radius
        ka_sin = ka * sin

        derivatives = np.empty((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[1:], dtype=np.complex128)
        J0 = j0(ka_sin)
        derivatives[0] = J0
        if orders > 0:
            r2 = r**2
            r3 = r**3
            cos_dx = (r2 * n[0] - diff[0] * dot) / r3 / norm
            cos_dy = (r2 * n[1] - diff[1] * dot) / r3 / norm
            cos_dz = (r2 * n[2] - diff[2] * dot) / r3 / norm

            with np.errstate(invalid='ignore'):
                J1_xi = np.where(sin == 0, 0.5, j1(ka_sin) / ka_sin)
            first_order_const = J1_xi * ka**2 * cos
            derivatives[1] = first_order_const * cos_dx
            derivatives[2] = first_order_const * cos_dy
            derivatives[3] = first_order_const * cos_dz

        if orders > 1:
            r5 = r2 * r3
            cos_dx2 = (3 * diff[0]**2 * dot - 2 * diff[0] * n[0] * r2 - dot * r2) / r5 / norm
            cos_dy2 = (3 * diff[1]**2 * dot - 2 * diff[1] * n[1] * r2 - dot * r2) / r5 / norm
            cos_dz2 = (3 * diff[2]**2 * dot - 2 * diff[2] * n[2] * r2 - dot * r2) / r5 / norm
            cos_dxdy = (3 * diff[0] * diff[1] * dot - r2 * (n[0] * diff[1] + n[1] * diff[0])) / r5 / norm
            cos_dxdz = (3 * diff[0] * diff[2] * dot - r2 * (n[0] * diff[2] + n[2] * diff[0])) / r5 / norm
            cos_dydz = (3 * diff[1] * diff[2] * dot - r2 * (n[1] * diff[2] + n[2] * diff[1])) / r5 / norm

            with np.errstate(invalid='ignore'):
                J2_xi2 = np.where(sin == 0, 0.125, (2 * J1_xi - J0) / ka_sin**2)
            second_order_const = J2_xi2 * ka**4 * cos**2 + J1_xi * ka**2
            derivatives[4] = second_order_const * cos_dx**2 + first_order_const * cos_dx2
            derivatives[5] = second_order_const * cos_dy**2 + first_order_const * cos_dy2
            derivatives[6] = second_order_const * cos_dz**2 + first_order_const * cos_dz2
            derivatives[7] = second_order_const * cos_dx * cos_dy + first_order_const * cos_dxdy
            derivatives[8] = second_order_const * cos_dx * cos_dz + first_order_const * cos_dxdz
            derivatives[9] = second_order_const * cos_dy * cos_dz + first_order_const * cos_dydz

        if orders > 2:
            r4 = r2**2
            r7 = r5 * r2
            cos_dx3 = (-15 * diff[0]**3 * dot + 9 * r2 * (diff[0]**2 * n[0] + diff[0] * dot) - 3 * r4 * n[0]) / r7 / norm
            cos_dy3 = (-15 * diff[1]**3 * dot + 9 * r2 * (diff[1]**2 * n[1] + diff[1] * dot) - 3 * r4 * n[1]) / r7 / norm
            cos_dz3 = (-15 * diff[2]**3 * dot + 9 * r2 * (diff[2]**2 * n[2] + diff[2] * dot) - 3 * r4 * n[2]) / r7 / norm
            cos_dx2dy = (-15 * diff[0]**2 * diff[1] * dot + 3 * r2 * (diff[0]**2 * n[1] + 2 * diff[0] * diff[1] * n[0] + diff[1] * dot) - r4 * n[1]) / r7 / norm
            cos_dx2dz = (-15 * diff[0]**2 * diff[2] * dot + 3 * r2 * (diff[0]**2 * n[2] + 2 * diff[0] * diff[2] * n[0] + diff[2] * dot) - r4 * n[2]) / r7 / norm
            cos_dy2dx = (-15 * diff[1]**2 * diff[0] * dot + 3 * r2 * (diff[1]**2 * n[0] + 2 * diff[1] * diff[0] * n[1] + diff[0] * dot) - r4 * n[0]) / r7 / norm
            cos_dy2dz = (-15 * diff[1]**2 * diff[2] * dot + 3 * r2 * (diff[1]**2 * n[2] + 2 * diff[1] * diff[2] * n[1] + diff[2] * dot) - r4 * n[2]) / r7 / norm
            cos_dz2dx = (-15 * diff[2]**2 * diff[0] * dot + 3 * r2 * (diff[2]**2 * n[0] + 2 * diff[2] * diff[0] * n[2] + diff[0] * dot) - r4 * n[0]) / r7 / norm
            cos_dz2dy = (-15 * diff[2]**2 * diff[1] * dot + 3 * r2 * (diff[2]**2 * n[1] + 2 * diff[2] * diff[1] * n[2] + diff[1] * dot) - r4 * n[1]) / r7 / norm
            cos_dxdydz = (-15 * diff[0] * diff[1] * diff[2] * dot + 3 * r2 * (n[0] * diff[1] * diff[2] + n[1] * diff[0] * diff[2] + n[2] * diff[0] * diff[1])) / r7 / norm

            with np.errstate(invalid='ignore'):
                J3_xi3 = np.where(sin == 0, 1 / 48, (4 * J2_xi2 - J1_xi) / ka_sin**2)
            third_order_const = J3_xi3 * ka**6 * cos**3 + 3 * J2_xi2 * ka**4 * cos
            derivatives[10] = third_order_const * cos_dx**3 + 3 * second_order_const * cos_dx2 * cos_dx + first_order_const * cos_dx3
            derivatives[11] = third_order_const * cos_dy**3 + 3 * second_order_const * cos_dy2 * cos_dy + first_order_const * cos_dy3
            derivatives[12] = third_order_const * cos_dz**3 + 3 * second_order_const * cos_dz2 * cos_dz + first_order_const * cos_dz3
            derivatives[13] = third_order_const * cos_dx**2 * cos_dy + second_order_const * (cos_dx2 * cos_dy + 2 * cos_dxdy * cos_dx) + first_order_const * cos_dx2dy
            derivatives[14] = third_order_const * cos_dx**2 * cos_dz + second_order_const * (cos_dx2 * cos_dz + 2 * cos_dxdz * cos_dx) + first_order_const * cos_dx2dz
            derivatives[15] = third_order_const * cos_dy**2 * cos_dx + second_order_const * (cos_dy2 * cos_dx + 2 * cos_dxdy * cos_dy) + first_order_const * cos_dy2dx
            derivatives[16] = third_order_const * cos_dy**2 * cos_dz + second_order_const * (cos_dy2 * cos_dz + 2 * cos_dydz * cos_dy) + first_order_const * cos_dy2dz
            derivatives[17] = third_order_const * cos_dz**2 * cos_dx + second_order_const * (cos_dz2 * cos_dx + 2 * cos_dxdz * cos_dz) + first_order_const * cos_dz2dx
            derivatives[18] = third_order_const * cos_dz**2 * cos_dy + second_order_const * (cos_dz2 * cos_dy + 2 * cos_dydz * cos_dz) + first_order_const * cos_dz2dy
            derivatives[19] = third_order_const * cos_dx * cos_dy * cos_dz + second_order_const * (cos_dx * cos_dydz + cos_dy * cos_dxdz + cos_dz * cos_dxdy) + first_order_const * cos_dxdydz

        return derivatives


class RectangularCylinderModes(TransducerModel):
    # TODO: Cleaup of comments.
    # DOCS: Needed!

    def __init__(self, Lx, Ly, Lz, dB_limit=(), selected_modes=(), damping=0.01, *args, **kwargs):
        # TODO: Better specification of size. Length + center? (side, side)?

        super().__init__(*args, **kwargs)

        self.Lx, self.Ly, self.Lz, self.dB_limit, self.selected_modes = Lx, Ly, Lz, dB_limit, selected_modes
        self.damping = damping

    def pressure_derivs(self, source_positions, source_normals, receiver_positions, orders=3):

        shift = np.array([self.Lx / 2, self.Ly / 2, 0])

        source_positions = np.asarray(source_positions)
        receiver_positions = np.asarray(receiver_positions)

        if receiver_positions.shape[1:] == ():
            receiver_positions = receiver_positions.reshape((3,) + (1,))

        source_positions = source_positions.reshape(source_positions.shape[:2] + (1,))
        receiver_positions = receiver_positions.reshape((3,) + (1,) + receiver_positions.shape[1:])

        derivatives = np.zeros((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[2:], dtype=np.complex128)

        # TODO: Check the conventions for the complex exponential. The jw might stem from d/ts -> jw, but with our conventions we have d/dt -> -iw. There's two j's I could find that might be relevant, one in the damping term, and one in the overall scaling.
        # IDEA: Include proper material based modelling of the damping coefficient?
        damping = self.damping

        # modal_derivatives = np.zeros((nx_max + 1, ny_max + 1, nz_max + 1, utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[2:], dtype=np.complex128)  # DEBUG
        # modal_frequencies = np.zeros((nx_max + 1, ny_max + 1, nz_max + 1))  # DEBUG

        if self.selected_modes == ():
            self.selected_modes = self.modes_selection()

        if np.ndim(self.selected_modes) == 1:
            self.selected_modes = np.asarray(self.selected_modes).reshape((1,) + (3,))

        for ii in range(len(self.selected_modes)):

            nx = self.selected_modes[ii][0]
            ny = self.selected_modes[ii][1]
            nz = self.selected_modes[ii][2]

            fact_x = nx * np.pi / self.Lx
            fact_y = ny * np.pi / self.Ly
            fact_z = nz * np.pi / self.Lz

            if nx != 0 and ny != 0 and nz != 0:
                Lambda = 1 / 8
            elif (nx != 0 and ny != 0) or (nx != 0 and nz != 0) or (ny != 0 and nz != 0):
                Lambda = 1 / 4
            else:
                Lambda = 1 / 2

            omega_mode = self.medium.c * np.sqrt((nx * np.pi / self.Lx)**2 + (ny * np.pi / self.Ly)**2 + (nz * np.pi / self.Lz)**2)
            source_modeshape = np.cos(fact_x * (source_positions[0] + shift[0])) * np.cos(fact_y * (source_positions[1] + shift[1])) * np.cos(fact_z * (source_positions[2] + shift[2]))

            constant = source_modeshape / (Lambda * (omega_mode**2 - self.omega**2 + 2 * 1j * omega_mode * damping))

            this_mode_derivatives = np.zeros((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[2:], dtype=np.complex128)  # DEBUG
            this_mode_derivatives[0] = constant * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))

            if orders > 0:
                this_mode_derivatives[1] = - constant * fact_x * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[2] = - constant * fact_y * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[3] = - constant * fact_z * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
            if orders > 1:
                this_mode_derivatives[4] = - constant * fact_x**2 * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[5] = - constant * fact_y**2 * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[6] = - constant * fact_z**2 * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[7] = constant * fact_x * fact_y * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[8] = constant * fact_x * fact_z * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[9] = constant * fact_y * fact_z * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
            if orders > 2:
                this_mode_derivatives[10] = constant * fact_x**3 * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[11] = constant * fact_y**3 * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[12] = constant * fact_z**3 * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[13] = constant * fact_x**2 * fact_y * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[14] = constant * fact_x**2 * fact_z * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[15] = constant * fact_y**2 * fact_x * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[16] = constant * fact_y**2 * fact_z * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[17] = constant * fact_z**2 * fact_x * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.cos(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[18] = constant * fact_z**2 * fact_y * np.cos(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.cos(fact_z * (receiver_positions[2] + shift[2]))
                this_mode_derivatives[19] = - constant * fact_x * fact_y * fact_z * np.sin(fact_x * (receiver_positions[0] + shift[0])) * np.sin(fact_y * (receiver_positions[1] + shift[1])) * np.sin(fact_z * (receiver_positions[2] + shift[2]))

            # modal_derivatives[nx, ny, nz] = this_mode_derivatives  # DEBUG
            # modal_frequencies[nx, ny, nz] = omega_mode / 2 / np.pi  # DEBUG
            derivatives += this_mode_derivatives # DEBUG: Add in place instead?

            # self.modal_derivatives = modal_derivatives  # DEBUG
            # self.modal_frequencies = modal_frequencies  # DEBUG

        # TODO: Make sure that the scaling makes sense!
        # We need a volume velocity! From Williams 1999, (6.71, p.198) pressure from a monopole:
        #   p = -i rho_0 c k / (4 pi) Q_s exp(ikr) / r
        #   Units: Pa = kg/m^3 m/s 1/m [Q] 1/m = kg/m^4/s [Q] => [Q] = Pa m^4 s / kg = N m^2 s/kg = kg m/s^2 m^2 s/kg = m^3 / s
        #   [Q] = m^3 / s, i.e. volume per second.
        # Our corresponding expression is:
        #   p = p_0 exp(ikr) / r
        # So p_0 = -i rho_0 c k / (4 pi) Q_s => Q_s =  4 pi p_0 / (i rho_0 c k) = 4 pi p_0 / (i w rho_0)
        # The expression
        #   jw rho_0 c^2 U / V = jw rho_0 c^2 (4 pi p_0 / (i w rho_0)) / V
        #   = ± 4 pi c^2 p_0 / V
        # Unit check: Pa = m^2/s^2 (Pa m) 1/m^3 sum s^2 = Pa. Ok!
        # self.modal_derivatives *= 1j * self.omega * self.medium.rho * self.medium.c**2 / (self.Lx * self.Ly * self.Lz)
        # derivatives *= 1j * self.omega * self.medium.rho * self.medium.c**2 / (self.Lx * self.Ly * self.Lz)
        derivatives *= 4 * np.pi * self.p0 * self.medium.c**2 / (self.Lx * self.Ly * self.Lz)  # This might be correct?

        return derivatives

        # Can this be generalized somehow to create a mode-shape transducer superclass?
        # Create separate functions for modeshape (+derivatives), modal frequency, modal amplitude.
        # Will be somewhat difficult since different geometries will have different number of indices for the modes.
        # Unless we choose to store them linearly and add indexing + generator methods somewhere?
        # Disadvantage: Storing all the values per mode takes several thousand times more memory, so a single transducer seems to need 2 GB of ram for just the pressure in a small slice...

    def modes_selection(self):

        damping = self.damping

        modal_amplitude = []
        modes_list = []
        selection = []

        nx = 0
        nx_max = np.floor(4 * self.freq * self.Lx / self.medium.c).astype(int)  # DEBUG
        while nx <= nx_max:
            ny = 0
            ny_max = np.floor(self.Ly * np.sqrt((4 * self.freq / self.medium.c) ** 2 - (nx / self.Lx) ** 2)).astype(int)  # DEBUG
            while ny <= ny_max:
                nz = 0
                nz_max = np.floor(self.Lz * np.sqrt((4 * self.freq / self.medium.c) ** 2 - (nx / self.Lx) ** 2 - (ny / self.Ly) ** 2)).astype(int)  # DEBUG
                while nz <= nz_max:

                    if nx != 0 and ny != 0 and nz != 0:
                        Lambda = 1 / 8
                    elif (nx != 0 and ny != 0) or (nx != 0 and nz != 0) or (ny != 0 and nz != 0):
                        Lambda = 1 / 4
                    else:
                        Lambda = 1 / 2

                    omega_mode = self.medium.c * np.sqrt((nx * np.pi / self.Lx) ** 2 + (ny * np.pi / self.Ly) ** 2 + (nz * np.pi / self.Lz) ** 2)

                    modal_amplitude.append(1 / (Lambda * (omega_mode ** 2 - self.omega ** 2 + 2 * 1j * omega_mode * damping)))
                    modes_list.append((nx, ny, nz))

                    nz += 1
                ny += 1
            nx += 1

        if self.dB_limit == ():
            selection = modes_list
        else:
            modal_amplitude_max = 20 * np.log10(max(np.absolute(modal_amplitude)))
            for ii in range(0, len(modal_amplitude)-1):
                if modal_amplitude_max - 20*np.log10(np.absolute(modal_amplitude[ii])) <= self.dB_limit:
                    selection.append(modes_list[ii])

        print(str(len(selection)) + " modes used")
        return selection

class CylinderModes(TransducerModel):
    # TODO: Cleaup of comments.
    # DOCS: Needed!

    def __init__(self, radius, height, dB_limit=(), selected_modes=(), damping=0.01, *args, **kwargs):
        # TODO: Better specification of size. Length + center? (side, side)?

        super().__init__(*args, **kwargs)

        self.radius, self.height, self.dB_limit, self.selected_modes = radius, height, dB_limit, selected_modes
        self.damping = damping

    def _maximum_bessel_order(self, wavenumber, one_above=False):
        """Find the highest order of Bessel function with resonances below a given wavenumber."""
        kappa = wavenumber * self.radius
        n_max = np.math.ceil(kappa - np.log(kappa))
        while jnp_zeros(n_max, 1)[0] < kappa:
            n_max += 1
        return n_max - (0 if one_above else 1)  # Remove one since we increment until we are above

    def _bessel_deriv_zeros(self, order, wavenumber, one_above=False):
        """Find the wavenumbers where the derivative of the Berssel function is zero, below a given wavenumber."""
        kappa = wavenumber * self.radius
        s_max = max(np.math.ceil((kappa - order) / 3), 1)
        kappa_zeros = jnp_zeros(order, s_max)

        while kappa_zeros[-1] < kappa:
            # Initial guess included too few zeros
            s_max += np.math.ceil((kappa - kappa_zeros[-1]) / 3)
            kappa_zeros = jnp_zeros(order, s_max)

        if kappa_zeros[-1] > kappa:
            # Initial guess included too many zeros
            s_max -= np.sum(kappa_zeros > kappa)
            kappa_zeros = kappa_zeros[:s_max + (1 if one_above else 0)]

        return kappa_zeros / self.radius

    def _closest_resoance(self):
        """Find the wavenumber of the closest resonance for the cavity."""
        n_max = self._maximum_bessel_order(self.k, one_above=True)

        smallest_difference = np.inf

        # Check for n=s=0
        m_above = np.math.ceil(self.height / np.pi * self.k)
        m_below = m_above - 1
        k_above = (m_above * np.pi / self.height)
        k_below = (m_below * np.pi / self.height)
        if np.abs(k_above - self.k) < smallest_difference:
            smallest_difference = np.abs(k_above - self.k)
            closest_wavenumber = k_above
            closest_mode = (0, 0, m_above)
        if np.abs(k_below - self.k) < smallest_difference:
            smallest_difference = np.abs(k_below - self.k)
            closest_wavenumber = k_below
            closest_mode = (0, 0, m_below)

        for n in range(0, n_max + 1):
            for s, radial_wavenumber in enumerate(self._bessel_deriv_zeros(n, self.k, one_above=True), 1):
                if radial_wavenumber > self.k:
                    if np.abs(radial_wavenumber - self.k) < smallest_difference:
                        smallest_difference = np.abs(radial_wavenumber - self.k)
                        closest_wavenumber = radial_wavenumber
                        closest_mode = (n, s, 0)
                else:
                    m_above = np.math.ceil(self.height / np.pi * (self.k**2 - radial_wavenumber**2)**0.5)
                    m_below = m_above - 1
                    k_above = (radial_wavenumber**2 + (m_above * np.pi / self.height)**2)**0.5
                    k_below = (radial_wavenumber**2 + (m_below * np.pi / self.height)**2)**0.5
                    if np.abs(k_above - self.k) < smallest_difference:
                        smallest_difference = np.abs(k_above - self.k)
                        closest_wavenumber = k_above
                        closest_mode = (n, s, m_above)
                    if np.abs(k_below - self.k) < smallest_difference:
                        smallest_difference = np.abs(k_below - self.k)
                        closest_wavenumber = k_below
                        closest_mode = (n, s, m_below)

        self._closest_mode = closest_mode
        return closest_wavenumber

    def _cutoff_wavenumber(self):
        """Find the cutoff wavenumber above which all modes will be below the chosen threshold."""
        closest_wavenumber = self._closest_resoance()
        max_strength = np.abs(closest_wavenumber**2 - self.k**2 + 2j * closest_wavenumber * self.damping / self.medium.c)
        min_strength = max_strength * 10**(self.dB_limit / 20)
        k_max = (min_strength + self.k**2)**0.5
        return k_max


    def pressure_derivs(self, source_positions, source_normals, receiver_positions, orders=3):
        source_positions = np.asarray(source_positions)
        receiver_positions = np.asarray(receiver_positions)

        source_dims = source_positions.ndim - 1
        receiver_dims = receiver_positions.ndim - 1
        source_positions = source_positions.reshape(source_positions.shape[:2] + receiver_dims * (1,))
        receiver_positions = receiver_positions.reshape((3,) + (1,) * source_dims + receiver_positions.shape[1:])

        x_src, y_src, z_src = source_positions
        x_rec, y_rec, z_rec = receiver_positions
        rho_src = (x_src**2 + y_src**2)**0.5
        rho_rec = (x_rec**2 + y_rec**2)**0.5
        theta_src = np.arctan2(y_src, x_src)
        theta_rec = np.arctan2(y_rec, x_rec)

        if orders > 0:
            # We use the normalized values in the calculations, but we don't
            # have to recalculate these values for every mode.
            xn = x_rec / rho_rec  # Normalized x
            yn = y_rec / rho_rec  # Normalized y
            xnxn = xn * xn
            ynyn = yn * yn
            xnyn = xn * yn
        else:
            xn = yn = xnxn = ynyn = xnyn = None

        output_shape = (utils.num_pressure_derivs[orders],) + source_positions.shape[1:source_dims + 1] + receiver_positions.shape[source_dims + 1:]

        derivatives = np.zeros(output_shape, dtype=np.complex128)
        damping = self.damping / self.medium.c
        radius = self.radius
        height = self.height

        k_max = self._cutoff_wavenumber()
        included_modes = {}
        # Handle n=s=0
        m_max = np.math.floor(k_max * height / np.pi)
        bessel_function = bessel_derivative = bessel_second_derivative = bessel_third_derivative = None
        for m in range(m_max + 1):
            axial_wavenumber = mode_wavenumber = m * np.pi / height
            axial_normalization = 1 if m == 0 else 0.5
            resonance_factor = mode_wavenumber**2 - self.k**2 + 2j * mode_wavenumber * damping

            cos_z = np.cos(axial_wavenumber * z_rec)
            sin_z = np.sin(axial_wavenumber * z_rec)

            included_modes[(0, 0, m)] = mode_wavenumber
            derivatives += self._single_mode_derivatives(
                orders, xn, yn, xnxn, ynyn, xnyn, rho_rec, output_shape,
                1, 0, 0, 0,  # J_0(0), J'_0(0), J''_0(0), J'''_0(0)
                0, 0, 1,  # n, sin(n phi'), cos(n phi')
                sin_z, cos_z, axial_wavenumber
            ) * np.cos(axial_wavenumber * z_src) / (resonance_factor * axial_normalization)

        n_max = self._maximum_bessel_order(k_max)
        for n in range(n_max + 1):
            angle = n * (theta_rec - theta_src)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            for s, radial_wavenumber in enumerate(self._bessel_deriv_zeros(n, k_max), 1):
                # Calculate kappa dependent quantities
                radial_normalization = (1 - (n / (radial_wavenumber * radius))**2) * jv(n, radial_wavenumber * radius)**2 / (1 if n == 0 else 2)
                bessel_function = jv(n, radial_wavenumber * rho_rec)
                source_bessel_function = jv(n, radial_wavenumber * rho_src)
                if orders > 0:
                    bessel_derivative = jvp(n, radial_wavenumber * rho_rec, 1) * radial_wavenumber
                if orders > 1:
                    bessel_second_derivative = jvp(n, radial_wavenumber * rho_rec, 2) * radial_wavenumber**2
                if orders > 2:
                    bessel_third_derivative = jvp(n, radial_wavenumber * rho_rec, 3) * radial_wavenumber**3

                m_max = np.math.floor(height / np.pi * (k_max**2 - radial_wavenumber**2)**0.5)
                for m in range(m_max + 1):
                    axial_wavenumber = m * np.pi / height
                    mode_wavenumber = (radial_wavenumber**2 + axial_wavenumber**2)**0.5

                    axial_normalization = 1 if m == 0 else 0.5
                    resonance_factor = mode_wavenumber**2 - self.k**2 + 2j * mode_wavenumber * damping

                    cos_z = np.cos(axial_wavenumber * z_rec)
                    sin_z = np.sin(axial_wavenumber * z_rec)

                    included_modes[(n, s, m)] = mode_wavenumber
                    tmp = self._single_mode_derivatives(
                        orders, xn, yn, xnxn, ynyn, xnyn, rho_rec, output_shape,
                        bessel_function, bessel_derivative, bessel_second_derivative, bessel_third_derivative,
                        n, sin_angle, cos_angle,
                        sin_z, cos_z, axial_wavenumber
                    )
                    tmp = tmp * (source_bessel_function * np.cos(axial_wavenumber * z_src)) / (resonance_factor * radial_normalization * axial_normalization)
                    derivatives += tmp

        derivatives *= self.p0 / (np.pi * radius ** 2 * height)
        self._included_modes = included_modes
        return derivatives

    def _single_mode_derivatives(
        self, orders,
        xn, yn, xnxn, ynyn, xnyn, rho_rec, output_shape,
        bessel_function, bessel_derivative, bessel_second_derivative, bessel_third_derivative,
        n, sin_angle, cos_angle,
        sin_z, cos_z, axial_wavenumber
    ):
        # This function have been optimized w.r.t. operation ordering. Keep that in mind before changing anything.
        this_mode_derivatives = np.zeros(output_shape, dtype=np.float64)
        this_mode_derivatives[0] = bessel_function * cos_z * cos_angle

        if orders > 0:
            bessel_function_over_rho = bessel_function / rho_rec
            # d/dx
            this_mode_derivatives[1] = (xn * bessel_derivative * cos_angle + n * yn * bessel_function_over_rho * sin_angle) * cos_z
            # d/dy
            this_mode_derivatives[2] = (yn * bessel_derivative * cos_angle - n * xn * bessel_function_over_rho * sin_angle) * cos_z
            # d/dz
            this_mode_derivatives[3] = -bessel_function * axial_wavenumber * sin_z * cos_angle

        if orders > 1:
            bessel_derivative_over_rho = bessel_derivative / rho_rec
            bessel_function_over_rho2 = bessel_function_over_rho / rho_rec
            # d^2/dx^2
            this_mode_derivatives[4] = (
                (
                    bessel_second_derivative * xnxn
                    + (bessel_derivative_over_rho - n * n * bessel_function_over_rho2) * ynyn
                ) * cos_angle
                + 2 * n * (bessel_derivative_over_rho - bessel_function_over_rho2) * xnyn * sin_angle
            ) * cos_z
            # d^2/dy^2
            this_mode_derivatives[5] = (
                (
                    bessel_second_derivative * ynyn
                    + (bessel_derivative_over_rho - n * n * bessel_function_over_rho2) * xnxn
                ) * cos_angle
                - 2 * n * (bessel_derivative_over_rho - bessel_function_over_rho2) * xnyn * sin_angle
            ) * cos_z
            # d^2/dz^2
            this_mode_derivatives[6] = -bessel_function * cos_z * axial_wavenumber**2 * cos_angle
            # d^2/dxdy
            this_mode_derivatives[7] = (
                (
                    bessel_second_derivative
                    - bessel_derivative_over_rho
                    + n * n * bessel_function_over_rho2
                ) * xnyn * cos_angle
                + (
                    bessel_derivative_over_rho - bessel_function_over_rho2
                ) * n * (ynyn - xnxn) * sin_angle
            ) * cos_z
            # d^2/dxdz
            this_mode_derivatives[8] = -(bessel_derivative * xn * cos_angle + n * yn * bessel_function_over_rho * sin_angle) * axial_wavenumber * sin_z
            # d^2/dydz
            this_mode_derivatives[9] = -(bessel_derivative * yn * cos_angle - n * xn * bessel_function_over_rho * sin_angle) * axial_wavenumber * sin_z

        if orders > 2:
            bessel_second_derivative_over_rho = bessel_second_derivative / rho_rec
            bessel_derivative_over_rho2 = bessel_derivative_over_rho / rho_rec
            bessel_function_over_rho3 = bessel_function_over_rho2 / rho_rec
            # d^3/dx^3
            this_mode_derivatives[10] = (
                (
                    bessel_third_derivative * xnxn
                    + (
                        bessel_second_derivative_over_rho
                        - bessel_derivative_over_rho2 * (n * n + 1)
                        + bessel_function_over_rho3 * (2 * n * n)
                    ) * 3 * ynyn
                ) * xn * cos_angle
                + (
                    bessel_second_derivative_over_rho * 3 * xnxn
                    - bessel_derivative_over_rho2 * 3 * (2 * xnxn - ynyn)
                    + bessel_function_over_rho3 * (6 * xnxn - (n * n + 2) * ynyn)
                ) * yn * n * sin_angle
            ) * cos_z
            # d^3/dy^3
            this_mode_derivatives[11] = (
                (
                    bessel_third_derivative * ynyn
                    + (
                        bessel_second_derivative_over_rho
                        - bessel_derivative_over_rho2 * (n * n + 1)
                        + bessel_function_over_rho3 * (2 * n * n)
                    ) * 3 * xnxn
                ) * yn * cos_angle
                + (
                    - bessel_second_derivative_over_rho * 3 * ynyn
                    + bessel_derivative_over_rho2 * 3 * (2 * ynyn - xnxn)
                    - bessel_function_over_rho3 * (6 * ynyn - (n * n + 2) * xnxn)
                ) * xn * n * sin_angle
            ) * cos_z
            # d^3/dz^3
            this_mode_derivatives[12] = bessel_function * sin_z * axial_wavenumber**3 * cos_angle
            # d^3/dx^2dy
            this_mode_derivatives[13] = (
                (
                    bessel_third_derivative * xnxn
                    + (
                        bessel_second_derivative_over_rho
                        - bessel_derivative_over_rho2 * (n * n + 1)
                        + bessel_function_over_rho3 * (n * n * 2)
                    ) * (ynyn - 2 * xnxn)
                ) * yn * cos_angle
                + (
                    bessel_second_derivative_over_rho * (-xnxn + 2 * ynyn)
                    + bessel_derivative_over_rho2 * (2 * xnxn - 7 * ynyn)
                    + bessel_function_over_rho3 * (-2 * xnxn + (n * n + 6) * ynyn)
                ) * n * xn * sin_angle
            ) * cos_z
            # d^3/dx^2dz
            this_mode_derivatives[14] = -(
                (
                    bessel_second_derivative * xnxn
                    + (bessel_derivative_over_rho - n * n * bessel_function_over_rho2) * ynyn
                ) * cos_angle
                + (bessel_derivative_over_rho - bessel_function_over_rho2) * (2 * n) * xnyn * sin_angle
            ) * sin_z * axial_wavenumber
            # d^3/dy^2dx
            this_mode_derivatives[15] = (
                (
                    bessel_third_derivative * ynyn
                    + (
                        bessel_second_derivative_over_rho
                        - bessel_derivative_over_rho2 * (n * n + 1)
                        + bessel_function_over_rho3 * (2 * n * n)
                    ) * (xnxn - 2 * ynyn)
                ) * xn * cos_angle
                + (
                    bessel_second_derivative_over_rho * (-2 * xnxn + ynyn)
                    + bessel_derivative_over_rho2 * (7 * xnxn - 2 * ynyn)
                    + bessel_function_over_rho3 * (-(n * n + 6) * xnxn + 2 * ynyn)
                ) * yn * n * sin_angle
            ) * cos_z
            # d^3/dy^2dz
            this_mode_derivatives[16] = -(
                (
                    bessel_second_derivative * ynyn
                    + (bessel_derivative_over_rho - n * n * bessel_function_over_rho2) * xnxn
                ) * cos_angle
                - (bessel_derivative_over_rho - bessel_function_over_rho2) * (2 * n) * xnyn * sin_angle
            ) * sin_z * axial_wavenumber
            # d^3/dz^dx
            this_mode_derivatives[17] = -(
                xn * bessel_derivative * cos_angle + n * yn * bessel_function_over_rho * sin_angle
            ) * cos_z * axial_wavenumber**2
            # d^3/dz^3dy
            this_mode_derivatives[18] = -(
                yn * bessel_derivative * cos_angle - n * xn * bessel_function_over_rho * sin_angle
            ) * cos_z * axial_wavenumber**2
            # d^3/dxdydz
            this_mode_derivatives[19] = -(
                (
                    bessel_second_derivative
                    - bessel_derivative_over_rho
                    + n * n * bessel_function_over_rho2
                ) * xnyn * cos_angle
                + (bessel_derivative_over_rho - bessel_function_over_rho2) * n * (ynyn - xnxn) * sin_angle
            ) * sin_z * axial_wavenumber

        return this_mode_derivatives


    def pressure_derivs_legacy(self, source_positions, source_normals, receiver_positions, orders=3):

        source_positions = np.asarray(source_positions)
        receiver_positions = np.asarray(receiver_positions)

        source_dims = source_positions.ndim - 1
        receiver_dims = receiver_positions.ndim - 1
        source_positions = source_positions.reshape(source_positions.shape[:2] + receiver_dims * (1,))
        receiver_positions = receiver_positions.reshape((3,) + (1,) * source_dims + receiver_positions.shape[1:])

        x_src, y_src, z_src = source_positions
        x_rec, y_rec, z_rec = receiver_positions
        rho_src = (x_src**2 + y_src**2)**0.5
        rho_rec = (x_rec**2 + y_rec**2)**0.5
        theta_src = np.arctan2(y_src, x_src)
        theta_rec = np.arctan2(y_rec, x_rec)

        output_shape = (utils.num_pressure_derivs[orders],) + source_positions.shape[1:source_dims + 1] + receiver_positions.shape[source_dims + 1:]

        derivatives = np.zeros(output_shape, dtype=np.complex128)
        damping = self.damping

        if orders > 0:
            # We use the normalized values in the calculations below, but we don't
            # have to recalculate these values for every mode.
            xn = x_rec / rho_rec  # Normalized x
            yn = y_rec / rho_rec  # Normalized y

        if self.selected_modes == ():
            self.selected_modes = self.modes_selection()
        else:
            for ii in range(len(self.selected_modes)):
                if isinstance(self.selected_modes[ii][1], int) and self.selected_modes[ii][1] != 0:
                    self.selected_modes[ii][1] = jnp_zeros(self.selected_modes[ii][0], self.selected_modes[ii][1])[-1]
                elif not isinstance(self.selected_modes[ii][1], int):
                    break

        if np.ndim(self.selected_modes) == 1:
            self.selected_modes = np.asarray(self.selected_modes).reshape((1,) + (3,))

        for ii in range(len(self.selected_modes)):

            n = self.selected_modes[ii][0]
            k_ns = self.selected_modes[ii][1]
            m = self.selected_modes[ii][2]

            if m == 0:
                e_m = 1
            else:
                e_m = 2

            if k_ns == 0 and n == 0:
                Lambda = 1 / e_m
            else:
                Lambda = (1 - (n / k_ns)**2) * jv(n, k_ns)**2 / e_m

            omega_mode = self.medium.c * np.sqrt((k_ns / self.radius)**2 + (m * np.pi / self.height)**2)
            source_modeshape = jv(n, k_ns * rho_src / self.radius) * np.exp(1j * n * theta_src) * np.cos(m * np.pi / self.height * z_src)
            constant = source_modeshape / (Lambda * (omega_mode**2 - self.omega**2 + 2 * 1j * omega_mode * damping))

            this_mode_derivatives = np.zeros(output_shape, dtype=np.complex128)  # DEBUG

            bessel_function = jv(n, k_ns * rho_rec / self.radius)
            z_factor = m * np.pi / self.height
            cos_z = np.cos(z_factor * z_rec)
            theta_phase = np.exp(1j * n * theta_rec)

            this_mode_derivatives[0] = constant * bessel_function * theta_phase * cos_z

            if orders > 0:
                sin_z = np.sin(m * np.pi * z_rec / self.height)  # Don't change the order of operations on this line. Numerical precision will break the tests...]
                bessel_derivative = jvp(n, k_ns * rho_rec / self.radius, 1) * k_ns / self.radius
                # d/dx
                this_mode_derivatives[1] = constant * (xn * bessel_derivative - 1j * n * yn / rho_rec * bessel_function) * theta_phase * cos_z
                # d/dy
                this_mode_derivatives[2] = constant * (yn * bessel_derivative + 1j * n * xn / rho_rec * bessel_function) * theta_phase * cos_z
                # d/dz
                this_mode_derivatives[3] = -constant * bessel_function * theta_phase * sin_z * z_factor
            if orders > 1:
                bessel_second_derivative = jvp(n, k_ns * rho_rec / self.radius, 2) * (k_ns / self.radius)**2
                # d^2/dx^2
                this_mode_derivatives[4] = constant * (
                    bessel_second_derivative * xn**2
                    + bessel_derivative / rho_rec * (-2j * n * xn + yn) * yn
                    + bessel_function / rho_rec**2 * (2j * xn - n * yn) * n * yn
                ) * cos_z * theta_phase
                # d^2/dy^2
                this_mode_derivatives[5] = constant * (
                    bessel_second_derivative * yn**2
                    + bessel_derivative / rho_rec * (xn + 2j * n * yn) * xn
                    + bessel_function / rho_rec**2 * (-n * xn - 2j * yn) * n * xn
                ) * cos_z * theta_phase
                # d^2/dz^2
                this_mode_derivatives[6] = -constant * bessel_function * theta_phase * cos_z * z_factor**2
                # d^2/dxdy
                this_mode_derivatives[7] = constant * (
                    bessel_second_derivative * xn * yn
                    + bessel_derivative / rho_rec * (1j * n * (xn**2 - yn**2) - xn * yn)
                    + bessel_function / rho_rec**2 * (1j * (yn**2 - xn**2) + n * xn * yn) * n
                ) * cos_z * theta_phase
                # d^2/dxdz
                this_mode_derivatives[8] = -constant * (bessel_derivative * xn - 1j * n * yn / rho_rec * bessel_function) * theta_phase * sin_z * z_factor
                # d^2/dydz
                this_mode_derivatives[9] = -constant * (bessel_derivative * yn + 1j * n * xn / rho_rec * bessel_function) * theta_phase * sin_z * z_factor
            if orders > 2:
                bessel_third_derivative = jvp(n, k_ns * rho_rec / self.radius, 3) * (k_ns / self.radius)**3
                # d^3/dx^3
                this_mode_derivatives[10] = constant * (
                    bessel_third_derivative * xn**3
                    + bessel_second_derivative / rho_rec * 3 * (-1j * n * xn + yn) * xn * yn
                    + bessel_derivative / rho_rec**2 * 3 * (1j * n * (2 * xn**2 - yn**2) - (n**2 + 1) * xn * yn) * yn
                    + bessel_function / rho_rec**3 * (-6j * xn**2 + 6 * n * xn * yn + 1j * (n**2 + 2) * yn**2) * yn * n
                ) * theta_phase * cos_z
                # d^3/dy^3
                this_mode_derivatives[11] = constant * (
                    bessel_third_derivative * yn**3
                    + bessel_second_derivative / rho_rec * 3 * (xn + 1j * n * yn) * xn * yn
                    + bessel_derivative / rho_rec**2 * 3 * (1j * n * xn**2 - (n**2 + 1) * xn * yn - 2j * n * yn**2) * xn
                    + bessel_function / rho_rec**3 * (-1j * (n**2 + 2) * xn**2 + 6 * n * xn * yn + 6j * yn**2) * xn * n
                ) * theta_phase * cos_z
                # d^3/dz^3
                this_mode_derivatives[12] = constant * bessel_function * theta_phase * sin_z * z_factor**3
                # d^3/dx^2dy
                this_mode_derivatives[13] = constant * (
                    bessel_third_derivative * xn**2 * yn
                    + bessel_second_derivative / rho_rec * (1j * n * xn**3 - 2 * xn**2 * yn - 2j * n * xn * yn**2 + yn**3)
                    + bessel_derivative / rho_rec**2 * (-2j * n * xn**3 + 2 * (n**2 + 1) * xn**2 * yn + 7j * n * xn * yn**2 - (n**2 + 1) * yn**3)
                    + bessel_function / rho_rec**3 * (2j * xn**3 - 4 * n * xn**2 * yn - 1j * (n**2 + 6) * xn * yn**2 + 2 * n * yn**3) * n
                ) * theta_phase * cos_z
                # d^3/dx^2dz
                this_mode_derivatives[14] = -constant * (
                    bessel_second_derivative * xn**2
                    + bessel_derivative / rho_rec * (-2j * n * xn + yn) * yn
                    + bessel_function / rho_rec**2 * (2j * xn - n * yn) * n * yn
                ) * theta_phase * sin_z * z_factor
                # d^3/dy^2dx
                this_mode_derivatives[15] = constant * (
                    bessel_third_derivative * xn * yn**2
                    + bessel_second_derivative / rho_rec * (xn**3 + 2j * n * xn**2 * yn - 2 * xn * yn**2 - 1j * n * yn**3)
                    + bessel_derivative / rho_rec**2 * (-(n**2 + 1) * xn**3 - 7j * n * xn**2 * yn + 2 * (n**2 + 1) * xn * yn**2 + 2j * n * yn**3)
                    + bessel_function / rho_rec**3 * (2 * n * xn**3 + 1j * (n**2 + 6) * xn**2 * yn - 4 * n * xn * yn**2 - 2j * yn**3) * n
                ) * theta_phase * cos_z
                # d^3/dy^2dz
                this_mode_derivatives[16] = -constant * (
                    bessel_second_derivative * yn**2
                    + bessel_derivative / rho_rec * (xn + 2j * n * yn) * xn
                    + bessel_function / rho_rec**2 * (-n * xn - 2j * yn) * n * xn
                ) * theta_phase * sin_z * z_factor
                # d^3/dz^dx
                this_mode_derivatives[17] = -constant * (
                    xn * bessel_derivative - 1j * n * yn / rho_rec * bessel_function
                ) * theta_phase * cos_z * z_factor**2
                # d^3/dz^3dy
                this_mode_derivatives[18] = -constant * (
                    yn * bessel_derivative + 1j * n * xn / rho_rec * bessel_function
                ) * theta_phase * cos_z * z_factor**2
                # d^3/dxdydz
                this_mode_derivatives[19] = -constant * (
                    bessel_second_derivative * xn * yn
                    + bessel_derivative / rho_rec * (1j * n * (xn**2 - yn**2) - xn * yn)
                    + bessel_function / rho_rec**2 * (1j * (yn**2 - xn**2) + n * xn * yn) * n
                ) * theta_phase * sin_z * z_factor

            derivatives += this_mode_derivatives

        derivatives *= self.p0 * self.medium.c**2 / (np.pi * self.radius**2 * self.height)  # This might be correct?

        return derivatives

    def modes_selection(self):

        damping = self.damping

        modal_amplitude = []
        modes_list = []
        selection = []

        n = 0
        k_n1 = jnp_zeros(0, 1)[-1]
        k_ns_max = np.floor(2*self.omega / self.medium.c * self.radius).astype(int)  # DEBUG
        while k_n1 <= k_ns_max:

            if n == 0:
                s = 0
                k_ns = 0
            else:
                s = 1
                k_ns = k_n1

            while k_ns <= k_ns_max:
                m = 0
                m_max = np.floor(self.radius * np.sqrt((2*self.omega / self.medium.c) ** 2 - (m *np.pi / self.height) ** 2)).astype(int)  # DEBUG
                # m_max = np.math.floor(self.height / self.radius / np.pi * (k_ns_max**2 - k_ns**2)**0.5)
                while m <= m_max:

                    if m == 0:
                        e_m = 1
                    else:
                        e_m = 2

                    if s == 0:
                        Lambda = 1 / e_m
                    else:
                        Lambda = (1 - (n/k_ns)**2) * jv(n, k_ns)**2 / e_m

                    omega_mode = self.medium.c * np.sqrt((k_ns / self.radius)**2 + (m * np.pi / self.height)**2)

                    modal_amplitude.append(1 / (Lambda * (omega_mode ** 2 - self.omega ** 2 + 2 * 1j * omega_mode * damping)))
                    # modes_list.append((n, k_ns, m))
                    modes_list.append((n, s, m))

                    m += 1
                s += 1
                k_ns = jnp_zeros(n, s)[-1]
            n += 1
            k_n1 = jnp_zeros(n, 1)[-1]

        if self.dB_limit == ():
            selection = modes_list
        else:
            modal_levels = 20 * np.log10(np.abs(modal_amplitude))
            modal_level_max = np.max(modal_levels)
            for ii in range(0, len(modal_amplitude)-1):
                if np.any((modal_level_max - modal_levels[ii]) <= self.dB_limit):
                    selection.append(modes_list[ii])
                    if modes_list[ii][0] != 0:
                        selection.append((-modes_list[ii][0], modes_list[ii][1], modes_list[ii][2]))

        print(str(len(selection)) + " modes used")
        return selection
