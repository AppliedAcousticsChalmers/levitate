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
from scipy.special import j0, j1
from scipy.special import spherical_jn, spherical_yn, sph_harm
from .materials import Air
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

    _repr_fmt_spec = '{:%cls(freq=%freq, p0=%p0, medium=%mediumfull)}'
    _str_fmt_spec = '{:%cls(freq=%freq, p0=%p0, medium=%medium)}'

    def __init__(self, freq=40e3, p0=6, medium=Air):
        self.medium = medium
        self.freq = freq
        self.p0 = p0
        # The murata transducers are measured to 85 dB SPL at 1 V at 1 m, which corresponds to ~6 Pa at 20 V
        # The datasheet specifies 120 dB SPL @ 0.3 m, which corresponds to ~6 Pa @ 1 m

    def __format__(self, fmt_spec):
        return fmt_spec.replace('%cls', self.__class__.__name__).replace('%freq', str(self.freq)).replace('%p0', str(self.p0)).replace('%mediumfull', repr(self.medium)).replace('%medium', str(self.medium))

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

    def greens_function(self, source_position, source_normal, receiver_positions):
        """Evaluate the transducer radiation.

        This function needs to be implemented by concrete subclasses.

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
        raise NotImplementedError('Transducer model of type `{}` has not implemented a greens function'.format(self.__class__.__name__))

    def pressure_derivs(self, source_position, source_normal, receiver_positions, orders=3, **kwargs):
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

    def greens_function(self, source_positions, source_normals, receiver_positions):
        """Evaluate the transducer radiation.

        This is a combination of spherically spreading waves, a directivity
        function, and a source strength.

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
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')
        return self.p0 * self.wavefront_spreading(source_positions, receiver_positions) * self.directivity(source_positions, source_normals, receiver_positions)

    def wavefront_spreading(self, source_positions, receiver_positions):
        """Evaluate spherical wavefronts.

        Parameters
        ----------
        source_positions : numpy.ndarray
            The location of the transducer, as a (3, ...) shape array.
        receiver_positions : numpy.ndarray
            The location(s) at which to evaluate the radiation, shape (3, ...).
            The first dimension must have length 3 and represent the coordinates of the points.

        Returns
        -------
        out : numpy.ndarray
            The amplitude and phase of the wavefront, shape `source_positions.shape[1:] + receiver_positions.shape[1:]`.
            Assuming 1Pa at 1m distance, phase referenced to the transducer center.

        """
        if receiver_positions.shape[0] != 3:
            raise ValueError('Incorrect shape of positions')
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        distance = np.einsum('i...,i...', diff, diff)**0.5
        return np.exp(1j * self.k * distance) / distance

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
        return np.ones(source_positions.shape[1:2] + receiver_positions.shape[1:])

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
    plane_distance : float
        The distance between the array and the reflector, along the normal.
    plane_normal : array_like, default (0,0,1)
        3 element vector with the plane normal.
    reflection_coefficient : complex float, default 1
        Reflection coefficient to tune the magnitude and phase of the reflection.

    Returns
    -------
    transducer
        The transducer model with reflections.

    """

    _repr_fmt_spec = '{:%cls(transducer=%transducer_full, plane_distance=%plane_distance, plane_normal=%plane_normal, reflection_coefficient=%reflection_coefficient)}'
    _str_fmt_spec = '{:%cls(transducer=%transducer, plane_distance=%plane_distance, plane_normal=%plane_normal, reflection_coefficient=%reflection_coefficient)}'

    def __init__(self, transducer, plane_distance, plane_normal=(0, 0, 1), reflection_coefficient=1, *args, **kwargs):
        if type(transducer) is type:
            transducer = transducer(*args, **kwargs)
        self._transducer = transducer
        self.plane_distance = plane_distance
        self.plane_normal = np.asarray(plane_normal, dtype='float64')
        self.plane_normal /= (self.plane_normal**2).sum()**0.5
        self.reflection_coefficient = reflection_coefficient

    def __format__(self, fmt_str):
        s_out = fmt_str.replace('%transducer_full', repr(self._transducer)).replace('%transducer', str(self._transducer))
        s_out = s_out.replace('%plane_distance', str(self.plane_distance)).replace('%plane_normal', str(tuple(self.plane_normal)))
        s_out = s_out.replace('%reflection_coefficient', str(self.reflection_coefficient))
        return super().__format__(s_out)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self._transducer == other._transducer
            and np.allclose(self.plane_distance, other.plane_distance)
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

    def greens_function(self, source_positions, source_normals, receiver_positions):
        r"""Evaluate the pressure at a point.

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
        plane_normal = self.plane_normal.reshape((3,) + (1,) * (source_positions.ndim - 1))
        direct = self._transducer.greens_function(source_positions, source_normals, receiver_positions)
        mirror_position = source_positions - 2 * plane_normal * ((source_positions * plane_normal).sum(axis=0) - self.plane_distance)
        mirror_normal = source_normals - 2 * plane_normal * (source_normals * plane_normal).sum(axis=0)
        reflected = self._transducer.greens_function(mirror_position, mirror_normal, receiver_positions)
        return direct + self.reflection_coefficient * reflected

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
            Array with the calculated derivatives. Has the shape `(M,) + source_positions.shape[1:] + receiver_positions.shape[1:]`.
            where `M` is the number of spatial derivatives, see `num_spatial_derivatives` and `spatial_derivative_order`.

        """
        plane_normal = self.plane_normal.reshape((3,) + (1,) * (source_positions.ndim - 1))
        direct = self._transducer.pressure_derivs(source_positions, source_normals, receiver_positions, orders, **kwargs)
        mirror_position = source_positions - 2 * plane_normal * ((source_positions * plane_normal).sum(axis=0) - self.plane_distance)
        mirror_normal = source_normals - 2 * plane_normal * (source_normals * plane_normal).sum(axis=0)
        reflected = self._transducer.pressure_derivs(mirror_position, mirror_normal, receiver_positions, orders, **kwargs)
        return direct + self.reflection_coefficient * reflected


class PlaneWaveTransducer(TransducerModel):
    """Class representing planar waves.

    This is not representing a physical transducer per se, but a traveling
    plane wave.
    """

    def greens_function(self, source_positions, source_normals, receiver_positions):
        r"""Evaluate the pressure at a point.

        The equation is that of a plane wave, :math:`G(\vec x) = p_0 \exp(i\vec k \cdot \vec x)`.

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
        source_normals = np.asarray(source_normals, dtype=np.float64)
        source_normals /= (source_normals**2).sum(axis=0)**0.5
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        x_dot_n = np.einsum('i..., i...', diff, source_normals)
        return self.p0 * np.exp(1j * self.k * x_dot_n)

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
        source_normals = np.asarray(source_normals, dtype=np.float64)
        source_normals /= (source_normals**2).sum(axis=0)**0.5
        derivatives = np.empty((utils.num_pressure_derivs[orders],) + source_positions.shape[1:2] + receiver_positions.shape[1:], dtype=np.complex128)
        derivatives[0] = self.greens_function(source_positions, source_normals, receiver_positions)
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
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dots = np.einsum('i...,i...', diff, source_normals)
        norm1 = np.einsum('i...,i...', source_normals, source_normals)**0.5
        norm2 = np.einsum('i...,i...', diff, diff)**0.5
        cos_angle = dots / norm2 / norm1
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
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dots = np.einsum('i...,i...', diff, source_normals)
        norm1 = np.einsum('i...,i...', source_normals, source_normals)**0.5
        norm2 = np.einsum('i...,i...', diff, diff)**0.5
        cos_angle = dots / norm2 / norm1
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
        source_normals = source_normals.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        diff = receiver_positions.reshape((3,) + (1,) * (source_positions.ndim - 1) + receiver_positions.shape[1:]) - source_positions.reshape(source_positions.shape[:2] + (receiver_positions.ndim - 1) * (1,))
        dot = np.einsum('i...,i...', diff, source_normals)
        # r = np.einsum('...i,...i', diff, diff)**0.5
        r = np.sum(diff**2, axis=0)**0.5
        n = source_normals
        norm = np.einsum('i...,i...', n, n)**0.5
        cos = dot / r / norm
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
