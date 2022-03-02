"""Tools for indexing of spatial derivatives and spherical harmonics."""
import numpy as np

pressure_derivs_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy', 'xyz']
"""Defines the order in which the pressure spatial derivatives are stored."""
num_pressure_derivs = [1, 4, 10, 20]
"""Quick access to the number of spatial derivatives up to and including a certain order."""


class SphericalHarmonicsIndexer:
    """
    Helper class to index spherical harmonics.

    This class is designed to assist in converting between index and
    order/mode form of spherical harmonics. All implementations in this
    package should follow the conventions given by this class.

    There are two main ways to convert between the forms, implemented by indexing or calling
    the objects. Indexing with an index gives the corresponding tuple `(order, mode)`,
    while calling the object with such a tuple gives the corresponding index.
    The objects are iterable, returning the `(order, mode)` from the start order
    to the maximum order.

    Parameters
    ----------
    max_order : int
        The maximum order to iterate to.
    min_order : int
        The order to start iterations from, default 0

    Note
    ----
    The objects will not prevent access to indices and orders beyond the max and min orders
    used to create the object.

    Examples
    --------
    To loop over the orders and modes in a single for loop::

        >>> for n, m in SphericalHarmonicsIndexer(2):
        >>>     print((n, m))
        (0, 0)
        (1, -1)
        (1, 0)
        (1, 1)
        (2, -2)
        (2, -1)
        (2, 0)
        (2, 1)
        (2, 2)

    To loop over orders and modes in nested loops::

        >>> sph_idx = SphericalHarmonicsIndexer(3)
        >>> for n in sph_idx.orders:
        >>>     print(n, end=': ')
        >>>     for m in sph_idx.modes:
        >>>         print(m, end=' ')
        >>>     print()
        0: 0
        1: -1 0 1
        2: -2 -1 0 1 2
        3: -3 -2 -1 0 1 2 3

    The `min_order` attribute will shift the indexing::

        >>> sph_idx = SphericalHarmonicsIndexer(2, 3)
        >>> print(sph_idx[0])
        (2, -2)
        >>> print(sph_idx(0, 0))
        -4

    To get all orders and all modes is separated variables,
    simply zip the object::

        >>> n, m = zip(*SphericalHarmonicsIndexer(2))
        >>> print(n)
        >>> print(m)
        (0, 1, 1, 1, 2, 2, 2, 2, 2)
        (0, -1, 0, 1, -2, -1, 0, 1, 2)

    """

    def __init__(self, max_order=None, min_order=0):
        if max_order is None:
            max_order = float('inf')
        if max_order < min_order:
            # Called as `SphericalHarmonicsIndexer(min_order, max_order)`.
            # Switch the arguments.
            tmp = min_order
            min_order = max_order
            max_order = tmp
        self.max_order = max_order
        self.min_order = min_order

    def __call__(self, order, mode):
        if abs(mode) > order:
            raise ValueError('Spherical harmonics mode cannot be higher than the order')
        try:
            return int(order**2 + order + mode - self._min_offset)
        except OverflowError:
            return float('inf')

    def __getitem__(self, index):
        if type(index) == slice:
            return self.__iter__(index.start, index.stop, index.step)
        index = index + self._min_offset
        order = (index**0.5) // 1
        mode = index - order**2 - order
        return int(order), int(mode)

    def __iter__(self, start=None, stop=None, step=None):
        # index = 0 if start is None else start
        # stop = float('inf') if stop is None else stop
        index = self(self.min_order, -self.min_order) if start is None else start
        stop = self(self.max_order, self.max_order) if stop is None else stop
        step = 1 if step is None else step
        while index * step <= stop * step:
            yield self[index]
            index += step

    def __len__(self):
        max_idx = self(self.max_order, self.max_order)
        min_idx = self(self.min_order, -self.min_order)
        return max_idx - min_idx + 1

    def __reversed__(self):
        start = self(self.min_order, -self.min_order)
        stop = self(self.max_order, self.max_order)
        return self.__iter__(stop, start, -1)

    @property
    def min_order(self):
        return self._min_order

    @min_order.setter
    def min_order(self, val):
        self._min_order = val
        self._min_offset = int((self.min_order - 1)**2 + 2 * self.min_order - 1)

    @property
    def orders(self):
        """Iterate over orders.

        Iterator to iterate over the orders in the indexer.
        Will enable the synchronized `modes` iterator.
        """
        self._current_order = self.min_order
        while self._current_order <= self.max_order:
            yield self._current_order
            self._current_order += 1
        del self._current_order

    @property
    def modes(self):
        """Iterate over modes.

        Synchronized iterator to iterate the modes in an order.
        """
        try:
            order = self._current_order
        except AttributeError:
            raise AttributeError('Cannot iterate over modes without iterating over orders!') from None
        mode = -order
        while mode <= order:
            yield mode
            mode += 1

    def ordersum(self, values, axis=None):
        """Sum spherical harmonics coefficients of the same order.

        Calculates the sum of the coefficients for all modes for each order
        individually. The `SphericalHarmonicsIndexer` needs to be created to
        match the orders of the expansion coefficients. This requires that the
        length of the summation axis is the same as the number of coefficients
        for the orders specified, i.e. `values.shape[axis] == len(self)`.
        If no axis is specified, the first suitable axis will be used.

        Parameters
        ----------
        values : numpy.ndarray
            The spherical harmonics expansion coefficients of a field.
        axis : int, optional
            The axis over which to sum.

        Returns
        -------
        order_summed_coefs : numpy.ndarray
            The summed coefficients for each order.

        """
        values = np.asarray(values)
        if axis is None:
            for axis in range(values.ndim):
                if values.shape[axis] == len(self):
                    break
            else:
                raise ValueError('Cannot find axis of length {} in the given values!'.format(len(self)))

        values = np.moveaxis(values, axis, 0)
        output = np.zeros((self.max_order - self.min_order + 1, ) + values.shape[1:], dtype=values.dtype)
        for idx, order in enumerate(self.orders):
            output[idx] = np.sum(values[self(order, -order):self(order, order) + 1], axis=0)
        return np.moveaxis(output, 0, axis)
