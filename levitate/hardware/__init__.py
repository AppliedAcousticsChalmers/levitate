"""Hardware related classes and functions.

Various classes and functions to interface with Ultrahaptics
hardware.

.. admonition:: Disclaimer

    This module is not related to Ultrahaptics as a company, and it not part of their
    SDK. It is simply a non-programmer's way around testing everything in C++.

    **Use at your own risk!**

.. autosummary::
    :nosignatures:

    TCPArray
    data_to_cpp
    data_from_cpp
"""

import os
import numpy as np

from ._dragonfly import dragonfly_grid
from ._TCPArray import TCPArray


def data_to_cpp(complex_values, filename):
    """Write data to a file suitable for c++.

    Takes numpy data and writes it to a file which is simple to read from c++.
    Internally uses `numpy.tofile` for the actual write.

    Parameters
    ----------
    complex_values : numpy.ndarray
        The complex transducer values for the array.
        The order of the transducers must match the internal order of the array.
    filename : string
        The filename of the file to create.

    Note
    ----
    - The data will be normalized to have a maximum amplitude of 1.
    - The data will be written as 64 bit complex floats, i.e. 32 bit real + 32 bit imaginary.
    - The data will be conjugated: Ultrahaptics uses a different phase convention.

    """
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.asarray(complex_values)
    normalization = np.max(np.abs(data))
    (data / normalization).conj().astype(np.complex64).tofile(filename)


def data_from_cpp(file, num_transducers):
    """Read data previously written for c++.

    This is the inverse of `data_to_cpp`, and is used to read data
    previously written with said function, or with similar conventions.

    Parameters
    ----------
    file : String or open file
        The file to read. See `numpy.fromfile`.
    num_transducers : int
        The number of transducers in the array. This is important to be able to reshape
        the data and have the correct number of states in the output.

    Returns
    -------
    data : numpy.ndarray
        The data read from the file. Shape `(M, N)` where `M` is the number
        of states in the file, and `N` is the number of transducers specified.

    Note
    ----
    - The data is assumed to be 64 bit complex floats, i.e. 32 bit real + 32 bit imaginary.
    - The data will be conjugated: Ultrahaptics uses a different phase convention.

    """
    return np.fromfile(file, dtype=np.complex64).conj().reshape((-1, num_transducers))
