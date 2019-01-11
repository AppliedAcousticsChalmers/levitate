import os
import numpy as np

from ._dragonfly import dragonfly_grid
from ._TCPArray import TCPArray


def data_to_cpp(complex_values, filename, normalize=True):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.asarray(complex_values)
    normalization = np.max(np.abs(data))
    (data / normalization).conj().astype(np.complex64).tofile(filename)


def data_from_cpp(file, num_transducers):
    return np.fromfile(file, dtype=np.complex64).conj().reshape((-1, num_transducers))
