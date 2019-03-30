import logging

logger = logging.getLogger(__name__)

__all__ = ['transducers', 'arrays', 'hardware', 'materials', 'optimization', 'algorithms']
__version__ = '1.1.0.dev'

pressure_derivs_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy']
num_pressure_derivs = [1, 4, 10, 19]


class _SphericalHarminicsIndexer:
    def __call__(self, order, mode):
        if abs(mode) > order:
            raise ValueError('Spherical harmonics mode cannot be higher than the order')
        return int(order**2 + order + mode)

    def __getitem__(self, index):
        if type(index) == slice:
            return self.__iter__(index.start, index.stop, index.step)
        order = (index**0.5) // 1
        mode = index - order**2 - order
        return int(order), int(mode)

    def __iter__(self, start=None, stop=None, step=None):
        index = 0 if start is None else start
        stop = float('inf') if stop is None else stop
        step = 1 if step is None else step
        while index < stop:
            yield self[index]
            index += step

    def orders(self, start=None, stop=None):
        start = 0 if start is None else start**2
        stop = float('inf') if stop is None else (stop + 1)**2
        return self.__iter__(start, stop)


spherical_harmonics_index = _SphericalHarminicsIndexer()
del _SphericalHarminicsIndexer

from . import *
