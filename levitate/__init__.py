import logging

logger = logging.getLogger(__name__)

__all__ = ['transducers', 'arrays', 'hardware', 'materials', 'optimization', 'algorithms']
__version__ = '1.1.0.dev'

pressure_derivs_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy']
num_pressure_derivs = [1, 4, 10, 19]

from . import *
