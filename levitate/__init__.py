import logging

logger = logging.getLogger(__name__)

__all__ = ['transducers', 'arrays', 'cost_functions', 'hardware', 'materials']
__version__ = '0.3.3dev'

spatial_derivative_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy']
num_spatial_derivatives = [1, 4, 10, 19]

from . import *
