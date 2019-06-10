"""Miscellaneous tools for small but common tasks."""

pressure_derivs_order = ['', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'xxx', 'yyy', 'zzz', 'xxy', 'xxz', 'yyx', 'yyz', 'zzx', 'zzy', 'xyz']
"""Defines the order in which the pressure spatial derivatives are stored."""
num_pressure_derivs = [1, 4, 10, 20]
"""Quick access to the number of spatial derivatives up to and including a certain order."""
