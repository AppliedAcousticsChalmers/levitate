"""A collection of levitation related mathematical implementations.

The fields is one of the most important parts of the package,
containing implementations of various ways to calculate levitate-related physical properties.
To simplify the management and manipulation of the implemented fields they are wrapped
in an additional abstraction layer.
The short version is that the classes implemented in the `~levitate.fields` module
will not return objects of the called class, but typically objects of `~levitate.field_wrappers.Field`.
These objects support algebraic operations, like `+`, `*`, and `abs`. The full description of
what the different operands do can be found in the documentation of `~levitate._field_wrappers`.

References
----------
.. [Gorkov] L. P. Gorkov, “On the Forces Acting on a Small Particle in an Acoustical Field in an Ideal Fluid”
            Soviet Physics Doklady, vol. 6, p. 773, Mar. 1962.

.. [Sapozhnikov] O. A. Sapozhnikov and M. R. Bailey, “Radiation force of an arbitrary acoustic beam on an elastic sphere in a fluid”
                 J Acoust Soc Am, vol. 133, no. 2, pp. 661–676, Feb. 2013.

"""


from ._implementations import *  # noqa: F401, F403
from ._wrappers import stack  # noqa: F401


def sum(*fields):
    if len(fields) == 1:
        try:
            # A single input which is some type of field should be summed.
            return fields[0].sum()
        except AttributeError:
            # If the input does not have a sum method it's probably an iterable of fields, which is handled by stack
            pass
    # Unpack the fields, stack them, and sum the stack.
    return stack(*fields).sum()


def sum_of_eigenvalues(field):
    from ._transformers import EigenvalueSum
    return field.copy()._append_transform(EigenvalueSum)


def softplus(field):
    from ._transformers import Softplus
    return field.copy()._append_transform(Softplus)


def exp(field):
    import numpy
    return numpy.e ** field


def real(field):
    from ._transformers import Real
    return field.copy()._append_transform(Real)


def imag(field):
    from ._transformers import Imag
    return field.copy()._append_transform(Imag)


def conjugate(field):
    from ._transformers import Conjugate
    return field.copy()._append_transform(Conjugate)
