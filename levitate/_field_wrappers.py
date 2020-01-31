"""Implementation of field wrapper protocol.

The API for the implemented fields consists of two parts:
the actual implementation of the fields in `FieldImplemention`
objects, and the wrapper classes `FieldBase` and its subclasses.
When objects of the `FieldImplementation` type is instantiated
they will automagically be wrapped inside a different object, which is
returned to the caller.
The wrapper objects are of different variants, corresponding to the use case.

Note that it is not intended that a user manually creates any of these objects,
since there are many pit-falls when choosing the correct type. Instead call the
implemented fields to get a basic field type and manipulate it using
the arithmetic API to create the desired functionality.
To validate that an API arithmetic manipulation actually does what was intended,
simply print the resulting object to inspect the new structure.

Basic Types
-----------
The basic type is a simple `Field`, which is the default return type from
instantiating. When called with transducer complex amplitudes and a set of positions,
the object evaluates the field implementation with the required parameters and
returns just the value from the field.

If the field is bound by using the `@` operation a new `FieldPoint`
object is created, and the position is implicit in the call.
This is more efficient for repeated calling with the same position,
since some parts of the calculation can be cached.

If the field is weighed with the `*` operation a new `CostField`
object is created, and the return from a call is changed. Cost field type
objects will return the weighed sum of the different parts of the implemented
field, as well as the jacobians of said value with respect to the transducers.
The jacobians are returned if a form that allow for simple calculation of the
jacobians with respect to transducer real part, imaginary part, amplitude, or phase.

If a field is both bound and weighted it is a `CostFieldPoint`, created either
by binding a `CostField` or by weighting a `FieldPoint`.
This will have the same caching and call signature as a `FieldPoint`, but
the same return values as an `CostField`. This form is the most suitable
for numerical optimizations.

.. autosummary::
    :nosignatures:

    Field
    FieldPoint
    CostField
    CostFieldPoint

Squared Types
-------------
Each of the above types can be used to change the values (and jacobians)
to calculate the squared magnitude of the values, possible with a static
target shift applied before taking the magnitude.
There are four types, `SquaredField`, `SquaredFieldPoint`,
`SquaredCostField`, and `SquaredCostFieldPoint`, each
corresponding to one of the basic types.
They are created by taking the absolute value of a basic object, or by subtracting
a fixed value from a basic object. Note that the square is not apparent from the API,
which is less intuitive that the other parts in the API.
In all other regards, the squared versions behave like their basic counterparts.

.. autosummary::
    :nosignatures:

    SquaredField
    SquaredFieldPoint
    SquaredCostField
    SquaredCostFieldPoint

MultiField
----------
MultiFields are objects which collect basic fields operating at the same point(s) in space.
Two objects of the same basic type (or a squared version of said basic type)
can be added together. If the fields are either unbound or bound to the same
position, a multi-field-type object is created. Again, there are four types, each corresponding
to one of the basic types: `MultiField`, `MultiFieldPoint`, `MultiCostField`,
and `MultiCostFieldPoint`.
The two field-style variants will evaluate all included fields and return the individual
values from the included fields. The two cost-field-style variants will sum the values
from the included cost fields.

.. autosummary::
    :nosignatures:

    MultiField
    MultiFieldPoint
    MultiCostField
    MultiCostFieldPoint

MultiPoint
----------
MultiPoints are objects which collect multi-field-type objects bound to different points.
There are only two types: `MultiFieldMultiPoint` similar to a `MutiFieldPoint`,
and `MultiCostFieldMultiPoint` similar to a `MultiCostFieldPoint`. It is not possible to
have unbound multi-points, they would simply be unbound.
A `MultiFieldMultiPoint` returns the values from the stored fields.
A `MultiCostFieldMultiPoint` will sum the values and jacobians of the stored objects.

.. autosummary::
    :nosignatures:

    MultiFieldMultiPoint
    MultiCostFieldMultiPoint

Implementation Details
----------------------
To make the API work as intended, there are a couple additional
classes and functions.
The base class for the implemented fields, `FieldImplementation` is
only used as a super class when implementing new physical fields. See its documentation
for more details on how to extend the package with new field implementations.

The wrapping of `FieldImplementation` inside `Field` objects is implemented
in the `FieldImplementationMeta` class, in the `~FieldImplementationMeta.__call__` method.
This also accepts additional `weight` and `position` parameters to directly create
the other three basic types, instead of the default `Field`.

`FieldBase` is the top-level class for all wrappers, and handles evaluation of
spatial structures, caching, etc.
Similarly there is a `SquaredFieldBase`, which is wrapping the base `Field` objects'
calculation functions with the magnitude and square.
"""

import numpy as np
import collections


class FieldImplementationMeta(type):
    """Metaclass to wrap `FieldImplementation` objects in `Field` objects.

    API-wise it is nice to call the implementation classes when requesting a field.
    Since the behavior of the objects should change depending on if they are added etc,
    it would be very difficult to keep track of both the current state and the actual field
    in the same top level object. This class will upon object creation instantiate the called class,
    but also instantiate and return a `Field`-type object.
    """

    def __call__(cls, *cls_args, weight=None, position=None, **cls_kwargs):
        """Instantiate an `Field`-type object, using the `cls` as the base field implementation.

        The actual `Field`-type will be chosen based on which optional parameters are passed.
        If no parameters are passed (default) a `Field` object is returned.
        If `weight` is passed a `CostField` object is returned.
        If `position` is passed a `FieldPoint` object is returned.
        If both `weight` and `position` is passed a `CostFieldPoint` object is returned.

        Parameters
        ----------
        cls : class
            The `FieldImplementation` class to use for calculations.
        *cls_args :
            Args passed to the `cls`.
        weight : numeric
            Optional weight.
        position : numpy.ndarray
            Optional array to bind the field to, shape (3,...).
        **cls_kwargs :
            Keyword arguments passed to `cls`.

        """
        obj = cls.__new__(cls, *cls_args, **cls_kwargs)
        obj.__init__(*cls_args, **cls_kwargs)
        if weight is None and position is None:
            alg = Field(field=obj)
        elif weight is None:
            alg = FieldPoint(field=obj, position=position)
        elif position is None:
            alg = CostField(field=obj, weight=weight)
        elif weight is not None and position is not None:
            alg = CostFieldPoint(field=obj, weight=weight, position=position)
        return alg


class FieldImplementation(metaclass=FieldImplementationMeta):
    """Base class for FieldImplementations.

    The attributes listed below are part of the API and should be
    implemented in subclasses.

    Parameters
    ----------
    array : TransducerArray
        The array object to use for calculations.

    Attributes
    ----------
    values_require : dict
        Each key in this dictionary specifies a requirement for
        the `values` method. The wrapper classes will manage
        calling the method with the specified arguments.
    jacobians_require : dict
        Each key in this dictionary specifies a requirement for
        the `jacobians` method. The wrapper classes will manage
        calling the method with the specified arguments.

    Methods
    -------
    values
        Method to calculate the value(s) for the field.
    jacobians
        Method to calculate the jacobians for the field.
        This method is optional if the implementation is not used
        as a cost function in optimizations.

    """

    def __init__(self, array, *args, **kwargs):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.

        """
        self.array = array

    def __eq__(self, other):
        return type(self) == type(other) and self.array == other.array

    class requirement(collections.UserDict):
        """Parse a set of requirements.

        `FieldImplementation` objects should define requirements for values and jacobians.
        This class parses the requirements and checks that the request can be met upon call.
        The requirements are stored as a non-mutable custom dictionary.
        Requirements can be added to each other to find the combined requirements.
        """

        possible_requirements = [
            'complex_transducer_amplitudes',
            'pressure_derivs_summed', 'pressure_derivs_individual',
            'spherical_harmonics_summed', 'spherical_harmonics_individual',
            'spherical_harmonics_gradient_summed', 'spherical_harmonics_gradient_individual',
        ]

        def __setitem__(self, key, value):
            if self.locked:
                raise TypeError("`Requirement` instances should not be mutated!")
            super().__setitem__(key, value)

        def __init__(self, *args, **kwargs):  # noqa: D205, D400
            """
            Keyword arguments
            ---------------------
            complex_transducer_amplitudes
                The field requires the actual complex transducer amplitudes directly.
                This is a fallback requirement when it is not possible to implement the field
                with the other requirements, and no performance optimization is possible.
            pressure_derivs_summed
                The number of orders of Cartesian spatial derivatives of the total sound pressure field.
                Currently implemented to third order derivatives.
                See `levitate.utils.pressure_derivs_order` and `levitate.utils.num_pressure_derivs`
                for a description of the structure.
            pressure_derivs_summed
                Like pressure_derivs_summed, but for individual transducers.
            spherical_harmonics_summed
                A spherical harmonics decomposition of the total sound pressure field, up to and
                including the order specified.
                where remaining dimensions are determined by the positions.
            spherical_harmonics_individual
                Like spherical_harmonics_summed, but for individual transducers.

            Raises
            ------
            NotImplementedError
                If one or more of the requested keys is not implemented.

            """
            self.locked = False
            super().__init__(*args, **kwargs)
            self.locked = True
            for requirement in self:
                if requirement not in self.possible_requirements:
                    raise NotImplementedError("Requirement '{}' is not implemented for a field. The possible requests are: {}".format(requirement, self.possible_requirements))

        def __add__(self, other):
            if not isinstance(other, (dict, collections.UserDict)):
                return NotImplemented
            unique_self = {key: self[key] for key in self.keys() - other.keys()}
            unique_other = {key: other[key] for key in other.keys() - self.keys()}
            max_common = {key: max(self[key], other[key]) for key in self.keys() & other.keys()}
            return type(self)(**unique_self, **unique_other, **max_common)


class FieldMeta(type):
    """Metaclass for `Field`-type objects.

    This metaclass is only needed to make the `_type` property available
    at both class and instance level.
    """

    @property
    def _type(cls):  # noqa: D401
        """The type of the field.

        In this context `type` refers for the combination of `bound` and `cost`.
        """
        return cls._is_bound, cls._is_cost


class FieldBase(metaclass=FieldMeta):
    """Base class for all field type objects.

    This wraps a few common procedures for fields,
    primarily dealing with preparation and evaluation of requirements
    for fields implementations.
    The fields support some numeric manipulations to simplify
    the creation of variants of the basic types.
    Not all types of fields support all operations, and the order of
    operation can matter in some cases.
    If unsure if the arithmetics return the desired outcome, print the
    resulting object to inspect the new structure.

    Note
    ----
    This class should not be instantiated directly.
    """

    def evaluate_requirements(self, complex_transducer_amplitudes, position=None):
        """Evaluate requirements for given complex transducer amplitudes.

        Parameters
        ----------
        complex_transducer_amplitudes: complex ndarray
            The transducer phase and amplitude on complex form,
            must correspond to the same array used to create the field.
        position: ndarray
            The position where to calculate the requirements needed.
            Shape (3,...). If position is `None` or not passed, it is assumed
            that the field is bound to a position and `self.position` will be used.

        Returns
        -------
        requirements : dict
            Has (at least) the same fields as `self.requires`, but instead of values specifying the level
            of the requirement, this dict has the evaluated requirement at the positions and
            transducer amplitudes specified.

        Note
        ----
        Fields which are bound to a position will cache the array requests, i.e. the requirements
        without any transducer amplitudes applied. It is therefore important to not manually change
        the position, since that will not clear the cache and the new position is not actually used.

        """
        if position is None:
            try:
                evaluated_requests = self._cached_requests
            except AttributeError:
                evaluated_requests = self._cached_requests = self.array.request(self.requires, self.position)
        else:
            evaluated_requests = self.array.request(self.requires, position)

        complex_transducer_amplitudes = np.asarray(complex_transducer_amplitudes)
        # Apply the input complex amplitudes
        evaluated_requrements = {}
        if 'complex_transducer_amplitudes' in self.requires:
            evaluated_requrements['complex_transducer_amplitudes'] = complex_transducer_amplitudes
        if 'pressure_derivs' in evaluated_requests:
            evaluated_requrements['pressure_derivs_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, evaluated_requests['pressure_derivs'])
            evaluated_requrements['pressure_derivs_summed'] = np.sum(evaluated_requrements['pressure_derivs_individual'], axis=1)
        if 'spherical_harmonics' in evaluated_requests:
            evaluated_requrements['spherical_harmonics_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, evaluated_requests['spherical_harmonics'])
            evaluated_requrements['spherical_harmonics_summed'] = np.sum(evaluated_requrements['spherical_harmonics_individual'], axis=1)
        if 'spherical_harmonics_gradient' in evaluated_requests:
            evaluated_requrements['spherical_harmonics_gradient_individual'] = np.einsum('i,jki...->jki...', complex_transducer_amplitudes, evaluated_requests['spherical_harmonics_gradient'])
            evaluated_requrements['spherical_harmonics_gradient_summed'] = np.sum(evaluated_requrements['spherical_harmonics_gradient_individual'], axis=2)
        return evaluated_requrements

    def _clear_cache(self):
        try:
            del self._cached_requests
        except AttributeError:
            pass

    @property
    def _type(self):  # noqa: D401
        """The type of the field.

        In this context `type` refers for the combination of `bound` and `cost`.
        """
        return type(self)._type

    def __eq__(self, other):
        return type(self) == type(other)

    def __abs__(self):
        return self - 0

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, weight):
        return self.__mul__(weight)

    def __str__(self, not_api_call=True):
        return self._str_format_spec.format(self)

    def __format__(self, format_spec):
        cls = self.__class__.__name__ + ': '
        weight = getattr(self, 'weight', None)
        pos = getattr(self, 'position', None)
        weight = ' * ' + str(weight) if weight is not None else ''
        pos = ' @ ' + str(pos) if pos is not None else ''
        return format_spec.replace('%cls', cls).replace('%weight', weight).replace('%position', pos)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Field(FieldBase):
    """Primary class for single point, single field.

    This is a wrapper class for `FieldImplementation` to simplify the manipulation
    and evaluation of the implemented fields. Normally it is not necessary to manually
    create the wrapper, since it should be done automagically.
    Many properties are inherited from the underlying field implementation, e.g.
    `ndim`, `array`, `values`, `jacobians`.

    Parameters
    ----------
    field : `FieldImplementation`
        The implemented field to use for calculations.

    Methods
    -------
    +
        Adds this field with another `Field` or `FieldPoint`.

        :return: `FieldPoint`.
    *
        Weight the field with a suitable weight.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `CostField`
    @
        Bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `FieldPoint`
    -
        Converts to a squared magnitude target field.

        :return: `SquaredField`

    """

    _str_format_spec = '{:%cls%name}'
    _is_bound = False
    _is_cost = False

    def __init__(self, field):
        self.field = field
        value_indices = ''.join(chr(ord('i') + idx) for idx in range(self.ndim))
        self._sum_str = value_indices + ', ' + value_indices + '...'
        self.requires = self.values_require

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.field == other.field
            and self.array == other.array
        )

    @property
    def name(self):
        return self.field.__class__.__name__

    @property
    def values(self):
        return self.field.values

    @property
    def jacobians(self):
        return self.field.jacobians

    @property
    def values_require(self):
        return self.field.values_require

    @property
    def jacobians_require(self):
        return self.field.jacobians_require

    @property
    def ndim(self):
        return self.field.ndim

    @property
    def array(self):
        return self.field.array

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate the field implementation.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.
        position : array-like
            The position(s) where to evaluate the field.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: ndarray
            The values of the implemented field used to create the wrapper.

        """
        # Prepare the requirements dict
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, position)
        # Call the function with the correct arguments
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return MultiField(self, other)
        else:
            return NotImplemented

    def __sub__(self, target):
        return SquaredField(field=self, target=target)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostField(weight=weight, field=self.field)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return FieldPoint(position=position, field=self.field)

    def __format__(self, format_spec):
        name = getattr(self, 'name', None) or 'Unknown'
        return super().__format__(format_spec.replace('%name', name))


class FieldPoint(Field):
    """Position-bound class for single point, single field.

    See `Field` for more precise description.

    Parameters
    ----------
    field : FieldImplementation
        The implemented field to use for calculations.
    position : numpy.ndarray
        The position to bind to.

    Methods
    -------
    +
        Adds this field with another `FieldPoint`,
        `MultiFieldPoint`, or `MultiFieldMultiPoint`.

        :return: `MultiFieldPoint` or `MultiFieldMultiPoint`.
    *
        Weight the field with a suitable weight.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `CostFieldPoint`
    @
        Re-bind the field to a new point in space. The point needs to have
        3 elements in the first dimension.

        :return: `FieldPoint`
    -
        Converts to a magnitude target field.

        :return: `SquaredFieldPoint`

    """

    _str_format_spec = '{:%cls%name%position}'
    _is_bound = True
    _is_cost = False

    def __init__(self, field, position, **kwargs):
        super().__init__(field=field, **kwargs)
        self.position = np.asarray(position)

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.position, other.position)
        )

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate the field implementation.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: ndarray
            The values of the implemented field used to create the wrapper.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes)
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return MultiFieldPoint(self, other)
            else:
                return MultiFieldMultiPoint(self, other)
        else:
            return NotImplemented

    def __sub__(self, target):
        return SquaredFieldPoint(field=self, target=target, position=self.position)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostFieldPoint(weight=weight, position=self.position, field=self.field)


class CostField(Field):
    """Unbound cost field for single point, single field.

    See `Field` for more precise description.

    Parameters
    ----------
    field : FieldImplementation
        The implemented field to use for calculations.
    weight : numpy.ndarray
        The weight to use for the summation of values. Needs to have the same
        number of dimensions as the `FieldImplementation` used.

    Methods
    -------
    +
        Adds this field with another `CostField` or `MultiCostField`.

        :return: `MultiCostField`.
    *
        Rescale the weight, i.e. multiplies the current weight with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `CostField`
    @
        Bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `CostFieldPoint`
    -
        Converts to a squared magnitude target field.

        :return: `SquaredCostField`

    """

    _str_format_spec = '{:%cls%name%weight}'
    _is_bound = False
    _is_cost = True

    def __init__(self, field, weight, **kwargs):
        super().__init__(field=field, **kwargs)
        self.weight = np.asarray(weight)
        if self.weight.ndim < self.ndim:
            extra_dims = self.ndim - self.weight.ndim
            self.weight.shape = (1,) * extra_dims + self.weight.shape
        self.requires = self.values_require + self.jacobians_require

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.weight, other.weight)
        )

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate the field implementation.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.
        position : array-like
            The position(s) where to evaluate the field.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: ndarray
            The values of the implemented fiield used to create the wrapper.
        jacobians : ndarray
            The jacobians of the values with respect to the transducers.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, position)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return MultiCostField(self, other)
        else:
            return NotImplemented

    def __sub__(self, target):
        return SquaredCostField(field=self, target=target, weight=self.weight)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostField(field=self.field, weight=self.weight * weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return CostFieldPoint(weight=self.weight, position=position, field=self.field)


class CostFieldPoint(CostField, FieldPoint):
    """Cost function for single point, single fields.

    See `Field` for more precise description.

    Parameters
    ----------
    field : FieldImplementation
        The implemented field to use for calculations.
    weight : numpy.ndarray
        The weight to use for the summation of values. Needs to have the same
        number of dimensions as the `FieldImplementation` used.
    position : numpy.ndarray
        The position to bind to.

    Methods
    -------
    +
        Adds this field with another `CostFieldPoint`,
        `MultiCostFieldPoint`, or `MultiCostFieldMultiPoint`.

        :return: `MultiCostFieldPoint`,or `MultiCostFieldMultiPoint`
    *
        Rescale the weight, i.e. multiplies the current weight with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `CostFieldPoint`
    @
        Re-bind the field to a new point in space. The point needs to have
        3 elements in the first dimension.

        :return: `CostFieldPoint`
    -
        Converts to a magnitude target field.

        :return: `SquaredCostFieldPoint`

    """

    _str_format_spec = '{:%cls%name%weight%position}'
    _is_bound = True
    _is_cost = True

    # Inheritance order is important here, we need to resolve to CostField.__mul__ and not FieldPoint.__mul__
    def __init__(self, field, weight, position, **kwargs):
        super().__init__(field=field, weight=weight, position=position, **kwargs)

    def __eq__(self, other):
        return super().__eq__(other)

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate the field implementation.

        Parameters
        ----------
        complex_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: ndarray
            The values of the implemented field used to create the wrapper.
        jacobians : ndarray
            The jacobians of the values with respect to the transducers.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return MultiCostFieldPoint(self, other)
            else:
                return MultiCostFieldMultiPoint(self, other)
        else:
            return NotImplemented

    def __sub__(self, target):
        return SquaredCostFieldPoint(field=self, target=target, weight=self.weight, position=self.position)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostFieldPoint(field=self.field, weight=self.weight * weight, position=self.position)


class SquaredFieldBase(Field):
    """Base class for magnitude target fields.

    Uses a field  :math:`A` to instead calculate :math:`V = |A - A_0|^2`,
    i.e. the squared magnitude difference to a target. For multi-dimensional fields
    the target needs to have the same (or a broadcastable) shape.
    The jacobians are calculated as :math:`dV = 2 dA (A-A_0)`.

    Parameters
    ----------
    field: Field-like
        A wrapper of a field implementation, of the same type as the magnitude target.
    target numpy.ndarray
        The static offset target value(s).

    Note
    ----
    This class should not be instantiated directly, only use subclasses.

    """

    def __init__(self, field, target, **kwargs):
        if type(self) == SquaredFieldBase:
            raise AssertionError('`SquaredFieldBase` should never be directly instantiated!')
        self.values_require = field.values_require
        if hasattr(field, 'jacobians_require'):
            self.jacobians_require = field.jacobians_require + field.values_require

        super().__init__(field=field, **kwargs)
        target = np.asarray(target)
        self.target = target

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.target, other.target)
        )

    @property
    def name(self):
        return self.field.name

    def values(self, **kwargs):
        """Calculate squared magnitude difference.

        If the underlying field returns :math:`A`, this function returns
        :math:`|A - A_0|^2`, where :math:`A_0` is the target value.

        For information about parameters, see the documentation of the values function
        of the underlying objects, accessed through the `field` properties.
        """
        values = self.field.values(**kwargs)
        values -= self.target.reshape(self.target.shape + (values.ndim - self.ndim) * (1,))
        return np.real(values * np.conj(values))

    def jacobians(self, **kwargs):
        """Calculate jacobians squared magnitude difference.

        If the underlying field returns :math:`dA`, the derivative of the value(s)
        with respect to the transducers, this function returns
        :math:`2 dA (A - A_0)`, where :math:`A_0` is the target value.

        For information about parameters, see the documentation of the values function
        of the underlying objects, accessed through the `field` properties.
        """
        values = self.field.values(**{key: kwargs[key] for key in self.field.values_require})
        values -= self.target.reshape(self.target.shape + (values.ndim - self.ndim) * (1,))
        jacobians = self.field.jacobians(**{key: kwargs[key] for key in self.field.jacobians_require})
        return 2 * jacobians * np.conj(values.reshape(values.shape[:self.ndim] + (1,) + values.shape[self.ndim:]))

    # These properties are needed to not overwrite the requirements defined in the field implementations.
    @property
    def values_require(self):
        return self._values_require

    @values_require.setter
    def values_require(self, val):
        self._values_require = val

    @property
    def jacobians_require(self):
        return self._jacobians_require

    @jacobians_require.setter
    def jacobians_require(self, val):
        self._jacobians_require = val

    def __sub__(self, target):
        kwargs = {}
        if self._is_bound:
            kwargs['position'] = self.position
        if self._is_cost:
            kwargs['weight'] = self.weight
        return type(self)(field=self.field, target=self.target + target, **kwargs)

    def __format__(self, format_spec):
        target_str = ' - %target' if not np.allclose(self.target, 0) else ''
        format_spec = format_spec.replace('%name', '|%name' + target_str + '|^2').replace('%target', str(self.target))
        return super().__format__(format_spec)


class SquaredField(SquaredFieldBase, Field):
    """Magnitude target field class.

    Calculates the squared magnitude difference between the field value(s)
    and a static target value(s).

    Parameters
    ----------
    field: Field
        A wrapper of an field implementation.
    target: numpy.ndarray
        The static offset target value(s).

    Methods
    -------
    +
        Adds this field with another `Field` or `MultiField`.

        :return: `MultiField`
    *
        Weight the field with a suitable weight.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `SquaredCostField`
    @
        Bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `SquaredFieldPoint`
    -
        Shifts the current target value(s) with the new values.

        :return: `SquaredField`

    """

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if SquaredFieldBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            return MultiField(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        field = self.field @ position
        return SquaredFieldPoint(field=field, target=self.target, position=field.position)

    def __mul__(self, weight):
        field = self.field * weight
        return SquaredCostField(field=field, target=self.target, weight=field.weight)


class SquaredFieldPoint(SquaredFieldBase, FieldPoint):
    """Magnitude target bound field class.

    Calculates the squared magnitude difference between the field value(s)
    and a static target value(s).

    Parameters
    ----------
    field: FieldPoint
        A wrapper of an field implementation.
    target: numpy.ndarray
        The static offset target value(s).

    Methods
    -------
    +
        Adds this field with another `FieldPoint`,
        `MultiFieldPoint`, or `MultiFieldMultiPoint`.

        :return: `MultiFieldPoint`, or `MultiFieldMultiPoint`
    *
        Weight the field with a suitable weight.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `SquaredCostFieldPoint`
    @
        Re-bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `SquaredFieldPoint`
    -
        Shifts the current target value(s) with the new values.

        :return: `SquaredFieldPoint`

    """

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if SquaredFieldBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            if np.allclose(self.position, other.position):
                return MultiFieldPoint(self, other)
            else:
                return MultiFieldMultiPoint(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        field = self.field @ position
        return SquaredFieldPoint(field=field, target=self.target, position=field.position)

    def __mul__(self, weight):
        field = self.field * weight
        return SquaredCostFieldPoint(field=field, target=self.target, weight=field.weight, position=field.position)


class SquaredCostField(SquaredFieldBase, CostField):
    """Magnitude target unbound cost field class.

    Calculates the squared magnitude difference between the field value(s)
    and a static target value(s).

    Parameters
    ----------
    field: CostField
        A wrapper of an field implementation.
    target: numpy.ndarray
        The static offset target value(s).

    Methods
    -------
    +
        Adds this field with another `CostField` or `MultiCostField`.

        :return: `MultiCostField`
    *
        Rescale the weight, i.e. multiplies the current weight with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `SquaredCostField`
    @
        Bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `SquaredCostFieldPoint`
    -
        Shifts the current target value(s) with the new values.

        :return: `SquaredCostField`

    """

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if SquaredFieldBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            return MultiCostField(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        field = self.field @ position
        return SquaredCostFieldPoint(field=field, target=self.target, position=field.position, weight=field.weight)

    def __mul__(self, weight):
        field = self.field * weight
        return SquaredCostField(field=field, target=self.target, weight=field.weight)


class SquaredCostFieldPoint(SquaredFieldBase, CostFieldPoint):
    """Magnitude target cost function class.

    Calculates the squared magnitude difference between the field value(s)
    and a static target value(s).

    Parameters
    ----------
    field: CostFieldPoint
        A wrapper of an field implementation.
    target: numpy.ndarray
        The static offset target value(s).

    Methods
    -------
    +
        Adds this field with another `CostFieldPoint`,
        `MultiCostFieldPoint`, or `MultiCostFieldMultiPoint`.

        :return: `MultiCostFieldPoint`, or `MultiCostFieldMultiPoint`
    *
        Rescale the weight, i.e. multiplies the current weight with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `SquaredCostFieldPoint`
    @
        Bind the field to a point in space. The point needs to have
        3 elements in the first dimension.

        :return: `SquaredCostFieldPoint`
    -
        Shifts the current target value(s) with the new values.

        :return: `SquaredCostFieldPoint`

    """

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if SquaredFieldBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            if np.allclose(self.position, other.position):
                return MultiCostFieldPoint(self, other)
            else:
                return MultiCostFieldMultiPoint(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        field = self.field @ position
        return SquaredCostFieldPoint(field=field, target=self.target, position=field.position, weight=field.weight)

    def __mul__(self, weight):
        field = self.field * weight
        return SquaredCostFieldPoint(field=field, target=self.target, weight=field.weight, position=field.position)


class MultiField(FieldBase):
    """Class for multiple fields, single position calculations.

    This class collects multiple `Field` objects for simultaneous evaluation at
    the same position(s). Since the fields can use the same spatial structures
    this is more efficient than to evaluate all the fields one by one.

    Parameters
    ----------
    *fields : Field
        Any number of `Field` objects.

    Methods
    -------
    +
        Adds an `Field` or `MultiField` to the current set
        of fields.

        :return: `FieldPoint`
    *
        Weights all fields with the same weight. This requires
        that all the fields can actually use the same weight.

        :return: `MultiCostField`
    @
        Binds all the fields to the same position.

        :return: `MultiFieldPoint`
    -
        Converts all the fields to squared target field.
        This requires that all the fields can use the same target.

        :return: `MultiField`

    """

    _str_format_spec = '{:%cls%fields%position}'
    _is_bound = False
    _is_cost = False

    def __init__(self, *fields):
        self.fields = []
        self.requires = FieldImplementation.requirement()
        for field in fields:
            self += field

    def __eq__(self, other):
        return super().__eq__(other) and self.fields == other.fields

    @property
    def array(self):
        return self.fields[0].array

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate all fields.

        Parameters
        ----------
        complex_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.
        position : array-like
            The position(s) where to evaluate the fields.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: list
            A list of the return values from the individual fields.
            Depending on the number of dimensions of the fields, the
            arrays in the list might not have compatible shapes.

        """
        # Prepare the requirements dict
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, position)
        # Call the function with the correct arguments
        return [field.values(**{key: requirements[key] for key in field.values_require}) for field in self.fields]

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return MultiField(*self.fields, *other.fields)
        elif self._type == other._type:
            return MultiField(*self.fields, other)
        else:
            return NotImplemented

    def __iadd__(self, other):
        add_element = False
        add_point = False
        if type(self) == type(other):
            add_point = True
        elif self._type == other._type:
            add_element = True
        old_requires = self.requires
        if add_element:
            self.requires = self.requires + other.requires
            self.fields.append(other)
        elif add_point:
            for field in other.fields:
                self += field
        else:
            return NotImplemented
        if self.requires != old_requires:
            # We have new requirements, if there are cached spatial structures they will
            # need to be recalculated at next call.
            self._clear_cache()
        return self

    def __sub__(self, other):
        return type(self)(*[field - other for field in self.fields])

    def __mul__(self, weight):
        return MultiCostField(*[field * weight for field in self.fields])

    def __matmul__(self, position):
        return MultiFieldPoint(*[field @ position for field in self.fields])

    def __format__(self, format_spec):
        if '%fields' in format_spec:
            field_start = format_spec.find('%fields')
            if len(format_spec) > field_start + 11 and format_spec[field_start + 11] == ':':
                field_spec_len = format_spec[field_start + 12].find(':')
                field_spec = format_spec[field_start + 12:field_start + 12 + field_spec_len]
                pre = format_spec[:field_start + 10]
                post = format_spec[field_start + 13 + field_spec_len:]
                format_spec = pre + post
            else:
                field_spec = '{:%name%weight}'
            field_str = '('
            for field in self.fields:
                field_str += field_spec.format(field) + ' + '
            format_spec = format_spec.replace('%fields', field_str.rstrip(' + ') + ')')
        return super().__format__(format_spec.replace('%name', ''))


class MultiFieldPoint(MultiField):
    """Class for multiple field, single fixed position calculations.

    This class collects multiple `FieldPoint` bound to the same position(s)
    for simultaneous evaluation. Since the fields can use the same spatial
    structures this is more efficient than to evaluate all the fields one by one.

    Parameters
    ----------
    *fields : FieldPoint
        Any number of `FieldPoint` objects.

    Warning
    --------
    If the class is initialized with fields bound to different points,
    some of the fields are simply discarded.

    Methods
    -------
    +
        Adds an `FieldPoint` or `MultiFieldPoint` to the current set
        of fields. If the newly added field is not bound to the same position,
        an `MultiFIeldMultiPoint` will be created and returned.

        :return: `MultiFieldPoint` or `MultiFieldMultiPoint`
    *
        Weights all fields with the same weight. This requires
        that all the fields can actually use the same weight.

        :return: `MultiCostFieldMultiPoint`
    @
        Re-binds all the fields to a new position.

        :return: `MultiFieldPoint`
    -
        Converts all the fields to magnitude target fields.
        This requires that all the fields can use the same target.

        :return: `MultiFieldPoint`

    """

    _is_bound = True
    _is_cost = False

    def __init__(self, *fields):
        self.position = fields[0].position
        super().__init__(*fields)

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate all fields.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: list
            A list of the return values from the individual fields.
            Depending on the number of dimensions of the fields, the
            arrays in the list might not have compatible shapes.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes)
        return [field.values(**{key: requirements[key] for key in field.values_require}) for field in self.fields]

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == MultiFieldPoint and np.allclose(self.position, other.position):
            return MultiFieldPoint(*self.fields, *other.fields)
        elif isinstance(other, FieldPoint) and np.allclose(self.position, other.position):
            return MultiFieldPoint(*self.fields, other)
        else:
            return MultiFieldMultiPoint(self, other)

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return MultiFieldMultiPoint(self, other)
        except AttributeError:
            return NotImplemented

    def __mul__(self, weight):
        return MultiCostFieldPoint(*[field * weight for field in self.fields])


class MultiCostField(MultiField):
    """Class for multiple cost function, single position calculations.

    This class collects multiple `CostField` objects for simultaneous evaluation at
    the same position(s). Since the fields can use the same spatial structures
    this is more efficient than to evaluate all the fields one by one.

    Parameters
    ----------
    *fields : CostField
        Any number of `CostField` objects.

    Methods
    -------
    +
        Adds an `CostField` or `MultiCostField` to the
        current set of fields.

        :return: `MultiCostField`
    *
        Rescale the weights of all fields, i.e. multiplies the current set of
        weight with the new value. The weight needs to have the correct number of
        dimensions, but will otherwise broadcast properly.

        :return: `MultiCostField`
    @
        Binds all the fields to the same position.

        :return: `MultiCostFieldPoint`
    -
        Converts all the fields to magnitude target fields.
        This requires that all the fields can use the same target.

        :return: `MultiCostField`

    """

    _is_bound = False
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate and sum the all fields.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.
        position : array-like
            The position(s) where to evaluate the fields.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: ndarray
            The summed values of all fields.
        jacobians : ndarray
            The the summed jacobians of all fields.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, position)
        value = 0
        jacobians = 0
        for field in self.fields:
            value += np.einsum(field._sum_str, field.weight, field.values(**{key: requirements[key] for key in field.values_require}))
            jacobians += np.einsum(field._sum_str, field.weight, field.jacobians(**{key: requirements[key] for key in field.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return MultiCostField(*self.fields, *other.fields)
        elif self._type == other._type:
            return MultiCostField(*self.fields, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        return MultiCostFieldPoint(*[field @ position for field in self.fields])


class MultiCostFieldPoint(MultiCostField, MultiFieldPoint):
    """Class for multiple cost function, single fixed position calculations.

    This class collects multiple `CostFieldPoint` bound to the same position(s)
    for simultaneous evaluation. Since the fields can use the same spatial
    structures this is more efficient than to evaluate all the fields one by one.

    Parameters
    ----------
    *fields : CostFieldPoint
        Any number of `CostFieldPoint` objects.

    Warning
    --------
    If the class is initialized with fields bound to different points,
    some of the fields are simply discarded.

    Methods
    -------
    +
        Adds an `CostFieldPoint` or `MultiCostFieldPoint` to the  current set of fields.
        If the newly added field is not bound to the same position,
        a `MultiCostFieldMultiPoint` will be created and returned.

        :return: `MultiCostFieldPoint` or `MultiCostFieldMultiPoint`
    *
        Rescale the weights of all fields, i.e. multiplies the current set of
        weight with the new value. The weight needs to have the correct number of
        dimensions, but will otherwise broadcast properly.

        :return: `MultiCostFieldPoint`
    @
        Re-binds all the fields to a new position.

        :return: `MultiCostFieldPoint`
    -
        Converts all the fields to magnitude target fields.
        This requires that all the fields can use the same target.

        :return: `MultiCostFieldPoint`

    """

    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate and sum the all fields.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: ndarray
            The summed values of all cost functions.
        jacobians : ndarray
            The the summed jacobians of all cost functions.

        """
        requirements = self.evaluate_requirements(complex_transducer_amplitudes)
        value = 0
        jacobians = 0
        for field in self.fields:
            value += np.einsum(field._sum_str, field.weight, field.values(**{key: requirements[key] for key in field.values_require}))
            jacobians += np.einsum(field._sum_str, field.weight, field.jacobians(**{key: requirements[key] for key in field.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == MultiCostFieldPoint and np.allclose(self.position, other.position):
            return MultiCostFieldPoint(*self.fields, *other.fields)
        elif isinstance(other, CostFieldPoint) and np.allclose(self.position, other.position):
            return MultiCostFieldPoint(*self.fields, other)
        else:
            return MultiCostFieldMultiPoint(self, other)

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return MultiCostFieldMultiPoint(self, other)
        except AttributeError:
            return NotImplemented


class MultiFieldMultiPoint(FieldBase):
    """Collects fields bound to different positions.

    Convenience class to evaluate and manipulate fields bound to
    different positions in space. Will not improve the computational
    efficiency beyond the gains from merging the fields bound
    to the same positions.

    Parameters
    ----------
    *fields: FieldPoint or MultiFieldPoint
        Any number of fields bound to any number of points.

    Methods
    -------
    +
        Adds an additional `FieldPoint`, `MultiFieldPoint`
        or `MultiFieldMultiPoint` to the set.

        :return: `MultiFieldMultiPoint`
    *
        Weights all fields in the set with the same weight.
        Requires that they can actually be weighted with the same
        weight.

        :return: `MultiCostFieldMultiPoint`

    """

    _str_format_spec = '{:%cls%points}'
    _is_bound = True
    _is_cost = False

    def __init__(self, *fields):
        self.fields = []
        for field in fields:
            self += field

    def __eq__(self, other):
        return super().__eq__(other) and self.fields == other.fields

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate all fields.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: list
            A list of the return values from the individual fields.
            Depending on the number of dimensions of the fields, the
            arrays in the list might not have compatible shapes, and some
            might be lists with values corresponding to the same point in space.

        """
        values = []
        for point in self.fields:
            values.append(point(complex_transducer_amplitudes))
        return values

    def __add__(self, other):
        if other == 0:
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            return type(self)(*self.fields, other)

    def __iadd__(self, other):
        if type(other) == type(self):
            for field in other.fields:
                self += field
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            for idx, point in enumerate(self.fields):
                if np.allclose(point.position, other.position):
                    # Mutating `point` will not update the contents in the list!
                    self.fields[idx] += other
                    break
            else:
                self.fields.append(other)
            return self

    def __mul__(self, weight):
        return MultiCostFieldMultiPoint(*[field * weight for field in self.fields])

    def __format__(self, format_spec):
        if '%points' in format_spec:
            points_start = format_spec.find('%points')
            if len(format_spec) > points_start + 7 and format_spec[points_start + 7] == ':':
                points_spec_len = format_spec[points_start + 8].rind(':')
                points_spec = format_spec[points_start + 8:points_start + 8 + points_spec_len]
                pre = format_spec[:points_start + 6]
                post = format_spec[points_start + 9 + points_spec_len:]
                format_spec = pre + post
            else:
                points_spec = '\t{:%cls%name%fields%weight%position}\n'
            points_str = '[\n'
            for field in self.fields:
                points_str += points_spec.format(field).replace('%fields', '')
            format_spec = format_spec.replace('%points', points_str + ']')
        return super().__format__(format_spec)


class MultiCostFieldMultiPoint(MultiFieldMultiPoint, MultiCostFieldPoint):
    """Collects cost fields bound to different positions.

    Convenience class to evaluate and manipulate cost fields bound to
    different positions in space. Will not improve the computational
    efficiency beyond the gains from merging the fields bound
    to the same positions.

    Parameters
    ----------
    *fields: CostFieldPoint or MultiCostFieldPoint
        Any number of cost fields bound to any number of points.

    Methods
    -------
    +
        Adds an additional `CostFieldPoint`, `MultiFieldPoint`
        or `MultiCostFieldMultiPoint` to the set.

        :return: `MultiCostFieldMultiPoint`
    *
        Rescale the weights, i.e. multiplies the current weights with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `MultiCostFieldMultiPoint`

    """

    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate and sum the all fields.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the field.

        Returns
        -------
        values: ndarray
            The summed values of all cost functions.
        jacobians : ndarray
            The the summed jacobians of all cost functions.

        """
        values = 0
        jacobians = 0
        for field in self.fields:
            val, jac = field(complex_transducer_amplitudes)
            values += val
            jacobians += jac
        return values, jacobians
