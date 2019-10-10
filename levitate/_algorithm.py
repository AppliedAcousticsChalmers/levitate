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
An `MultiFieldMultiPoint` returns the values from the stored algorithms.
A `MultiCostFieldMultiPoint` will sum the values and jacobians of the stored objects.

.. autosummary::
    :nosignatures:

    MultiFieldMultiPoint
    MultiCostFieldMultiPoint

Implementation Details
----------------------
To make the API work as intended, there are a couple additional
classes and functions.
The base class for the implemented algorithms, `FieldImplementation` is
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

    possible_requirements = [
        'complex_transducer_amplitudes',
        'pressure_derivs_summed', 'pressure_derivs_individual',
        'spherical_harmonics_summed', 'spherical_harmonics_individual',
    ]

    @staticmethod
    def requirement(**requirements):
        """Parse a set of requirements.

        `FieldImplementation` objects should define requirements for values and jacobians.
        This function parses the requirements and checks that the request can be met upon call.
        Currently the inputs are converted to a dict and returned as is, but this might change
        without warning in the future.

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

        Returns
        -------
        requirements : dict
            The parsed requirements.

        Raises
        ------
        NotImplementedError
            If one or more of the requested keys is not implemented.

        """
        for requirement in requirements:
            if requirement not in FieldImplementation.possible_requirements:
                raise NotImplementedError("Requirement '{}' is not implemented for a field. The possible requests are: {}".format(requirement, FieldImplementation.possible_requirements))
        return requirements


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

    @property
    def _type(self):  # noqa: D401
        """The type of the field.

        In this context `type` refers for the combination of `bound` and `cost`.
        """
        return type(self)._type

    def __eq__(self, other):
        return type(self) == type(other)

    def _evaluate_requirements(self, complex_transducer_amplitudes, spatial_structures):
        """Evaluate requirements for given complex transducer amplitudes.

        Parameters
        ----------
        complex_transducer_amplitudes: complex ndarray
            The transducer phase and amplitude on complex form,
            must correspond to the same array used to create the field.
        spatial_structures: dict
            Dictionary with the calculated spatial structures required by the field(s).

        Returns
        -------
        requirements : dict
            Has (at least) the same fields as `self.requires`, but instead of values specifying the level
            of the requirement, this dict has the evaluated requirement at the positions and
            transducer amplitudes specified.

        """
        requirements = {}
        if 'complex_transducer_amplitudes' in self.requires:
            requirements['complex_transducer_amplitudes'] = complex_transducer_amplitudes
        if 'pressure_derivs' in spatial_structures:
            requirements['pressure_derivs_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, spatial_structures['pressure_derivs'])
            requirements['pressure_derivs_summed'] = np.sum(requirements['pressure_derivs_individual'], axis=1)
        if 'spherical_harmonics' in spatial_structures:
            requirements['spherical_harmonics_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, spatial_structures['spherical_harmonics'])
            requirements['spherical_harmonics_summed'] = np.sum(requirements['spherical_harmonics_individual'], axis=1)
        return requirements

    def _spatial_structures(self, position=None):
        """Calculate spatial structures.

        Uses `self.requires` to fill a dictionary of calculated required
        spatial structures at a give position to satisfy the fields(s) used
        for calculations.

        Parameters
        ----------
        position: ndarray
            The position where to calculate the spatial structures needed.
            Shape (3,...). If position is `None` or not passed, it is assumed
            that the field is bound to a position and `self.position` will be used.

        Returns
        -------
        sptaial_structures : dict
            Dictionary with the spatial structures required to fulfill the evaluation
            of the field(s).

        Note
        ----
        Fields which are bound to a position will cache the spatial structures. It is
        therefore important to not manually change the position, since that will not clear the cache
        and the new position is not actually used.

        """
        # If called without a position we are using a field point, check the cache and calculate it if needed
        if position is None:
            try:
                return self._cached_spatial_structures
            except AttributeError:
                self._cached_spatial_structures = self._spatial_structures(self.position)
                return self._cached_spatial_structures
        # Check what spatial structures we need from the array to fulfill the requirements
        spatial_structures = {}
        for key, value in self.requires.items():
            if key.find('pressure_derivs') > -1:
                spatial_structures['pressure_derivs'] = max(value, spatial_structures.get('pressure_derivs', -1))
            elif key.find('spherical_harmonics') > -1:
                spatial_structures['spherical_harmonics'] = max(value, spatial_structures.get('spherical_harmonics', -1))
            elif key != 'complex_transducer_amplitudes':
                raise ValueError("Unknown requirement '{}'".format(key))
        # Replace the requests with values calculated by the array
        if 'pressure_derivs' in spatial_structures:
            spatial_structures['pressure_derivs'] = self.array.pressure_derivs(position, orders=spatial_structures['pressure_derivs'])
        if 'spherical_harmonics' in spatial_structures:
            spatial_structures['spherical_harmonics'] = self.array.spherical_harmonics(position, orders=spatial_structures['spherical_harmonics'])
        return spatial_structures

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
        self.requires = self.field.values_require.copy()

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
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return AlgorithmPoint(self, other)
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
        self.position = position

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
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return BoundAlgorithmPoint(self, other)
            else:
                return AlgorithmCollection(self, other)
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
        Converts to a magnitude target algorithm.

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
        for key, value in self.jacobians_require.items():
            self.requires[key] = max(value, self.requires.get(key, -1))

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
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return UnboundCostFunctionPoint(self, other)
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
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return CostFunctionPoint(self, other)
            else:
                return CostFunctionCollection(self, other)
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
        self.values_require = field.values_require.copy()
        self.jacobians_require = field.jacobians_require.copy()
        for key, value in field.values_require.items():
            self.jacobians_require[key] = max(value, self.jacobians_require.get(key, -1))
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
            return AlgorithmPoint(self, other)
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
                return BoundAlgorithmPoint(self, other)
            else:
                return AlgorithmCollection(self, other)
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
            return UnboundCostFunctionPoint(self, other)
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
                return CostFunctionPoint(self, other)
            else:
                return CostFunctionCollection(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        field = self.field @ position
        return SquaredCostFieldPoint(field=field, target=self.target, position=field.position, weight=field.weight)

    def __mul__(self, weight):
        field = self.field * weight
        return SquaredCostFieldPoint(field=field, target=self.target, weight=field.weight, position=field.position)


class AlgorithmPoint(FieldBase):
    """Class for multiple algorithm, single position calculations.

    This class collects multiple `Algorithm` objects for simultaneous evaluation at
    the same position(s). Since the algorithms can use the same spatial structures
    this is more efficient than to evaluate all the algorithms one by one.

    Parameters
    ----------
    *algorithms : Algorithm
        Any number of `Algorithm` objects.

    Methods
    -------
    +
        Adds an `Algorithm` or `AlgorithmPoint` to the current set
        of algorithms.

        :return: `AlgorithmPoint`
    *
        Weights all algorithms with the same weight. This requires
        that all the algorithms can actually use the same weight.

        :return: `UnboundCostFunctionPoint`
    @
        Binds all the algorithms to the same position.

        :return: `BoundAlgorithmPoint`
    -
        Converts all the algorithms to magnitude target algorithms.
        This requires that all the algorithms can use the same target.

        :return: `AlgorithmPoint`

    """

    _str_format_spec = '{:%cls%algorithms%position}'
    _is_bound = False
    _is_cost = False

    def __init__(self, *algorithms):
        self.algorithms = []
        self.requires = {}
        for algorithm in algorithms:
            self += algorithm

    def __eq__(self, other):
        return super().__eq__(other) and self.algorithms == other.algorithms

    @property
    def array(self):
        return self.algorithms[0].array

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate all algorithms.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.
        position : array-like
            The position(s) where to evaluate the algorithms.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: list
            A list of the return values from the individual algorithms.
            Depending on the number of dimensions of the algorithms, the
            arrays in the list might not have compatible shapes.

        """
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return [algorithm.values(**{key: requirements[key] for key in algorithm.values_require}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return AlgorithmPoint(*self.algorithms, *other.algorithms)
        elif self._type == other._type:
            return AlgorithmPoint(*self.algorithms, other)
        else:
            return NotImplemented

    def __iadd__(self, other):
        add_element = False
        add_point = False
        if type(self) == type(other):
            add_point = True
        elif self._type == other._type:
            add_element = True
        old_requires = self.requires.copy()
        if add_element:
            for key, value in other.requires.items():
                self.requires[key] = max(value, self.requires.get(key, -1))
            self.algorithms.append(other)
        elif add_point:
            for algorithm in other.algorithms:
                self += algorithm
        else:
            return NotImplemented
        if self.requires != old_requires:
            # We have new requirements, if there are cached spatial structures they will
            # need to be recalculated at next call.
            try:
                del self._cached_spatial_structures
            except AttributeError:
                pass
        return self

    def __sub__(self, other):
        return type(self)(*[algorithm - other for algorithm in self.algorithms])

    def __mul__(self, weight):
        return UnboundCostFunctionPoint(*[algorithm * weight for algorithm in self.algorithms])

    def __matmul__(self, position):
        return BoundAlgorithmPoint(*[algorithm @ position for algorithm in self.algorithms])

    def __format__(self, format_spec):
        if '%algorithms' in format_spec:
            alg_start = format_spec.find('%algorithms')
            if len(format_spec) > alg_start + 11 and format_spec[alg_start + 11] == ':':
                alg_spec_len = format_spec[alg_start + 12].find(':')
                alg_spec = format_spec[alg_start + 12:alg_start + 12 + alg_spec_len]
                pre = format_spec[:alg_start + 10]
                post = format_spec[alg_start + 13 + alg_spec_len:]
                format_spec = pre + post
            else:
                alg_spec = '{:%name%weight}'
            alg_str = '('
            for algorithm in self.algorithms:
                alg_str += alg_spec.format(algorithm) + ' + '
            format_spec = format_spec.replace('%algorithms', alg_str.rstrip(' + ') + ')')
        return super().__format__(format_spec.replace('%name', ''))


class BoundAlgorithmPoint(AlgorithmPoint):
    """Class for multiple algorithm, single fixed position calculations.

    This class collects multiple `BoundAlgorithm` bound to the same position(s)
    for simultaneous evaluation. Since the algorithms can use the same spatial
    structures this is more efficient than to evaluate all the algorithms one by one.

    Parameters
    ----------
    *algorithms : BoundAlgorithm
        Any number of `BoundAlgorithm` objects.

    Warning
    --------
    If the class is initialized with algorithms bound to different points,
    some of the algorithms are simply discarded.

    Methods
    -------
    +
        Adds an `BoundAlgorithm` or `BoundAlgorithmPoint` to the current set
        of algorithms. If the newly added algorithm is not bound to the same position,
        an `AlgorithmCollection` will be created and returned.

        :return: `BoundAlgorithmPoint` or `AlgorithmCollection`
    *
        Weights all algorithms with the same weight. This requires
        that all the algorithms can actually use the same weight.

        :return: `CostFunctionPoint`
    @
        Re-binds all the algorithms to a new position.

        :return: `BoundAlgorithmPoint`
    -
        Converts all the algorithms to magnitude target algorithms.
        This requires that all the algorithms can use the same target.

        :return: `BoundAlgorithmPoint`

    """

    _is_bound = True
    _is_cost = False

    def __init__(self, *algorithms):
        self.position = algorithms[0].position
        super().__init__(*algorithms)

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate all algorithms.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.

        Returns
        -------
        values: list
            A list of the return values from the individual algorithms.
            Depending on the number of dimensions of the algorithms, the
            arrays in the list might not have compatible shapes.

        """
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return [algorithm.values(**{key: requirements[key] for key in algorithm.values_require}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == BoundAlgorithmPoint and np.allclose(self.position, other.position):
            return BoundAlgorithmPoint(*self.algorithms, *other.algorithms)
        elif isinstance(other, FieldPoint) and np.allclose(self.position, other.position):
            return BoundAlgorithmPoint(*self.algorithms, other)
        else:
            return AlgorithmCollection(self, other)

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return AlgorithmCollection(self, other)
        except AttributeError:
            return NotImplemented

    def __mul__(self, weight):
        return CostFunctionPoint(*[algorithm * weight for algorithm in self.algorithms])


class UnboundCostFunctionPoint(AlgorithmPoint):
    """Class for multiple cost function, single position calculations.

    This class collects multiple `UnboundCostFunction` objects for simultaneous evaluation at
    the same position(s). Since the algorithms can use the same spatial structures
    this is more efficient than to evaluate all the algorithms one by one.

    Parameters
    ----------
    *algorithms : UnboundCostFunction
        Any number of `UnboundCostFunction` objects.

    Methods
    -------
    +
        Adds an `UnboundCostFunction` or `UnboundCostFunctionPoint` to the
        current set of algorithms.

        :return: `UnboundCostFunctionPoint`
    *
        Rescale the weights of all algorithms, i.e. multiplies the current set of
        weight with the new value. The weight needs to have the correct number of
        dimensions, but will otherwise broadcast properly.

        :return: `UnboundCostFunctionPoint`
    @
        Binds all the algorithms to the same position.

        :return: `CostFunctionPoint`
    -
        Converts all the algorithms to magnitude target algorithms.
        This requires that all the algorithms can use the same target.

        :return: `UnboundCostFunctionPoint`

    """

    _is_bound = False
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes, position):
        """Evaluate the all cost functions.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.
        position : array-like
            The position(s) where to evaluate the algorithm.
            The first dimension needs to have 3 elements.

        Returns
        -------
        values: ndarray
            The summed values of all cost functions.
        jacobians : ndarray
            The the summed jacobians of all cost functions.

        """
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.values(**{key: requirements[key] for key in algorithm.values_require}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.jacobians(**{key: requirements[key] for key in algorithm.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return UnboundCostFunctionPoint(*self.algorithms, *other.algorithms)
        elif self._type == other._type:
            return UnboundCostFunctionPoint(*self.algorithms, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        return CostFunctionPoint(*[algorithm @ position for algorithm in self.algorithms])


class CostFunctionPoint(UnboundCostFunctionPoint, BoundAlgorithmPoint):
    """Class for multiple cost function, single fixed position calculations.

    This class collects multiple `CostFunction` bound to the same position(s)
    for simultaneous evaluation. Since the algorithms can use the same spatial
    structures this is more efficient than to evaluate all the algorithms one by one.

    Parameters
    ----------
    *algorithms : CostFunction
        Any number of `CostFunction` objects.

    Warning
    --------
    If the class is initialized with algorithms bound to different points,
    some of the algorithms are simply discarded.

    Methods
    -------
    +
        Adds an `CostFunction` or `CostFunctionPoint` to the  current set of algorithms.
        If the newly added algorithm is not bound to the same position,
        a `CostFunctionCollection` will be created and returned.

        :return: `CostFunctionPoint` or `CostFunctionCollection`
    *
        Rescale the weights of all algorithms, i.e. multiplies the current set of
        weight with the new value. The weight needs to have the correct number of
        dimensions, but will otherwise broadcast properly.

        :return: `CostFunctionPoint`
    @
        Re-binds all the algorithms to a new position.

        :return: `CostFunctionPoint`
    -
        Converts all the algorithms to magnitude target algorithms.
        This requires that all the algorithms can use the same target.

        :return: `CostFunctionPoint`

    """

    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate the all cost functions.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.

        Returns
        -------
        values: ndarray
            The summed values of all cost functions.
        jacobians : ndarray
            The the summed jacobians of all cost functions.

        """
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.values(**{key: requirements[key] for key in algorithm.values_require}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.jacobians(**{key: requirements[key] for key in algorithm.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == CostFunctionPoint and np.allclose(self.position, other.position):
            return CostFunctionPoint(*self.algorithms, *other.algorithms)
        elif isinstance(other, CostFieldPoint) and np.allclose(self.position, other.position):
            return CostFunctionPoint(*self.algorithms, other)
        else:
            return CostFunctionCollection(self, other)

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return CostFunctionCollection(self, other)
        except AttributeError:
            return NotImplemented


class AlgorithmCollection(FieldBase):
    """Collects algorithms bound to different positions.

    Convenience class to evaluate and manipulate algorithms bound to
    different positions in space. Will not improve the computational
    efficiency beyond the gains from merging the algorithms bound
    to the same positions.

    Parameters
    ----------
    *algorithms: BoundAlgorithm or BoundAlgorithmPoint
        Any number of algorithms bound to any number of points.

    Methods
    -------
    +
        Adds an additional `BoundAlgorithm`, `BoundAlgorithmPoint`
        or `AlgorithmCollection` to the set.

        :return: `AlgorithmCollection`
    *
        Weights all algorithms in the set with the same weight.
        Requires that they can actually be weighted with the same
        weight.

        :return: `CostFuncitonCollection`

    """

    _str_format_spec = '{:%cls%points}'
    _is_bound = True
    _is_cost = False

    def __init__(self, *algorithms):
        self.algorithms = []
        for algorithm in algorithms:
            self += algorithm

    def __eq__(self, other):
        return super().__eq__(other) and self.algorithms == other.algorithms

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate all algorithms.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.

        Returns
        -------
        values: list
            A list of the return values from the individual algorithms.
            Depending on the number of dimensions of the algorithms, the
            arrays in the list might not have compatible shapes, and some
            might be lists with values corresponding to the same point in space.

        """
        values = []
        for point in self.algorithms:
            values.append(point(complex_transducer_amplitudes))
        return values

    def __add__(self, other):
        if other == 0:
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            return type(self)(*self.algorithms, other)

    def __iadd__(self, other):
        if type(other) == type(self):
            for algorithm in other.algorithms:
                self += algorithm
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            for idx, point in enumerate(self.algorithms):
                if np.allclose(point.position, other.position):
                    # Mutating `point` will not update the contents in the list!
                    self.algorithms[idx] += other
                    break
            else:
                self.algorithms.append(other)
            return self

    def __mul__(self, weight):
        return CostFunctionCollection(*[algorithm * weight for algorithm in self.algorithms])

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
                points_spec = '\t{:%cls%name%algorithms%weight%position}\n'
            points_str = '[\n'
            for algorithm in self.algorithms:
                points_str += points_spec.format(algorithm).replace('%algorithms', '')
            format_spec = format_spec.replace('%points', points_str + ']')
        return super().__format__(format_spec)


class CostFunctionCollection(AlgorithmCollection, CostFunctionPoint):
    """Collects cost functions bound to different positions.

    Convenience class to evaluate and manipulate cost functions bound to
    different positions in space. Will not improve the computational
    efficiency beyond the gains from merging the algorithms bound
    to the same positions.

    Parameters
    ----------
    *algorithms: CostFunction or CostFunctionPoint
        Any number of cost functions bound to any number of points.

    Methods
    -------
    +
        Adds an additional `CostFunction`, `CostFunctionPoint`
        or `CostFunctionCollection` to the set.

        :return: `CostFunctionCollection`
    *
        Rescale the weights, i.e. multiplies the current weights with the new value.
        The weight needs to have the correct number of dimensions, but will
        otherwise broadcast properly.

        :return: `CostFuncitonCollection`

    """

    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        """Evaluate the all cost functions.

        Parameters
        ----------
        compelx_transducer_amplitudes : complex numpy.ndarray
            Complex representation of the transducer phases and amplitudes of the
            array used to create the algorithm.

        Returns
        -------
        values: ndarray
            The summed values of all cost functions.
        jacobians : ndarray
            The the summed jacobians of all cost functions.

        """
        values = 0
        jacobians = 0
        for algorithm in self.algorithms:
            val, jac = algorithm(complex_transducer_amplitudes)
            values += val
            jacobians += jac
        return values, jacobians
