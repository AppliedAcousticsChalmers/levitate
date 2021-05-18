import numpy as np
import collections
import inspect

from . import _transformers


class IncompatibleFieldsError(TypeError):
    pass


class FieldImplementationMeta(type):
    """Metaclass to wrap `FieldImplementation` objects in `Field` objects.

    API-wise it is nice to call the implementation classes when requesting a field.
    Since the behavior of the objects should change depending on if they are added etc,
    it would be very difficult to keep track of both the current state and the actual field
    in the same top level object. This class will upon object creation instantiate the called class,
    but also instantiate and return a `Field`-type object.
    """

    def __init__(cls, clsname, bases, attrs):
        # Restore the signature
        # This seems to be needed for certain tools to properly see the init
        # signature of the implemented class.
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

        return super().__init__(clsname, bases, attrs)

    def __call__(cls, *cls_args, **cls_kwargs):
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
        **cls_kwargs :
            Keyword arguments passed to `cls`.

        """
        obj = cls.__new__(cls, *cls_args, **cls_kwargs)
        obj.__init__(*cls_args, **cls_kwargs)
        return Field(field=obj)


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

    def __init__(self, array):  # noqa: D205, D400
        """
        Parameters
        ----------
        array : TransducerArray
            The object modeling the array.

        """
        self.array = array

    def __eq__(self, other):
        return type(self) == type(other) and self.array == other.array

    @property
    def ndim(self):
        return len(self.shape)

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

        def includes(self, other):
            if not isinstance(other, (dict, collections.UserDict)):
                raise TypeError(f'Cannot check if a {type(self).__name__} includes a {type(other).__name__}')

            # For self to include other, all keys in other must exist in self
            if len(other.keys() - self.keys()) > 0:
                return False

            # For self to include other, all keys in other must exist in self, with a larger or equal value.
            for key in other:
                if key not in self:
                    return False
                if other[key] > self[key]:
                    return False

            return True


class FieldBase:
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

    def __init__(self, *, transforms=None):
        self.transforms = transforms if transforms is not None else tuple()

    def copy(self):
        new_obj = type(self).__new__(type(self))
        new_obj.transforms = self.transforms
        return new_obj

    def evaluate_requirements(self, complex_transducer_amplitudes, requests):
        complex_transducer_amplitudes = np.asarray(complex_transducer_amplitudes)
        # Apply the input complex amplitudes
        evaluated_requrements = {}
        evaluated_requrements['complex_transducer_amplitudes'] = complex_transducer_amplitudes
        if 'pressure_derivs' in requests:
            evaluated_requrements['pressure_derivs_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, requests['pressure_derivs'])
            evaluated_requrements['pressure_derivs_summed'] = np.sum(evaluated_requrements['pressure_derivs_individual'], axis=1)
        if 'spherical_harmonics' in requests:
            evaluated_requrements['spherical_harmonics_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, requests['spherical_harmonics'])
            evaluated_requrements['spherical_harmonics_summed'] = np.sum(evaluated_requrements['spherical_harmonics_individual'], axis=1)
        if 'spherical_harmonics_gradient' in requests:
            evaluated_requrements['spherical_harmonics_gradient_individual'] = np.einsum('i,jki...->jki...', complex_transducer_amplitudes, requests['spherical_harmonics_gradient'])
            evaluated_requrements['spherical_harmonics_gradient_summed'] = np.sum(evaluated_requrements['spherical_harmonics_gradient_individual'], axis=2)
        return evaluated_requrements

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__format__('')

    def __format__(self, fmt_spec):
        for transform in self.transforms:
            fmt_spec = transform._transform_str(fmt_spec)
        return fmt_spec

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def jacobians(self, requirements, transform=True):
        return self.values_jacobians(requirements, transform=transform)[1]

    @property
    def values_jacobians_require(self):
        values_require = self.values_require
        jacobians_require = self.jacobians_require

        if isinstance(values_require, FieldImplementation.requirement):
            return values_require + jacobians_require
        else:
            return [vals + jacs for (vals, jacs) in zip(values_require, jacobians_require)]

    @property
    def _output_layer(self):
        if len(self.transforms) > 0:
            return self.transforms[-1]
        try:
            return self.field
        except AttributeError:
            return self

    @property
    def shape(self):
        output = self._output_layer
        if output is self:
            return [field.shape for field in output]
        else:
            return output.shape

    @property
    def ndim(self):
        shape = self.shape
        if isinstance(shape, tuple):
            return len(shape)
        return None

    def _append_transform(self, transform_type, *args, **kwargs):
        self.transforms = self.transforms + (transform_type(self._output_layer, *args, **kwargs),)
        return self

    def __add__(self, other):
        try:
            return stack(self, other)._append_transform(_transformers.FieldSum)
        except IncompatibleFieldsError:
            try:
                return self.copy()._append_transform(_transformers.Shift, other)
            except _transformers.NonNumericError:
                return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            return stack(self, other)._append_transform(_transformers.Product)
        except IncompatibleFieldsError:
            try:
                return self.copy()._append_transform(_transformers.Scale, other)
            except _transformers.NonNumericError:
                return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, FieldBase):
            return self.__mul__(other ** -1)
        else:
            return self.__mul__(1 / np.ararray(other))

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __pow__(self, other):
        try:
            return self.copy()._append_transform(_transformers.Power, other)
        except _transformers.NonNumericError:
            return NotImplemented

    def __rpow__(self, other):
        try:
            return self.copy()._append_transform(_transformers.Exponential, other)
        except _transformers.NonNumericError:
            return NotImplemented

    def __neg__(self):
        return self.copy()._append_transform(_transformers.Negate)

    def __abs__(self):
        return self.copy()._append_transform(_transformers.Absolute)

    def sum(self, axis=None):
        if self.ndim is None:
            return self.copy()._append_transform(_transformers.FieldSum)
        elif self.ndim >= 0:
            return self.copy()._append_transform(_transformers.ComponentSum, axis=axis)


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

    """

    def __init__(self, field, **kwargs):
        super().__init__(**kwargs)
        self.field = field
        value_indices = ''.join(chr(ord('i') + idx) for idx in range(self.ndim))
        self._sum_str = value_indices + ', ' + value_indices + '...'

    def copy(self):
        new_obj = super().copy()
        new_obj.field = self.field
        return new_obj

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.field == other.field
            and self.array == other.array
        )

    @property
    def name(self):
        return self.field.__class__.__name__

    def values(self, requirements, transform=True):
        values = self.field.values(**{key: requirements[key] for key in self.values_require})
        if transform:
            for transform in self.transforms:
                values = transform.values(values)
        return values

    def values_jacobians(self, requirements, transform=True):
        values = self.field.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.field.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        if transform:
            for transform in self.transforms:
                values, jacobians = transform.values_jacobians(values, jacobians)
        return values, jacobians

    @property
    def values_require(self):
        return self.field.values_require

    @property
    def jacobians_require(self):
        return self.field.jacobians_require

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
        requests = self.array.request(self.values_require, position)
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, requests)
        values = self.values(requirements)
        return values

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return FieldPoint(position=position, field=self.field, transforms=self.transforms)

    def __format__(self, fmt_spec):
        fmt_spec = fmt_spec or '%name'
        name = getattr(self, 'name', None) or 'Unknown'
        return super().__format__(fmt_spec.replace('%name', name))


class FieldPoint(Field):
    """Position-bound class for single point, single field.

    See `Field` for more precise description.

    Parameters
    ----------
    field : FieldImplementation
        The implemented field to use for calculations.
    position : numpy.ndarray
        The position to bind to.

    """

    def __init__(self, field, position, **kwargs):
        super().__init__(field=field, **kwargs)
        self.position = np.asarray(position)

    def copy(self):
        new_obj = super().copy()
        new_obj.position = self.position
        return new_obj

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.position, other.position)
        )

    def _clear_cache(self):
        try:
            del self._cached_requests
        except AttributeError:
            pass

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
        try:
            requests = self._cached_requests
        except AttributeError:
            requests = self._cached_requests = self.array.request(self.values_require, self.position)
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, requests)
        values = self.values(requirements)
        return values

    def __format__(self, fmt_spec):
        fmt_spec = fmt_spec or '%name%position'
        pos = ' @ ' + str(self.position).replace('\n', '')
        return super().__format__(fmt_spec.replace('%position', pos))


class MultiFieldBase(FieldBase):

    def extend(self, other):
        for field in other:
            self.append(field)
        return self

    def __getitem__(self, idx):
        return self.fields[idx]

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        # Needed in case the object is modified while iterating.
        # This will freeze the fields before iteration starts.
        for field in tuple(self.fields):
            yield field

    @property
    def array(self):
        return self.fields[0].array

    def __format__(self, fmt_spec):
        fmt_spec = fmt_spec or '%fields'
        if '%fields' in fmt_spec:
            field_strings = [field.__format__('') for field in self.fields]
            if sum(map(len, field_strings)) + 2 * len(field_strings) + 2 < 80:
                # All the fields fit on a single line
                field_str = '[' + ', '.join(field_strings) + ']'
            else:
                # Put each field on a separate line, indented one level
                field_strings = [s.replace('\n', '\n\t') for s in field_strings]
                field_str = '[\n\t' + ',\n\t'.join(field_strings) + ',\n]'
            fmt_spec = fmt_spec.replace('%fields', field_str)
        return super().__format__(fmt_spec)


class MultiField(MultiFieldBase):
    """Class for multiple fields, single position calculations.

    This class collects multiple `Field` objects for simultaneous evaluation at
    the same position(s). Since the fields can use the same spatial structures
    this is more efficient than to evaluate all the fields one by one.

    Parameters
    ----------
    *fields : Field
        Any number of `Field` objects.
    """

    def __init__(self, *fields, **kwargs):
        super().__init__(**kwargs)
        self.fields = []
        self.values_require = FieldImplementation.requirement()
        self.jacobians_require = FieldImplementation.requirement()
        self.extend(fields)

    def copy(self):
        new_obj = super().copy()
        new_obj.fields = list(self.fields)
        new_obj.values_require = self.values_require
        new_obj.jacobians_require = self.jacobians_require
        return new_obj

    def __eq__(self, other):
        return super().__eq__(other) and self.fields == other.fields

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
        requests = self.array.request(self.values_require, position)
        requirements = self.evaluate_requirements(complex_transducer_amplitudes, requests)
        values = self.values(requirements)
        return values

    def values(self, requirements, transform=True):
        values = []
        for field in self.fields:
            values.append(field.values(requirements))
        if transform:
            for transform in self.transforms:
                values = transform.values(values)
        return values

    def values_jacobians(self, requirements, transform=True):
        values = []
        jacobians = []
        for field in self.fields:
            field_values, field_jacobians = field.values_jacobians(requirements)
            values.append(field_values)
            jacobians.append(field_jacobians)
        if transform:
            for transform in self.transforms:
                values, jacobians = transform.values_jacobians(values, jacobians)
        return values, jacobians

    def append(self, other):
        if not isinstance(other, (Field, MultiField)):
            raise IncompatibleFieldsError(f'Cannot append a {type(other).__name__} to a {type(self).__name__}')
        if not self.values_require.includes(other.values_require):
            self.values_require = self.values_require + other.values_require
        if not self.jacobians_require.includes(other.jacobians_require):
            self.jacobians_require = self.jacobians_require + other.jacobians_require
        self.fields.append(other)
        return self

    def __matmul__(self, position):
        return MultiFieldPoint(*[field @ position for field in self.fields])


class MultiFieldPoint(MultiFieldBase):
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

    """

    def __init__(self, *fields, **kwargs):
        super().__init__(*kwargs)
        self.fields = []
        self.values_require = []
        self.jacobians_require = []
        self._field_position_idx = []

        self.positions = []
        self._cached_requests = []
        self.extend(fields)

    def copy(self):
        new_obj = super().copy()
        new_obj.fields = list(self.fields)
        new_obj.values_require = list(self.values_require)
        new_obj.jacobians_require = list(self.jacobians_require)
        new_obj.positions = list(self.positions)
        new_obj._field_position_idx = list(self._field_position_idx)
        new_obj._clear_cache()
        return new_obj

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
        for idx, (position, requirement) in enumerate(zip(self.positions, self.values_require)):
            if self._cached_requests[idx] is None:
                self._cached_requests[idx] = self.array.request(requirement, position)

        requirements = [self.evaluate_requirements(complex_transducer_amplitudes, request) for request in self._cached_requests]
        values = self.values(requirements)
        return values

    def map_positions_to_fields(self, position_quanties):
        field_quantities = []
        for field, pos_idx in zip(self.fields, self._field_position_idx):
            if type(pos_idx) is int:
                field_quantity = position_quanties[pos_idx]
            else:
                field_quantity = [position_quanties[idx] for idx in pos_idx]
            field_quantities.append(field_quantity)
        return field_quantities

    def _find_pos_idx(self, position):
        for idx, pos in enumerate(self.positions):
            if pos.shape != position.shape:
                continue
            if not np.allclose(pos, position):
                continue
            return idx
        # The position does not match any of the existing positions.
        # Add the new position, as well as an empty requirements dict.
        self.positions.append(position)
        self.values_require.append(FieldImplementation.requirement())
        self.jacobians_require.append(FieldImplementation.requirement())
        self._cached_requests.append(None)
        return len(self.positions) - 1

    def values(self, requirements, transform=True):
        values = []
        requirements = self.map_positions_to_fields(requirements)
        for field, requirement in zip(self.fields, requirements):
            values.append(field.values(requirement))

        if transform:
            for transform in self.transforms:
                values = transform.values(values)
        return values

    def values_jacobians(self, requirements, transform=True):
        values = []
        jacobians = []
        requirements = self.map_positions_to_fields(requirements)
        for field, requirement in zip(self.fields, requirements):
            field_values, field_jacobians = field.values_jacobians(requirement)
            values.append(field_values)
            jacobians.append(field_jacobians)

        if transform:
            for transform in self.transforms:
                values, jacobians = transform.values_jacobians(values, jacobians)
        return values, jacobians

    def append(self, other):
        if isinstance(other, FieldPoint):
            position_idx = self._find_pos_idx(other.position)
            self._field_position_idx.append(position_idx)
            if not self.values_require[position_idx].includes(other.values_require):
                self.values_require[position_idx] = self.values_require[position_idx] + other.values_require
                self._clear_cache(position_idx)
            if not self.jacobians_require[position_idx].includes(other.jacobians_require):
                self.jacobians_require[position_idx] = self.jacobians_require[position_idx] + other.jacobians_require
                self._clear_cache(position_idx)

        elif isinstance(other, MultiFieldPoint):
            self._field_position_idx.append([])
            for position, values_require, jacobians_require in zip(other.positions, other.values_require, other.jacobians_require):
                position_idx = self._find_pos_idx(position)
                self._field_position_idx[-1].append(position_idx)

                if not self.values_require[position_idx].includes(values_require):
                    self.values_require[position_idx] = self.values_require[position_idx] + values_require
                    self._clear_cache(position_idx)
                if not self.jacobians_require[position_idx].includes(jacobians_require):
                    self.jacobians_require[position_idx] = self.jacobians_require[position_idx] + jacobians_require
                    self._clear_cache(position_idx)

        else:
            raise IncompatibleFieldsError(f'Cannot append a {type(other).__name__} to a {type(self).__name__}')

        self.fields.append(other)
        return self

    def _clear_cache(self, idx=None):
        if idx is None:
            self._cached_requests = [None] * len(self.positions)
        else:
            self._cached_requests[idx] = None


class CostFunction(MultiFieldPoint):
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
        for idx, (position, values_require, jacobians_require) in enumerate(zip(self.positions, self.values_require, self.jacobians_require)):
            if self._cached_requests[idx] is None:
                self._cached_requests[idx] = self.array.request(values_require + jacobians_require, position)

        requirements = [self.evaluate_requirements(complex_transducer_amplitudes, request) for request in self._cached_requests]
        values, jacobians = self.values_jacobians(requirements)
        return values, jacobians

    def __format__(self, fmt_spec):
        base = super().__format__(fmt_spec)
        if '\n' in base or len(base) > 75:
            cost = 'Cost:\n'
        else:
            cost = 'Cost: '
        return cost + base


unbound_fields = (Field, MultiField)
bound_fields = (FieldPoint, MultiFieldPoint)


def stack(*fields):
    if len(fields) == 1 and isinstance(fields[0], collections.abc.Iterable):
        # A single input which is an iterable can and should be stacked on its own.
        fields = fields[0]
    is_bound = isinstance(fields[0], bound_fields)
    is_stackable = all([isinstance(field, bound_fields) == is_bound for field in fields])
    if not is_stackable:
        raise IncompatibleFieldsError('Cannot stack a mix of bound and unbound fields')
    if is_bound:
        return MultiFieldPoint(*fields)
    else:
        return MultiField(*fields)
