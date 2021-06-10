import numpy as np
import math


class InvalidParameterError(TypeError):
    pass


class IncompatibleShapeError(ValueError):
    pass


class DomainError(ValueError):
    pass


def broadcast_shapes(*shapes):
    ndim = max(len(s) for s in shapes)
    padded_shapes = [(1,) * (ndim - len(s)) + s for s in shapes]
    out_shape = [max(s) for s in zip(*padded_shapes)]
    if not all([dim == 1 or dim == out_dim for dims, out_dim in zip(zip(*padded_shapes), out_shape) for dim in dims]):
        raise IncompatibleShapeError(f"Input shapes {shapes} cannot be broadcast together")
    return tuple(out_shape)


class Transform:
    def __init_subclass__(cls):
        super().__init_subclass__()
        has_values = hasattr(cls, 'values')
        has_jacobians = hasattr(cls, 'jacobians')
        has_values_jacobians = hasattr(cls, 'values_jacobians')
        if all([has_values, has_jacobians, has_values_jacobians]):
            return
        if not has_values:
            raise TypeError(f'Class {cls.__name__} does not implement necessary values method')
        if has_values_jacobians:
            cls.jacobians = Transform._jacobians
            return
        if has_jacobians:
            cls.values_jacobians = Transform._values_jacobians
            return
        raise TypeError(f'Class {cls.__name__} does not implement one of jacobians or values_jacobians')

    def _jacobians(self, values, jacobians):
        return self.values_jacobians(values, jacobians)[1]

    def _values_jacobians(self, values, jacobians):
        return self.values(values), self.jacobians(values, jacobians)

    def _transform_str(self, input_str):
        return input_str

    def __eq__(self, other):
        return type(self) == type(other)


class SingleInput:
    def __init__(self, input):
        self.input = input
        if self.input.ndim is None:
            raise IncompatibleShapeError(f'Cannot use multi output object of type {type(self.input).__name__} as input to single input transform')
        self._val_reshape = (slice(None),) * self.input.ndim + (None, Ellipsis)

    @property
    def shape(self):
        return self.input.shape

    @property
    def ndim(self):
        return len(self.shape)


class MultiInput:
    def __init__(self, input):
        self.input = input
        if self.input.ndim is not None:
            raise IncompatibleShapeError(f'Cannot use single output object of type {type(self.input).__name__} as input to multi input transform')
        self._input_val_reshapes = [(slice(None),) * input.ndim + (None, Ellipsis) for input in self.input]

    @property
    def shape(self):
        return self.input.shape

    @property
    def ndim(self):
        return None


class MultiInputReducer(MultiInput):
    def __init__(self, input):
        super().__init__(input)
        self._output_val_reshape = (slice(None),) * self.ndim + (None, Ellipsis)

    @property
    def shape(self):
        return broadcast_shapes(*[input.shape for input in self.input])

    @property
    def ndim(self):
        return len(self.shape)


class Shift(SingleInput, Transform):
    def __init__(self, input, shift):
        self.shift = np.asarray(shift)
        super().__init__(input)
        if not np.issubdtype(self.shift.dtype, np.number):
            raise InvalidParameterError(f'Cannot shift with value {shift} of type {type(shift).__name__}')
        self.ndim  # Checks that the shapes are compatible

    def values(self, values):
        return values + self.shift

    def jacobians(self, values, jacobians):
        return jacobians

    @property
    def shape(self):
        return broadcast_shapes(self.shift.shape, self.input.shape)

    def _transform_str(self, input_str):
        return f'({input_str} + {self.shift})'

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.shift, other.shift)


class Scale(SingleInput, Transform):
    def __init__(self, input, scale):
        super().__init__(input)
        self.scale = np.asarray(scale)
        if not np.issubdtype(self.scale.dtype, np.number):
            raise InvalidParameterError(f'Cannot scale with value {scale} of type {type(scale).__name__}')
        if np.issubdtype(self.scale.dtype, np.complex):
            raise InvalidParameterError(f'Cannot scale with complex value {scale}')
        self.ndim  # Checks that the shapes are compatible

    def values(self, values):
        return values * self.scale

    def jacobians(self, values, jacobians):
        return jacobians * self.scale

    @property
    def shape(self):
        return broadcast_shapes(self.scale.shape, self.input.shape)

    def _transform_str(self, input_str):
        return f'({input_str} * {self.scale})'

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.scale, other.scale)


class Power(SingleInput, Transform):
    def __init__(self, input, exponent):
        super().__init__(input)
        self.exponent = np.asarray(exponent)
        if not np.issubdtype(self.exponent.dtype, np.number):
            raise InvalidParameterError(f'Cannot raise to value {exponent} of type {type(exponent).__name__}')
        self.ndim  # Checks that the shapes are compatible

    def values(self, values):
        with np.errstate(invalid='raise'):
            try:
                return values ** self.exponent
            except FloatingPointError:
                raise DomainError('Cannot take a non-integer exponent of a negative base')

    def jacobians(self, values, jacobians):
        with np.errstate(invalid='raise'):
            try:
                return jacobians * self.exponent * values[self._val_reshape] ** (self.exponent - 1)
            except FloatingPointError:
                raise DomainError('Cannot take a non-integer exponent of a negative base')

    @property
    def shape(self):
        return broadcast_shapes(self.exponent.shape, self.input.shape)

    def _transform_str(self, input_str):
        return f'({input_str} ** {self.exponent})'

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.exponent, other.exponent)


class Exponential(SingleInput, Transform):
    def __init__(self, input, base):
        super().__init__(input)
        self.base = np.asarray(base)
        if not np.issubdtype(self.base.dtype, np.number):
            raise InvalidParameterError(f'Cannot exponentiate with base {base} of type {type(base).__name__}')
        if np.issubdtype(self.base.dtype, np.complex):
            raise InvalidParameterError(f'Cannot exponentiate complex value {base} to a field')
        if np.min(self.base) < 0:
            raise DomainError(f'Cannot use negative base {base} for exponentiation')
        self.ndim  # Checks that the shapes are compatible

    def values(self, values):
        with np.errstate(invalid='raise'):
            try:
                return self.base ** values
            except FloatingPointError:
                raise DomainError('Cannot take a non-integer exponent of a negative base')

    def values_jacobians(self, values, jacobians):
        values = self.values(values)
        with np.errstate(invalid='raise'):
            try:
                log_base = np.log(self.base)
            except FloatingPointError:
                raise DomainError('Cannot take a non-integer exponent of a negative base')
        jacobians = jacobians * values[self._val_reshape] * log_base
        return values, jacobians

    @property
    def shape(self):
        return broadcast_shapes(self.base.shape, self.input.shape)

    def _transform_str(self, input_str):
        return f'({self.base} ** {input_str})'

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.base, other.base)


class ComponentSum(SingleInput, Transform):
    def __init__(self, input, axis=None):
        super().__init__(input)
        try:
            self.axis = tuple(axis)
        except TypeError:
            if axis is None:
                self.axis = tuple(range(self.input.ndim))
            else:
                self.axis = (axis,)

    def values(self, values):
        return np.sum(values, axis=self.axis)

    def jacobians(self, values, jacobians):
        return np.sum(jacobians, axis=self.axis)

    @property
    def shape(self):
        return tuple(s for ax, s in enumerate(self.input.shape) if ax not in self.axis)

    def _transform_str(self, input_str):
        return f'sum({input_str})'


class Absolute(SingleInput, Transform):
    def values(self, values):
        return np.abs(values)

    def values_jacobians(self, values, jacobians):
        abs_values = np.abs(values)
        jacobians = jacobians * (np.conjugate(values) / abs_values)[self._val_reshape]
        return abs_values, jacobians

    def _transform_str(self, input_str):
        return f'abs({input_str})'


class Negate(SingleInput, Transform):
    def values(self, values):
        return -values

    def jacobians(self, values, jacobians):
        return -jacobians

    def _transform_str(self, input_str):
        return f'-{input_str}'


class Real(SingleInput, Transform):
    def values(self, values):
        return np.real(values)

    def jacobians(self, values, jacobians):
        return jacobians

    def _transform_str(self, input_str):
        return f'real({input_str})'


class Imag(SingleInput, Transform):
    def values(self, values):
        return np.imag(values)

    def jacobians(self, values, jacobians):
        if np.iscomplexobj(values):
            return -1j * jacobians
        return np.zeros_like(jacobians)

    def _transform_str(self, input_str):
        return f'imag({input_str})'


class Conjugate(SingleInput, Transform):
    def values(self, values):
        return np.conjugate(values)

    def jacobians(self, values, jacobians):
        return jacobians

    def _transform_str(self, input_str):
        return f'conj({input_str})'


class EigenvalueSum(SingleInput, Transform):
    # The initial idea was to just return the eigenvalues as they are.
    # However, it turned out that something is not working with the jacobians
    # when the eigenvalues are complex. Summing the eigenvalues and the
    # corresponding jacobians seems to solve this for some magic reason...
    # It is probably related to that the sum of the eigenvalues is always real,
    # since the complex eigenvalues always appear in conjugated pairs.
    # The intention with this was originally to sum the real parts anyhow, so
    # this was deemed an acceptable compromise for the time being.
    def __init__(self, input):
        super().__init__(input)
        if self.input.ndim != 2 or self.input.shape[0] != self.input.shape[1]:
            raise IncompatibleShapeError(f'Cannot calculate the eigenvalues of an input of shape {self.input.shape}')

    @property
    def shape(self):
        return ()

    def values(self, values):
        values = np.moveaxis(values, [0, 1], [-2, 1])
        values = np.linalg.eigvals(values)
        values = np.moveaxis(values, -1, 0)
        return np.sum(values, axis=0).real
        # return np.sort(values, axis=0)

    def values_jacobians(self, values, jacobians):
        values_moved = np.moveaxis(values, [0, 1], [-2, -1])  # Move the matrices to the last dimensions, which eig expects.
        values_transposed = np.moveaxis(values, [0, 1], [-1, -2])  # The same but for transposed matrices.

        evals_right, evecs_right = np.linalg.eig(values_moved)
        evals_left, evecs_left = np.linalg.eig(values_transposed)

        # Move the data back to have eigenvalue index first, then component of the eigenvectors
        evals_right = np.moveaxis(evals_right, -1, 0)
        evals_left = np.moveaxis(evals_left, -1, 0)
        evecs_right = np.moveaxis(evecs_right, [-2, -1], [1, 0])
        evecs_left = np.moveaxis(evecs_left, [-2, -1], [1, 0])

        # Sort the values in ascending order.
        # We need to sort the left and right to match, and sorting both in ascending order
        # using fast algorithms is cheaper than what we could implement here.
        idx_right = np.argsort(evals_right, axis=0)
        idx_left = np.argsort(evals_left, axis=0)
        evals = np.choose(idx_right, evals_right)
        evecs_right_sorted = np.choose(idx_right[:, None], evecs_right)
        evecs_left_sorted = np.choose(idx_left[:, None], evecs_left)

        norms = 1 / np.einsum('pi...,pi...->p...', evecs_left_sorted, evecs_right_sorted)
        eigen_jacobians = np.einsum('pi...,pj...,ij...,p...->p...', evecs_left_sorted, evecs_right_sorted, jacobians, norms)
        return np.sum(evals, axis=0).real, np.sum(eigen_jacobians, axis=0)
        # return evals, eigen_jacobians

    def _transform_str(self, input_str):
        return f'sum(eigenvalues({input_str}))'


class FieldSum(MultiInputReducer, Transform):
    def values(self, values):
        return sum(values)

    def jacobians(self, values, jacobians):
        return sum(jacobians)

    def _transform_str(self, input_str):
        return f'sum({input_str})'


class Product(MultiInputReducer, Transform):
    def values(self, values):
        return np.prod(values)

    def values_jacobians(self, values, jacobians):
        value_product = np.prod(values)
        log_derivs = [jac / val[val_shape] for (val, jac, val_shape) in zip(values, jacobians, self._input_val_reshapes)]
        jacobian_product = sum(log_derivs) * value_product[self._output_val_reshape]
        return value_product, jacobian_product

    def _transform_str(self, input_str):
        return f'product({input_str})'
