import numpy as np
import math

class NonNumericError(TypeError):
    pass


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


class SingleInput:
    def __init__(self, input):
        self.input = input
        self._val_reshape = (slice(None),) * self.input.ndim + (None, Ellipsis)

    @property
    def shape(self):
        return self.input.shape

    @property
    def ndim(self):
        return len(self.shape)


class MultiInput:
    def __init__(self, inputs):
        self.inputs = inputs
        self._input_val_reshapes = [(slice(None),) * input.ndim + (None, Ellipsis) for input in self.inputs]

    @property
    def shape(self):
        return [input.shape for input in self.inputs]

    @property
    def ndim(self):
        return -1


class MultiInputReducer(MultiInput):
    def __init__(self, inputs):
        super().__init__(inputs)
        self._output_val_reshape = (slice(None),) * self.ndim + (None, Ellipsis)

    @property
    def shape(self):
        shapes = [input.shape for input in self.inputs]
        ndim = max(len(s) for s in shapes)
        padded_shapes = [(1,) * (ndim - len(s)) + s for s in shapes]
        out_shape = [max(s) for s in zip(*padded_shapes)]
        if not all([dim == 1 or dim == out_dim for dims, out_dim in zip(zip(*padded_shapes), out_shape) for dim in dims]):
            raise ValueError(f"Shapes {shapes} cannot be broadcast together")
        return tuple(out_shape)

    @property
    def ndim(self):
        return len(self.shape)


class Shift(SingleInput, Transform):
    def __init__(self, input, shift):
        super().__init__(input)
        self.shift = np.asarray(shift)
        if not np.issubdtype(self.shift.dtype, np.number):
            raise NonNumericError(f'Cannot shift with value {shift} of type {type(shift).__name__}')

    def values(self, values):
        return values + self.shift

    def jacobians(self, values, jacobians):
        return jacobians


class Scale(SingleInput, Transform):
    def __init__(self, input, scale):
        super().__init__(input)
        self.scale = np.asarray(scale)
        if not np.issubdtype(self.scale.dtype, np.number):
            raise NonNumericError(f'Cannot scale with value {scale} of type {type(scale).__name__}')

    def values(self, values):
        return values * self.scale

    def jacobians(self, values, jacobians):
        return jacobians * self.scale


class Power(SingleInput, Transform):
    def __init__(self, input, exponent):
        super().__init__(input)
        self.exponent = np.asarray(exponent)
        if not np.issubdtype(self.exponent.dtype, np.number):
            raise NonNumericError(f'Cannot raise to value {exponent} of type {type(exponent).__name__}')

    def values(self, values):
        return values ** self.exponent

    def jacobians(self, values, jacobians):
        return jacobians * self.exponent * values[self._val_reshape] ** (self.exponent - 1)


class Exponential(SingleInput, Transform):
    def __init__(self, input, base):
        super().__init__(input)
        self.base = np.asarray(base)
        if not np.issubdtype(self.base.dtype, np.number):
            raise NonNumericError(f'Cannot exponentiate with base {base} of type {type(base).__name__}')

    def values(self, values):
        return self.base ** values

    def values_jacobians(self, values, jacobians):
        values = self.values(values)
        jacobians = jacobians * values[self._val_reshape] * np.log(self.base)
        return values, jacobians


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


class Absolute(SingleInput, Transform):
    def values(self, values):
        return np.abs(values)

    def values_jacobians(self, values, jacobians):
        abs_values = np.abs(values)
        jacobians = jacobians * (np.conjugate(values) / abs_values)[self._val_reshape]
        return abs_values, jacobians


class Negate(SingleInput, Transform):
    def values(self, values):
        return -values

    def jacobians(self, values, jacobians):
        return -jacobians


class Real(SingleInput, Transform):
    def values(self, values):
        return np.real(values)

    def jacobians(self, values, jacobians):
        return jacobians


class Imag(SingleInput, Transform):
    def values(self, values):
        return np.imag(values)

    def jacobians(self, values, jacobians):
        if np.iscomplexobj(values):
            return -1j * jacobians
        return np.zeros_like(jacobians)


class Conjugate(SingleInput, Transform):
    def values(self, values):
        return np.conjugate(values)

    def jacobians(self, values, jacobians):
        return jacobians


class FieldSum(MultiInputReducer, Transform):
    def values(self, values):
        return sum(values)

    def jacobians(self, values, jacobians):
        return sum(jacobians)


class Product(MultiInputReducer, Transform):
    def values(self, values):
        return math.prod(values)

    def values_jacobians(self, values, jacobians):
        value_product = math.prod(values)
        log_derivs = [jac / val[val_shape] for (val, jac, val_shape) in zip(values, jacobians, self._input_val_reshapes)]
        jacobian_product = sum(log_derivs) * value_product[self._output_val_reshape]
        return value_product, jacobian_product
