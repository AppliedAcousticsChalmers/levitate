import numpy as np


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


class Shift(SingleInput, Transform):
    def __init__(self, input, shift):
        super().__init__(input)
        self.shift = shift

    def values(self, values):
        return values + self.shift

    def jacobians(self, values, jacobians):
        return jacobians


class Scale(SingleInput, Transform):
    def __init__(self, input, scale):
        super().__init__(input)
        self.scale = scale

    def values(self, values):
        return values * self.scale

    def jacobians(self, values, jacobians):
        return jacobians * self.scale


class Power(SingleInput, Transform):
    def __init__(self, input, exponent):
        super().__init__(input)
        self.exponent = exponent

    def values(self, values):
        return values ** self.exponent

    def jacobians(self, values, jacobians):
        return jacobians * self.exponent * values[self._val_reshape] ** (self.exponent - 1)


class Exponential(SingleInput, Transform):
    def __init__(self, input, base):
        super().__init__(input)
        self.base = base

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
