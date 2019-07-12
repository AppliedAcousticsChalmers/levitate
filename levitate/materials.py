"""Manages material properties.

Many functions need access to some material properties.
In order to ensure that the properties of a specific material
is the same everywhere, they are all collected in classes here.

Warning
-------
When pickling objects in the package, any changes made to
material properties will not be saved.

.. autosummary:
    :nosignatures:

    air
    styrofoam

"""

import warnings


class MaterialMeta(type):
    def __new__(cls, name, bases, dct):
        dct.setdefault('properties', set())
        dct['_instances'] = []
        for base in bases:
            try:
                dct['properties'] |= base.properties
            except AttributeError:
                pass

        for prop in dct['properties']:
            dct[prop] = cls.class_instance_property(prop)
            # dct[prop] = ClassInstanceProperty(prop)
            dct.setdefault('_' + prop, None)  # We need at least a placeholder value not to break the first instance.
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def class_instance_property(name):
        key = '_' + name
        def getter(self):
            if self._use_global:
                return getattr(self.__class__, key)
            else:
                return getattr(self, key)
        def setter(self, val):
            if self._use_global:
                return setattr(self.__class__, key, val)
            else:
                return setattr(self, key, val)
        return property(getter, setter)


class Material(metaclass=MaterialMeta):
    _str_fmt_spec = '{:%name}'
    _repr_fmt_spec = '{:%name(%props)}'
    _use_global_bool = True
    properties = {'c', 'rho'}

    def __init__(self, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            self.__setstate__(kwargs)
        if len(w) > 0:
            warnings.warn(w[0].message, category=w[0].category, stacklevel=2)

    @property
    def compressibility(self):
        return 1 / (self.c**2 * self.rho)

    @property
    def impedance(self):
        return self.rho * self.c

    @property
    def _use_global(self):
        return self._use_global_bool

    @_use_global.setter
    def _use_global(self, val):
        if val is False and self._use_global is True:
            warnings.warn(
                'Material `{}` with modified properties created, '
                'continuing with multiple definitions of properties. '
                'It is highly recommended to resolve this with one of '
                '`load_from_global`, `push_to_global`, or `force_all_to_global`.'
                .format(self.__class__.__name__),
                stacklevel=3)
        self._use_global_bool = val

    def load_from_global(self):
        self._use_global = True
        for prop in self.properties:
            setattr(self, '_' + prop, getattr(self.__class__, '_' + prop))

    def push_to_global(self):
        self._use_global = True
        for prop in self.properties:
            setattr(self.__class__, '_' + prop, getattr(self, '_' + prop))

    @classmethod
    def force_all_to_global(cls):
        for instance in cls._instances:
            instance._use_global = True

    def __eq__(self, other):
        return type(self) == type(other) and all(getattr(self, prop) == getattr(other, prop) for prop in self.properties)

    def __getstate__(self):
        return {key: getattr(self, key) for key in self.properties}

    def __setstate__(self, state):
        if (len(self.__class__._instances) > 0  # The first instance must be global and we don't want to check the other conditions
            and any(getattr(self.__class__, '_' + prop) != state[prop]  # Only use local if the material properties differ from global
                    for prop in self.properties & state.keys())):  # Check supplied properties if they are an actual property, ignore the rest.
                self._use_global = False
        for prop in self.properties & state.keys():
            # Set all supplied properties to the supplied value, defaulting to the current value.
            # The current value will default to class globals for new instances.
            setattr(self, prop, state.get(prop, getattr(self, prop)))
        self.__class__._instances.append(self)

    def __format__(self, fmt_spec):
        prop_str = ''.join('{}={}, '.format(prop, getattr(self, prop)) for prop in self.properties).rstrip(', ')
        global_str = '' if self._use_global is True else 'Local '
        name_str = global_str + self.__class__.__name__
        return fmt_spec.replace('%name', name_str).replace('%props', prop_str)

    def __str__(cls):
        return cls._str_fmt_spec.format(cls)

    def __repr__(cls):
        return cls._repr_fmt_spec.format(cls)

    def _repr_pretty_(cls, p, cycle):
        p.text(repr(cls))


class Gas(Material):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_properties(self, temperature=None, pressure=None):
        temperature = temperature if temperature is not None else 20
        pressure = pressure if pressure is not None else 101325
        self.c = (self._gamma * self._R_spec * (temperature + 273.15))**0.5
        self.rho = pressure / self._R_spec / (temperature + 273.15)


class Solid(Material):
    properties = {'poisson_ratio'}

    @property
    def c_transversal(self):
        nu = self.poisson_ratio
        return self.c * (0.5 * (1 - 2 * nu) / (1 - nu))**0.5


class Air(Gas):
    _R_spec = 287.05864074988347
    _gamma = 1.4


class Styrofoam(Solid):
    _c = 2350
    _rho = 25
    _poisson_ratio = 0.35


air = Air()
air.update_properties()  # Default values for air are calculated, not hardcoded.
styrofoam = Styrofoam()
