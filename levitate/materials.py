"""Manages material properties.

Many functions need access to some material properties.
In order to ensure that the properties of a specific material
is the same everywhere, they are all collected in classes here.

.. admonition:: Pickling

    Pickling materials require some special consideration. In general there
    should be a single set of properties defining a material, but loading
    some data which is saved with modified properties will create a conflict.
    The newly loaded material will be in a "Local" state, using modified properties
    different from the global ones. It is recommended to resolve this by avoiding the
    problem entirely by modifying the global properties before loading the old data.
    If this is not possible or preferable, there are three functions intended to
    resolve the conflict using either the global or the local properties.

Note
----
Updating global material properties will change the properties throughout
the entire package, but some classes (notably the algorithms) pre-calculate
a lot of material-dependent properties. These properties will **NOT** be
updated after a material update. It is therefore highly recommended to
define the material properties once in the beginning of a session.

.. autosummary:
    :nosignatures:

    air
    styrofoam

"""

import warnings


class MaterialMeta(type):
    """Metaclass for materials.

    This metaclass will automatically create the properties
    defines in the `properties` variable in the class or its
    bases. The properties implement a local/global system,
    where each instance will default to use the global properties
    unless otherwise specified.
    """

    def __new__(cls, name, bases, dct):
        dct.setdefault('properties', {})
        dct['_instances'] = []
        for base in bases:
            try:
                for prop in base.properties:
                    dct['properties'].setdefault(prop, getattr(base, prop).__doc__)
            except AttributeError:
                pass

        for prop in dct['properties']:
            dct[prop] = cls.class_instance_property(prop, dct['properties'][prop])
            dct.setdefault('_' + prop, None)  # We need at least a placeholder value not to break the first instance.
        dct['properties'] = set(dct['properties'].keys())
        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def class_instance_property(name, doc=None):
        """Create a local/global property."""
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
        return property(getter, setter, doc=doc)


class Material(metaclass=MaterialMeta):
    r"""Main base class for materials.

    This class handles most of the functionality of the materials in the package.
    Each material is required to have (at least) a speed of sound and a density,
    from which the impedance and the compressibility can be calculated.
    In most cases there should only be a single instance of each material class,
    defining the properties of said material. Multiple instances might be
    created while pickling see the section below. If a new material of an
    existing material class is created with modified properties, it will
    also be created in a "Local" state.

    """

    _str_fmt_spec = '{:%name}'
    _repr_fmt_spec = '{:%name(%props)}'
    _use_global_bool = True
    properties = {
        'c': 'The (longitudinal) speed of sound in the material, in m/s.',
        'rho': 'The density of the material, in kg/m^3.',
    }

    def __init__(self, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            self.__setstate__(kwargs)
        if len(w) > 0:
            warnings.warn(w[0].message, category=w[0].category, stacklevel=2)

    @property
    def compressibility(self):
        r"""Compressibility :math:`{1 \over \rho c^2}`, non settable."""
        return 1 / (self.c**2 * self.rho)

    @property
    def impedance(self):
        r"""(Specific) Acoustic (wave) impedance :math:`\rho c`, non settable."""
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
        """Load properties from the global state.

        Useful only on materials in a local state to resolve conflicts.
        Replaces the current local properties with the global properties
        and goes to global mode, completely removing the stored values.
        """
        self._use_global = True
        for prop in self.properties:
            setattr(self, '_' + prop, getattr(self.__class__, '_' + prop))

    def push_to_global(self):
        """Push the local properties to the global state.

        Useful only on materials in a local state to resolve conflicts.
        Replaces the current global properties with the modified local
        ones, completely overriding the global properties for all global
        instances.
        """
        self._use_global = True
        for prop in self.properties:
            setattr(self.__class__, '_' + prop, getattr(self, '_' + prop))

    @classmethod
    def force_all_to_global(cls):
        """Force all instances of this material to use global properties.

        Useful to resolve material conflicts by choosing the global state
        for all instances of the material. Will never change the global
        properties, even if called from a locally modified instance.
        """
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
    """Base class for ideal gases.

    All ideal gases can determine the wave speed and density from
    the ambient temperature and pressure using more basic material
    constants, see `update_properties`.
    """

    def update_properties(self, temperature=None, pressure=None):
        r"""Update the material properties with the ambient conditions.

        Sets the material properties of air according to

        .. math::
            c &= \sqrt{\gamma R T}\\
            \rho &= {P \over R T}

        where :math:`T` is the ambient temperature in Kelvin, :math:`P` is the
        ambient pressure, :math:`\gamma` is the adiabatic index, and :math:`R`
        is the specific gas constant for the gas.

        Parameters
        ----------
        temperature : float
            The ambient temperature, in degrees Celsius. Defaults to 20.
        pressure : float
            The static ambient air pressure, in Pa. Defaults to 101325.

        """
        temperature = temperature if temperature is not None else 20
        pressure = pressure if pressure is not None else 101325
        self.c = (self._gamma * self._R_spec * (temperature + 273.15))**0.5
        self.rho = pressure / self._R_spec / (temperature + 273.15)


class Solid(Material):
    """Base class for elastic solids.

    Solids can support shear waves, which is important for some
    scattering problems.
    """

    properties = {'poisson_ratio': "Poisson's ratio, related to shear wave speed."}

    @property
    def c_transversal(self):
        r"""Transversal wave speed.

        The speed of sound for transversal waves, i.e. shear waves.
        Calculated as :math:`c\sqrt{{1-2\nu}\over{2-2\nu}}`, where
        :math:`\nu` is the Poisson's ratio.
        """
        nu = self.poisson_ratio
        return self.c * (0.5 * (1 - 2 * nu) / (1 - nu))**0.5


class Air(Gas):
    """Properties of air.

    Has default values::

        c = 343.2367605312694
        rho = 1.2040847588826422
    """

    _R_spec = 287.05864074988347
    _gamma = 1.4


class Styrofoam(Solid):
    """Properties of styrofoam.

    Has default values::

        c = 2350
        rho = 25
        poisson_ratio = 0.35
    """

    _c = 2350
    _rho = 25
    _poisson_ratio = 0.35


air = Air()
air.update_properties()  # Default values for air are calculated, not hardcoded.
styrofoam = Styrofoam()
