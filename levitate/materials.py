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

    Air
    Styrofoam

"""


class Material(type):
    r"""Metaclass for materials.

    Essentially acts like the class of the materials, if the classes are
    seen as objects. The distinction is made mostly due to pickling and
    binding of methods.

    Attributes
    ----------
    c : float
        The speed of sound in the material
    rho : float
        The density of the material
    compressibility : float
        Compressibility :math:`{1 \over \rho c^2}`, non settable.

    """

    _str_fmt_spec = '{:%name}'
    _repr_fmt_spec = '{:%cls(name=%name, c=%c, rho=%rho)}'

    @property
    def compressibility(cls):
        return 1 / (cls.c**2 * cls.rho)

    @property
    def name(cls):
        return cls.__name__

    def __format__(cls, fmt_spec):
        return fmt_spec.replace('%cls', cls.__class__.__name__).replace('%name', cls.name).replace('%c', str(cls.c)).replace('%rho', str(cls.rho))

    def __str__(cls):
        return cls._str_fmt_spec.format(cls)

    def __repr__(cls):
        return cls._repr_fmt_spec.format(cls)

    def _repr_pretty_(cls, p, cycle):
        p.text(str(cls))


class Air(metaclass=Material):
    """Properties of air.

    Has default values::

        c = 343.2367605312694
        rho = 1.2040847588826422
    """

    c = 343.2367605312694
    rho = 1.2040847588826422

    @classmethod
    def update_properties(cls, temperature=None, pressure=None):
        r"""Update properties of air.

        Sets the material properties of air according to

        .. math::
            c &= \sqrt{\gamma R T}\\
            \rho &= {P \over R T}

        where :math:`T` is the ambient temperature in Kelvin, :math:`P` is the ambient
        pressure, :math:`\gamma=1.4` is the adiabatic index, and :math:`R=287.058` J/(kg K)
        is the specific gas constant.

        Parameters
        ----------
        temperature : float
            The ambient temperature, in degrees Celsius. Defaults to 20.
        pressure : float
            The static ambient air pressure, in Pa. Defaults to 101325.

        """
        R_spec = 287.058
        gamma = 1.4
        temperature = temperature or 20
        pressure = pressure or 101325
        cls.c = (gamma * R_spec * (temperature + 273.15))**0.5
        cls.rho = pressure / R_spec / (temperature + 273.15)


class Styrofoam(metaclass=Material):
    """Properties of styrofoam.

    Has default values::

        c = 2350
        rho = 25
    """

    c = 2350
    rho = 25
