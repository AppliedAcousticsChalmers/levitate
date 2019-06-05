class Material(type):
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
    c = 343.2367605312694
    rho = 1.2040847588826422

    @classmethod
    def update_properties(cls, temperature=None, pressure=None):
        R_spec = 287.058
        gamma = 1.4
        temperature = temperature or 20
        pressure = pressure or 101325
        cls.c = (gamma * R_spec * (temperature + 273.15))**0.5
        cls.rho = pressure / R_spec / (temperature + 273.15)


class Styrofoam(metaclass=Material):
    c = 2350
    rho = 25
