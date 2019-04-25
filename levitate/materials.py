class Material:
    _str_fmt_spec = '{:%name}'
    _repr_fmt_spec = '{:%cls(name=%name, c=%c, rho=%rho)}'
    def __init__(self, c, rho, name):
        self.c = c
        self.rho = rho
        self.name = name

    @property
    def compressibility(self):
        return 1 / (self.c**2 * self.rho)

    def __format__(self, fmt_spec):
        return fmt_spec.replace('%cls', self.__class__.__name__).replace('%name', self.name).replace('%c', str(self.c)).replace('%rho', str(self.rho))

    def __str__(self):
        return self._str_fmt_spec.format(self)

    def __repr__(self):
        return self._repr_fmt_spec.format(self)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


Air = Material(c=343.2367605312694, rho=1.2040847588826422, name='Air')


def _update_properties(temperature=None, pressure=None):
    R_spec = 287.058
    gamma = 1.4
    temperature = temperature or 20
    pressure = pressure or 101325
    Air.c = (gamma * R_spec * (temperature + 273.15))**0.5
    Air.rho = pressure / R_spec / (temperature + 273.15)


Air.update_properties = _update_properties

Styrofoam = Material(c=2350, rho=25, name='Styrofoam')
