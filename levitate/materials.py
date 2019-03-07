class Material:
    def __init__(self, c, rho):
        self.c = c
        self.rho = rho

    @property
    def compressibility(self):
        return 1 / (self.c**2 * self.rho)


Air = Material(c=343.2367605312694, rho=1.2040847588826422)


def _update_properties(temperature=None, pressure=None):
    R_spec = 287.058
    gamma = 1.4
    temperature = temperature or 20
    pressure = pressure or 101325
    Air.c = (gamma * R_spec * (temperature + 273.15))**0.5
    Air.rho = pressure / R_spec / (temperature + 273.15)


Air.update_properties = _update_properties

Styrofoam = Material(c=2350, rho=25)
