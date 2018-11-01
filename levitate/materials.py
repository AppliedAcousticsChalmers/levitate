class Air:
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


class Styrofoam:
    c = 2350
    rho = 25
