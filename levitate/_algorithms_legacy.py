import warnings


def gorkov_potential(*args, **kwargs):
    warnings.warn("""`gorkov_potential` is deprecated, use `GorkovPotential` instead.""")
    from .fields import GorkovPotential
    return GorkovPotential(*args, **kwargs)


def gorkov_gradient(*args, **kwargs):
    warnings.warn("""`gorkov_gradient` is deprecated, use `GorkovGradient` instead.""")
    from .fields import GorkovGradient
    return GorkovGradient(*args, **kwargs)


def gorkov_laplacian(*args, **kwargs):
    warnings.warn("""`gorkov_laplacian` is deprecated, use `GorkovLaplacian` instead.""")
    from .fields import GorkovLaplacian
    return GorkovLaplacian(*args, **kwargs)


def second_order_force(*args, **kwargs):
    warnings.warn("""`second_order_force` is deprecated, use `RadiationForce` instead.""")
    from .fields import RadiationForce
    return RadiationForce(*args, **kwargs)


def second_order_stiffness(*args, **kwargs):
    warnings.warn("""`second_order_stiffness` is deprecated, use `RadiationForceStiffness` instead.""")
    from .fields import RadiationForceStiffness
    return RadiationForceStiffness(*args, **kwargs)


def second_order_curl(*args, **kwargs):
    warnings.warn("""`second_order_curl` is deprecated, use `RadiationForceCurl` instead.""")
    from .fields import RadiationForceCurl
    return RadiationForceCurl(*args, **kwargs)


def second_order_force_gradient(*args, **kwargs):
    warnings.warn("""`second_order_force_gradient` is deprecated, use `RadiationForceGradient` instead.""")
    from .fields import RadiationForceGradient
    return RadiationForceGradient(*args, **kwargs)


def pressure_squared_magnitude(*args, **kwargs):
    warnings.warn("""`pressure_squared_magnitude` is deprecated, use `abs(Pressure())` instead.""")
    from .fields import Pressure
    return abs(Pressure(*args, **kwargs))


def velocity_squared_magnitude(*args, **kwargs):
    warnings.warn("""`velocity_squared_magnitude` is deprecated, use `abs(Velocity())` instead.""")
    from .fields import Velocity
    return abs(Velocity(*args, **kwargs))


def spherical_harmonics_force(*args, **kwargs):
    warnings.warn("""`spherical_harmonics_force` is deprecated, use `SphericalHarmonicsForce` instead.""")
    from .fields import SphericalHarmonicsForce
    return SphericalHarmonicsForce(*args, **kwargs)
