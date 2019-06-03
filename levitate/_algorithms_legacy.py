import warnings


def gorkov_potential(*args, **kwargs):
    warnings.warn("""`gorkov_potential` is deplrecated, use `GorkovPotential` instead.""")
    from .algorithms import GorkovPotential
    return GorkovPotential(*args, **kwargs)


def gorkov_gradient(*args, **kwargs):
    warnings.warn("""`gorkov_gradient` is deplrecated, use `GorkovGradient` instead.""")
    from .algorithms import GorkovGradient
    return GorkovGradient(*args, **kwargs)


def gorkov_laplacian(*args, **kwargs):
    warnings.warn("""`gorkov_laplacian` is deplrecated, use `GorkovLaplacian` instead.""")
    from .algorithms import GorkovLaplacian
    return GorkovLaplacian(*args, **kwargs)


def second_order_force(*args, **kwargs):
    warnings.warn("""`second_order_force` is deplrecated, use `SecondOrderForce` instead.""")
    from .algorithms import SecondOrderForce
    return SecondOrderForce(*args, **kwargs)


def second_order_stiffness(*args, **kwargs):
    warnings.warn("""`second_order_stiffness` is deplrecated, use `SecondOrderStiffness` instead.""")
    from .algorithms import SecondOrderStiffness
    return SecondOrderStiffness(*args, **kwargs)


def second_order_curl(*args, **kwargs):
    warnings.warn("""`second_order_curl` is deplrecated, use `SecondOrderCurl` instead.""")
    from .algorithms import SecondOrderCurl
    return SecondOrderCurl(*args, **kwargs)


def second_order_force_gradient(*args, **kwargs):
    warnings.warn("""`second_order_force_gradient` is deplrecated, use `SecondOrderForceGradient` instead.""")
    from .algorithms import SecondOrderForceGradient
    return SecondOrderForceGradient(*args, **kwargs)


def pressure_squared_magnitude(*args, **kwargs):
    warnings.warn("""`pressure_squared_magnitude` is deplrecated, use `PressureMagnitudeSquared` instead.""")
    from .algorithms import PressureMagnitudeSquared
    return PressureMagnitudeSquared(*args, **kwargs)


def velocity_squared_magnitude(*args, **kwargs):
    warnings.warn("""`velocity_squared_magnitude` is deplrecated, use `VelocityMagnitudeSquared` instead.""")
    from .algorithms import VelocityMagnitudeSquared
    return VelocityMagnitudeSquared(*args, **kwargs)


def spherical_harmonics_force(*args, **kwargs):
    warnings.warn("""`spherical_harmonics_force` is deplrecated, use `SphericalHarmonicsForce` instead.""")
    from .algorithms import SphericalHarmonicsForce
    return SphericalHarmonicsForce(*args, **kwargs)
