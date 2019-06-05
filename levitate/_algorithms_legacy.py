import warnings


def gorkov_potential(*args, **kwargs):
    warnings.warn("""`gorkov_potential` is deprecated, use `GorkovPotential` instead.""")
    from .algorithms import GorkovPotential
    return GorkovPotential(*args, **kwargs)


def gorkov_gradient(*args, **kwargs):
    warnings.warn("""`gorkov_gradient` is deprecated, use `GorkovGradient` instead.""")
    from .algorithms import GorkovGradient
    return GorkovGradient(*args, **kwargs)


def gorkov_laplacian(*args, **kwargs):
    warnings.warn("""`gorkov_laplacian` is deprecated, use `GorkovLaplacian` instead.""")
    from .algorithms import GorkovLaplacian
    return GorkovLaplacian(*args, **kwargs)


def second_order_force(*args, **kwargs):
    warnings.warn("""`second_order_force` is deprecated, use `SecondOrderForce` instead.""")
    from .algorithms import SecondOrderForce
    return SecondOrderForce(*args, **kwargs)


def second_order_stiffness(*args, **kwargs):
    warnings.warn("""`second_order_stiffness` is deprecated, use `SecondOrderStiffness` instead.""")
    from .algorithms import SecondOrderStiffness
    return SecondOrderStiffness(*args, **kwargs)


def second_order_curl(*args, **kwargs):
    warnings.warn("""`second_order_curl` is deprecated, use `SecondOrderCurl` instead.""")
    from .algorithms import SecondOrderCurl
    return SecondOrderCurl(*args, **kwargs)


def second_order_force_gradient(*args, **kwargs):
    warnings.warn("""`second_order_force_gradient` is deprecated, use `SecondOrderForceGradient` instead.""")
    from .algorithms import SecondOrderForceGradient
    return SecondOrderForceGradient(*args, **kwargs)


def pressure_squared_magnitude(*args, **kwargs):
    warnings.warn("""`pressure_squared_magnitude` is deprecated, use `PressureMagnitudeSquared` instead.""")
    from .algorithms import PressureMagnitudeSquared
    return PressureMagnitudeSquared(*args, **kwargs)


def velocity_squared_magnitude(*args, **kwargs):
    warnings.warn("""`velocity_squared_magnitude` is deprecated, use `VelocityMagnitudeSquared` instead.""")
    from .algorithms import VelocityMagnitudeSquared
    return VelocityMagnitudeSquared(*args, **kwargs)


def spherical_harmonics_force(*args, **kwargs):
    warnings.warn("""`spherical_harmonics_force` is deprecated, use `SphericalHarmonicsForce` instead.""")
    from .algorithms import SphericalHarmonicsForce
    return SphericalHarmonicsForce(*args, **kwargs)
