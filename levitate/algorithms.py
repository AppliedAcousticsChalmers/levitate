"""A collection of levitation related mathematical implementations."""

import numpy as np
import functools
from . import materials


def requires(**requirements):
    # This function must define an actual decorator which gets the defined cost function as input.
    # TODO: Document the list of possible choises, and compare the input to what is possible.
    def wrapper(func):
        # This function will be called at load time, i.e. when the levitate code is imported
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Save the weights and requirements as attribures in the function, since the cost_function_point needs it.
            return func(*args, **kwargs)
        wrapped.requires = requirements
        # cost_function_defer_initialization.requires = requirements
        return wrapped
    return wrapper
