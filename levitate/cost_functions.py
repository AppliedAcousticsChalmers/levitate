"""A collection of cost functions, and a minimizer for them."""

import warnings
import numpy as np
import scipy.optimize
import logging
import itertools

from .materials import Air
from .utils import num_pressure_derivs

logger = logging.getLogger(__name__)
warnings.warn("""The cost_functions module is deprecated and will be removed in a further release. Use the algorithms module or the optimization module instead.""")

def minimize(functions, array, variable_amplitudes=False,
             constrain_transducers=None, callback=None, precall=None,
             basinhopping=False, return_optim_status=False, minimize_kwargs=None,
             ):
    """Minimizes a set of cost functions.

    Each cost function should have the signature `f(phases, amplitudes)`
    where `phases` is an ndarray with the phase of each element in the array,
    and `amplitudes` is an array with the amplitude of each element in the array.
    The functions should return `value, phase_jacobian, amplitude_jacobian`
    where the two jacobians are the derivative of the value w.r.t each input.
    If the jacobians does not exist, set `minimize_kwargs['jac'] = False` and
    return only `value`.

    This function supports minimization sequences. Pass an iterable of iterables
    of cost functions to start sequenced minimization, e.g. a list of lists of
    functions.
    When using multiple cost functions, either all functions return the
    jacobians, or no functions return jacobians.
    The arguments: `variable_amplitudes`, `constrain_transducers`, `callback`,
    `precall`, `basinhopping`, and  `minimize_kwargs` can be given as single
    values or as iterables of the same length as `functions`.

    Parameters
    ----------
    functions
        The cost functions that should be minimized. A single callable, an
        iterable of callables, or an iterable of iterables of callables, as
        described above.
    array : `TransducerArray`
        The array from which the cost functions are created.
    variable_amplitudes : bool
        Toggles the usage of varying amplitudes in the minimization.
    constrain_transducers : array_like
        Specifies a number of transducers which are constant elements in the
        minimization. Will be used as the second argument in `np.delete`
    callback : callable
        A callback function which will be called after each step in sequenced
        minimization. Return false from the callback to break the sequence.
        Should have the signature :
        `callback(array=array, result=result, optim_status=opt_res, idx=idx)`
    precall : callable
        Initialization function which will be called with the array phases,
        amplitudes, and the sequence index before each sequence step.
        Must return the initial phases and amplitudes for the sequence step.
        Default sets the phases and amplitudes to the solution of the previous
        sequence step, or the original state for the first iteration.
        Should have the signature :
        `precall(phases, amplitudes, idx)`
    basinhopping : bool or int
        Specifies if basinhopping should be used. Pass an int to specify the
        number of basinhopping iterations, or True to use default value.
    return_optim_status : bool
        Toggles the `optim_status` output.
    minimize_kwargs : dict
        Extra keyword arguments which will be passed to `scipy.minimize`.

    Returns
    -------
    result : `ndarray`
        The array phases and amplitudes after minimization.
        Stacks sequenced result in the first dimension.
    optim_status : `OptimizeResult`
        Scipy optimization result structure. Optional output,
        toggle with the corresponding input argument.


    """
    if constrain_transducers is None or constrain_transducers is False:
        constrain_transducers = []
    # Handle single function input case
    try:
        iter(functions)
    except TypeError:
        functions = [functions]
    # Check if we should do sequenced optimization
    try:
        iter(next(iter(functions)))
    except TypeError:
        # =================================================
        # Single minimization of cost functions start here!
        # =================================================
        unconstrained_transducers = np.delete(np.arange(array.num_transducers), constrain_transducers)
        num_unconstrained_transducers = len(unconstrained_transducers)

        call_phases = array.phases.copy()
        call_amplitudes = array.amplitudes.copy()
        if variable_amplitudes:
            start = np.concatenate([call_phases[unconstrained_transducers], call_amplitudes[unconstrained_transducers]]).copy()
        else:
            start = call_phases[unconstrained_transducers].copy()

        bounds = [(None, None)] * num_unconstrained_transducers
        if variable_amplitudes:
            bounds += [(1e-3, 1)] * num_unconstrained_transducers
        opt_args = {'jac': True, 'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'gtol': 1e-9, 'ftol': 1e-15}}
        if minimize_kwargs is not None:
            opt_args.update(minimize_kwargs)

        if opt_args['jac']:
            def func(phases_amplitudes):
                call_phases[unconstrained_transducers] = phases_amplitudes[:num_unconstrained_transducers]
                if variable_amplitudes:
                    call_amplitudes[unconstrained_transducers] = phases_amplitudes[num_unconstrained_transducers:]

                results = [f(call_phases, call_amplitudes) for f in functions]
                value = sum(result[0] for result in results)
                phase_jacobian = sum(result[1] for result in results)
                amplitude_jacobian = sum(result[2] for result in results)

                if variable_amplitudes:
                    jacobian = np.concatenate([phase_jacobian[unconstrained_transducers], amplitude_jacobian[unconstrained_transducers]])
                else:
                    jacobian = phase_jacobian[unconstrained_transducers]
                return value, jacobian
        else:
            def func(phases_amplitudes):
                call_phases[unconstrained_transducers] = phases_amplitudes[:num_unconstrained_transducers]
                if variable_amplitudes:
                    call_amplitudes[unconstrained_transducers] = phases_amplitudes[num_unconstrained_transducers:]
                value = sum([f(call_phases, call_amplitudes) for f in functions])
                return value

        if basinhopping:
            if basinhopping is True:
                # It's not a number, use default value
                basinhopping = 20
            opt_result = scipy.optimize.basinhopping(func, start, T=1e-7, minimizer_kwargs=opt_args, niter=basinhopping)
        else:
            opt_result = scipy.optimize.minimize(func, start, **opt_args)

        call_phases[unconstrained_transducers] = opt_result.x[:num_unconstrained_transducers]
        if variable_amplitudes:
            call_amplitudes[unconstrained_transducers] = opt_result.x[num_unconstrained_transducers:]
        if return_optim_status:
            return call_amplitudes * np.exp(1j * call_phases), opt_result
        else:
            return call_amplitudes * np.exp(1j * call_phases)
    else:
        # ====================================================
        # Sequenced minimization of cost functions start here!
        # ====================================================
        initial_array_state = array.complex_amplitudes
        try:
            iter(variable_amplitudes)
        except TypeError:
            variable_amplitudes = itertools.repeat(variable_amplitudes)
        if callback is None:
            def callback(**kwargs): pass
        try:
            iter(callback)
        except TypeError:
            callback = itertools.repeat(callback)
        if precall is None:
            def precall(phase, amplitude, idx): return phase, amplitude
        try:
            iter(precall)
        except TypeError:
            precall = itertools.repeat(precall)
        try:
            iter(minimize_kwargs)  # Exception for None
            if type(next(iter(minimize_kwargs))) is not dict:
                raise TypeError
        except TypeError:
            minimize_kwargs = itertools.repeat(minimize_kwargs)
        try:
            next(iter(constrain_transducers))
            iter(next(iter(constrain_transducers)))
        except StopIteration:  # Empty list case
            constrain_transducers = itertools.repeat(constrain_transducers)
        except TypeError:
            # We need to make sure that we don't have a list of lists but with the first element as None or False
            first_val = next(iter(constrain_transducers))
            if first_val is not False and first_val is not None:
                constrain_transducers = itertools.repeat(constrain_transducers)
        try:
            iter(basinhopping)
        except TypeError:
            basinhopping = itertools.repeat(basinhopping)
        results = []
        opt_results = []

        for idx, (function, var_amp, const_trans, basinhop, clbck, precl, min_kwarg) in enumerate(zip(functions, variable_amplitudes, constrain_transducers, basinhopping, callback, precall, minimize_kwargs)):
            array.phases, array.amplitudes = precl(array.phases, array.amplitudes, idx)
            result, opt_res = minimize(function, array, variable_amplitudes=var_amp,
                constrain_transducers=const_trans, basinhopping=basinhop, return_optim_status=True, minimize_kwargs=min_kwarg)
            results.append(result.copy())
            opt_results.append(opt_res)
            array.complex_amplitudes = result
            if clbck(array=array, retult=result, optim_status=opt_res, idx=idx) is False:
                break

        array.complex_amplitudes = initial_array_state
        if return_optim_status:
            return np.asarray(results), opt_results
        else:
            return np.asarray(results)


class RadndomDisplacer:
    def __init__(self, num_transducers, variable_amplitude=False, stepsize=0.05):
        self.stepsize = stepsize
        self.num_transducers = num_transducers
        self.variable_amplitude = variable_amplitude

    def __call__(self, x):
        if self.variable_amplitude:
            x[:self.num_transducers] += np.random.uniform(-np.pi * self.stepsize, np.pi * self.stepsize, self.num_transducers)
            x[:self.num_transducers] = np.mod(x[:self.num_transducers] + np.pi, 2 * np.pi) - np.pi  # Don't step out of bounds, instead wrap the phase
            x[self.num_transducers:] += np.random.uniform(-self.stepsize, self.stepsize, self.num_transducers)
            x[self.num_transducers:] = np.clip(x[self.num_transducers:], 1e-3, 1)  # Don't step out of bounds!
        else:
            x += np.random.uniform(-np.pi * self.stepsize, np.pi * self.stepsize, self.num_transducers)
        return x


def vector_target(vector_calculator, target_vector=(0, 0, 0), weights=(1, 1, 1)):
    """Create a function which calculates the weighted squared difference between a target vector and a varying vector.

    This can create cost functions representing :math:`||(v - v_0)||^2_w`, i.e.
    the weighted square norm between a varying vector and a fixed vector.
    Note that the values in the weights will be squared, i.e. have the inverse
    unit compared to the vectors.

    Parameters
    ----------
    vector_calculator : callable
        A function which calculates the varying vector from array phases and
        amplitudes, along with the jacobian of said varying vector.
        This function must return `(v, dv_p, dv_a)`, where `v` is a 3 element ndarray,
        and `dv_p` and `dv_a` are shape 3xn ndarrays with the phase and amplitude
        jacobians. Suitable functions can be created by passing `False` as weights
        to other cost function generators in this module.
    target_vector : 3 element numeric, default (0, 0, 0)
        The fixed target vector, should be a 3 element ndarray or a scalar.
    weights : 3 element numeric, default (1, 1, 1)
        Specifies how the three parts should be weighted in the calculation.

    Returns
    -------
    vector_target : callable
        A function which given phases and optional amplitudes for an array
        evaluates the above equation, as well as the jacobian of said equation.
    """
    target_vector = np.asarray(target_vector)
    weights = np.asarray(weights)

    def vector_target(*args, **kwargs):
        """Weighted squared magnitude difference.

        Calculates the weighted square magnitude difference between
        a varying vector and a fixed vector.
        """
        v, dv_phase, dv_ampl = vector_calculator(*args, **kwargs)
        difference = v - target_vector
        value = np.sum(np.abs(difference * weights)**2)
        jacobian_phase = (2 * weights**2 * difference).dot(dv_phase)
        jacobian_ampl = (2 * weights**2 * difference).dot(dv_ampl)
        return value, jacobian_phase, jacobian_ampl
    return vector_target


def create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights):
    """Create weighted cost function.

    Utility function for creating cost functions based on common conventions for
    the weights and output formats. The `weights` argument specifies what the
    new function should output.

    1. `weights=None`: No weighting is done, and no calculation of the jacobians.
       The function will return the unweighted parts of the cost function.
    2. `weights=False`: No weighting is done, but the jacobians are calculated.
       The function will return `(values, phase_jacobians, amplitudes_jacobians)`
       where each part of the cost function is separated.
    3. weights is numeric array_like: The parts of the cost function and the
       jacobians are weighted and summed with their corresponding weights.
       Otherwise the outputs are as in 2).

    Parameters
    ----------
    calc_values : callable
        Function accepting the total derivatives in the field, returning
        the unweighted parts of the cost function.
    calc_jacobian : callable
        Function accepting the individual and total derivatives in the field,
        returning the unweighted jacobians of the cost function.
    spatial_derivatives : ndarray
        The spatial derivatives needed for the calculations.
    weights : array_like
        The weights for the individual parts in the cost function.

    Returns
    -------
    cost_function : callable
        A function with variable return according to weights.
    """
    import textwrap
    num_transducers = spatial_derivatives.shape[1]

    def parse_inputs(*args, **kwargs):
        """
        The input argument should be either phases AND amplitudes, OR complex amplitudes.
        The arguments can be passed as keywords or normal arguments.

        Parameters
        ----------
        phases : ndarray
            The phases of the transducer elements in the array.
        amplitudes : ndarray
            The amplitudes of the transducer elements in the array.
        complrex_amplitudes : ndarray
            The complex amplitudes, corresponding to `amplitudes * np.exp(1j * phases)`
        """
        if len(args) == 1 and np.iscomplexobj(args[0]) and len(args[0]) == num_transducers:
            # Single input with complex value
            return np.angle(args[0]), np.abs(args[0])
        elif len(args) == 2 and len(args[0]) == num_transducers and len(args[1]) == num_transducers:
            # Two inputs, should be phase, amplitude
            return args
        elif 'amplitudes' in kwargs and 'phases' in kwargs:
            # Keyword input
            return parse_inputs(kwargs['phases'], kwargs['amplitudes'])
        elif 'complex_amplitudes' in kwargs:
            return parse_inputs(kwargs['complex_amplitudes'])

    def wrapper(f):
        try:
            len(weights)
        except TypeError:
            if weights is None:
                def func(*args, **kwargs):
                    """
                    Returns
                    -------
                    values : ndarray
                        The calculated values, shape `(M, ...)` where M is the relevant number of values,
                        see above, and `...` is the shape of the positions where to calculate.
                    """
                    phases, amplitudes = parse_inputs(*args, **kwargs)
                    complex_coeff = amplitudes * np.exp(1j * phases)
                    values = calc_values(np.einsum('i,ji...->j...', complex_coeff, spatial_derivatives))
                    if values.shape[0] == 1:
                        values = values[0]
                    return values
            else:
                def func(*args, **kwargs):
                    """
                    Returns
                    -------
                    values : ndarray
                        The calculated values, shape `(M, ...)` where M is the relevant number of values,
                        see above, and `...` is the shape of the positions where to calculate.
                    phase_jacobian : ndarray
                        The jacobian of the values w.r.t the phase, shape `(M, N...)`
                        where M is the relevant number of values, see above, N is the number of transducers,
                        and `...` is the shape of the positions where to calculate.
                    amplitude_jacobian : ndarray
                        The jacobian of the values w.r.t the amplitude, shape `(M, N...)`
                        where M is the relevant number of values, see above, N is the number of transducers,
                        and `...` is the shape of the positions where to calculate.
                    """
                    phases, amplitudes = parse_inputs(*args, **kwargs)
                    complex_coeff = amplitudes * np.exp(1j * phases)
                    ind_der = np.einsum('i,ji...->ji...', complex_coeff, spatial_derivatives)
                    tot_der = np.sum(ind_der, axis=1)
                    value = calc_values(tot_der)
                    jacobian = calc_jacobian(tot_der, ind_der)
                    return value, jacobian.imag, np.einsum('i,ji...->ji...', 1 / amplitudes, jacobian.real)
        else:
            def func(*args, **kwargs):
                """
                Returns
                -------
                values : ndarray
                    The weighted sum of the values, has the same as the positions where to calculate.
                phase_jacobian : ndarray
                    The jacobian of the weighted sum of the values w.r.t the phase, shape `(N...)`
                    where N is the number of transducers, and `...` is the shape of the positions where to calculate.
                amplitude_jacobian : ndarray
                    The jacobian of the weighted sum of the values w.r.t the amplitude, shape `(N...)`
                    where N is the number of transducers, and `...` is the shape of the positions where to calculate.
                """
                phases, amplitudes = parse_inputs(*args, **kwargs)
                complex_coeff = amplitudes * np.exp(1j * phases)
                ind_der = np.einsum('i,ji...->ji...', complex_coeff, spatial_derivatives)
                tot_der = np.sum(ind_der, axis=1)
                values = calc_values(tot_der)
                jacobians = calc_jacobian(tot_der, ind_der)
                value = np.einsum('i, i...', weights, values)
                jacobian = np.einsum('i, i...', weights, jacobians)
                return value, jacobian.imag, np.einsum('i,i...->i...', 1 / amplitudes, jacobian.real)

        func.__name__ = f.__name__
        func.__qualname__ = f.__qualname__
        func.__module__ = f.__module__
        func.__doc__ = textwrap.dedent(parse_inputs.__doc__) + textwrap.dedent(func.__doc__)
        if f.__doc__ is not None:
            func.__doc__ = textwrap.dedent(f.__doc__) + textwrap.dedent(func.__doc__)
        return func
    return wrapper


def gorkov_divergence(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """Create a gorkov_divergence calculation function.

    Creates a function, which calculates the divergence and the jacobian of the field
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    Modes:
        1) weights = None: returns the x, y and z derivatives as a numpy array
        2) weights = False: returns the derivatives as an array and the jacobian as a 3 x num_transducers 2darray as a tuple
        3) else: returns the weighted sum of the divergence and the corresponding jacobian

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the divergence at
    weights : bool, (float, float, float) or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones
    c_sphere : float, optional, default 2350
        Speed of sound in Polystyrene (the material used to be floated)
    rho_sphere : float, optional, default 25
        Density of the Styrofoam balls. Note: Isn't it technically incorrect to assume
        the speed of sound in polystyrene for the whole ball while it only contains small
        amounts of it as evidenced by the density of the balls?
    radius_sphere : float, optional, default 1e-3
        Radius of the Styrofoam balls used

    Returns
    -------
    gorkov_divergence : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location, orders=2)

    V = 4 / 3 * np.pi * radius_sphere**3
    compressibility_air = 1 / (Air.rho * Air.c**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    monopole_coefficient = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (rho_sphere / Air.rho - 1) / (2 * rho_sphere / Air.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * Air.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * compressibility_air * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * Air.rho

    def calc_values(tot_der):
        Ux = (pressure_coefficient * (tot_der[1] * np.conj(tot_der[0])).real -  # Gx G
              gradient_coefficient * (tot_der[4] * np.conj(tot_der[1])).real -  # Gxx Gx
              gradient_coefficient * (tot_der[7] * np.conj(tot_der[2])).real -  # Gxy Gy
              gradient_coefficient * (tot_der[8] * np.conj(tot_der[3])).real) * 2  # Gxz Gz
        Uy = (pressure_coefficient * (tot_der[2] * np.conj(tot_der[0])).real -  # Gy G
              gradient_coefficient * (tot_der[7] * np.conj(tot_der[1])).real -  # Gxy Gx
              gradient_coefficient * (tot_der[5] * np.conj(tot_der[2])).real -  # Gyy Gy
              gradient_coefficient * (tot_der[9] * np.conj(tot_der[3])).real) * 2  # Gyz Gz
        Uz = (pressure_coefficient * (tot_der[3] * np.conj(tot_der[0])).real -  # Gz G
              gradient_coefficient * (tot_der[8] * np.conj(tot_der[1])).real -  # Gxz Gx
              gradient_coefficient * (tot_der[9] * np.conj(tot_der[2])).real -  # Gyz Gy
              gradient_coefficient * (tot_der[6] * np.conj(tot_der[3])).real) * 2  # Gzz Gz

        return np.stack((Ux, Uy, Uz), axis=0)

    def calc_jacobian(tot_der, ind_der):
        dUx = (pressure_coefficient * (tot_der[1] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[1])) -  # Gx G(i) + G Gx(i)
               gradient_coefficient * (tot_der[4] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[4])) -  # Gxx Gx(i) + Gx Gxx(i)
               gradient_coefficient * (tot_der[7] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[7])) -  # Gxy Gy(i) + Gy Gxy(i)
               gradient_coefficient * (tot_der[8] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[8]))) * 2  # Gxz Gz(i) + Gz Gxz(i)
        dUy = (pressure_coefficient * (tot_der[2] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[2])) -  # Gy G(i) + G Gy(i)
               gradient_coefficient * (tot_der[7] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[7])) -  # Gxy Gx(i) + Gx Gxy(i)
               gradient_coefficient * (tot_der[5] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[5])) -  # Gyy Gy(i) + Gy Gyy(i)
               gradient_coefficient * (tot_der[9] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[9]))) * 2  # Gyz Gz(i) + Gz Gyz(i)
        dUz = (pressure_coefficient * (tot_der[3] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[3])) -  # Gz G(i) + G Gz(i)
               gradient_coefficient * (tot_der[8] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[8])) -  # Gxz Gx(i) + Gx Gxz(i)
               gradient_coefficient * (tot_der[9] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[9])) -  # Gyz Gy(i) + Gy Gyz(i)
               gradient_coefficient * (tot_der[6] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[6]))) * 2  # Gzz Gz(i) + Gz Gzz(i)

        return np.stack((dUx, dUy, dUz), axis=0)
    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def gorkov_divergence(*args, **kwargs):
        """
        Calculates the Cartesian divergence of the Gor'kov potential.
        """
        pass
    return gorkov_divergence


def gorkov_laplacian(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """Create a gorkov_laplacian calculation function.

    Creates a function, which calculates the Laplacian and the jacobian of the field
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    Modes:
        1) weights = None: returns the x, y and z second derivatives as a numpy array
        2) weights = False: returns the second derivatives as an array and the jacobian as a 3 x num_transducers 2darray as a tuple
        3) else: returns the weighted Laplacian and the corresponding jacobian

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the laplacian at
    weights : bool, (float, float, float) or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones
    c_sphere : float, optional, default 2350
        Speed of sound in Polystyrene (the material used to be floated)
    rho_sphere : float, optional, default 25
        Density of the Styrofoam balls.
    radius_sphere : float, optional, default 1e-3
        Radius of the Styrofoam balls used

    Returns
    -------
    gorkov_laplacian : func
        The function described above
    """
    # Before defining the cost function and the jacobian, we need to initialize the following variables:
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location)

    V = 4 / 3 * np.pi * radius_sphere**3
    compressibility_air = 1 / (Air.rho * Air.c**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    monopole_coefficient = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (rho_sphere / Air.rho - 1) / (2 * rho_sphere / Air.rho + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * Air.rho)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * compressibility_air * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * Air.rho

    def calc_values(tot_der):
        Uxx = (pressure_coefficient * (tot_der[4] * np.conj(tot_der[0]) + tot_der[1] * np.conj(tot_der[1])).real -  # Gxx G + Gx Gx
               gradient_coefficient * (tot_der[10] * np.conj(tot_der[1]) + tot_der[4] * np.conj(tot_der[4])).real -  # Gxxx Gx + Gxx Gxx
               gradient_coefficient * (tot_der[13] * np.conj(tot_der[2]) + tot_der[7] * np.conj(tot_der[7])).real -  # Gxxy Gy + Gxy Gxy
               gradient_coefficient * (tot_der[14] * np.conj(tot_der[3]) + tot_der[8] * np.conj(tot_der[8])).real) * 2  # Gxxz Gz + Gxz Gxz
        Uyy = (pressure_coefficient * (tot_der[5] * np.conj(tot_der[0]) + tot_der[2] * np.conj(tot_der[2])).real -  # Gyy G + Gy Gy
               gradient_coefficient * (tot_der[15] * np.conj(tot_der[1]) + tot_der[7] * np.conj(tot_der[7])).real -  # Gyyx Gx + Gxy Gxy
               gradient_coefficient * (tot_der[11] * np.conj(tot_der[2]) + tot_der[5] * np.conj(tot_der[5])).real -  # Gyyy Gy + Gyy Gyy
               gradient_coefficient * (tot_der[16] * np.conj(tot_der[3]) + tot_der[9] * np.conj(tot_der[9])).real) * 2  # Gyyz Gz + Gyz Gyz
        Uzz = (pressure_coefficient * (tot_der[6] * np.conj(tot_der[0]) + tot_der[3] * np.conj(tot_der[3])).real -  # Gzz G + Gz Gz
               gradient_coefficient * (tot_der[17] * np.conj(tot_der[1]) + tot_der[8] * np.conj(tot_der[8])).real -  # Gzzx Gx + Gxz Gxz
               gradient_coefficient * (tot_der[18] * np.conj(tot_der[2]) + tot_der[9] * np.conj(tot_der[9])).real -  # Gzzy Gy + Gyz Gyz
               gradient_coefficient * (tot_der[12] * np.conj(tot_der[3]) + tot_der[6] * np.conj(tot_der[6])).real) * 2  # Gzzz Gz + Gzz Gzz
        return np.array((Uxx, Uyy, Uzz))

    def calc_jacobian(tot_der, ind_der):
        dUxx = (pressure_coefficient * (tot_der[4] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[4]) + 2 * tot_der[1] * np.conj(ind_der[1])) -  # Gxx G(i) + G Gxx(i) + 2 Gx Gx(i)
                gradient_coefficient * (tot_der[10] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[10]) + 2 * tot_der[4] * np.conj(ind_der[4])) -  # Gxxx Gx(i) + Gx Gxxx(i) + 2 Gxx Gxx(i)
                gradient_coefficient * (tot_der[13] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[13]) + 2 * tot_der[7] * np.conj(ind_der[7])) -  # Gxxy Gy(i) + Gy Gxxy(i) + 2 Gxy Gxy(i)
                gradient_coefficient * (tot_der[14] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[14]) + 2 * tot_der[8] * np.conj(ind_der[8]))) * 2  # Gxxz Gz(i) + Gz Gxxz(i) + 2 Gxz Gxz(i)
        dUyy = (pressure_coefficient * (tot_der[5] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[5]) + 2 * tot_der[2] * np.conj(ind_der[2])) -  # Gyy G(i) + G Gyy(i) + 2 Gy Gy(i)
                gradient_coefficient * (tot_der[15] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[15]) + 2 * tot_der[7] * np.conj(ind_der[7])) -  # Gyyx Gx(i) + Gx Gyyx(i) + 2 Gxy Gxy(i)
                gradient_coefficient * (tot_der[11] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[11]) + 2 * tot_der[5] * np.conj(ind_der[5])) -  # Gyyy Gy(i) + Gy Gyyy(i) + 2 Gyy Gyy(i)
                gradient_coefficient * (tot_der[16] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[16]) + 2 * tot_der[9] * np.conj(ind_der[9]))) * 2  # Gyyz Gz(i) + Gz Gyyz(i) + 2 Gyz Gyz(i)
        dUzz = (pressure_coefficient * (tot_der[6] * np.conj(ind_der[0]) + tot_der[0] * np.conj(ind_der[6]) + 2 * tot_der[3] * np.conj(ind_der[3])) -  # Gzz G(i) + G Gzz(i) + 2 Gz Gz(i)
                gradient_coefficient * (tot_der[17] * np.conj(ind_der[1]) + tot_der[1] * np.conj(ind_der[17]) + 2 * tot_der[8] * np.conj(ind_der[8])) -  # Gzzx Gx(i) + Gx Gzzx(i) + 2 Gxz Gxz(i)
                gradient_coefficient * (tot_der[18] * np.conj(ind_der[2]) + tot_der[2] * np.conj(ind_der[18]) + 2 * tot_der[9] * np.conj(ind_der[9])) -  # Gzzy Gy(i) + Gy Gzzy(i) + 2 Gyz Gyz(i)
                gradient_coefficient * (tot_der[12] * np.conj(ind_der[3]) + tot_der[3] * np.conj(ind_der[12]) + 2 * tot_der[6] * np.conj(ind_der[6]))) * 2  # Gzzz Gz(i) + Gz Gzzz(i) + 2 Gzz Gzz(i)
        return np.array((dUxx, dUyy, dUzz))

    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def gorkov_laplacian(*args, **kwargs):
        """
        Calculates the cartesian parts of the Laplacian of the Gor'kov potential.
        """
        pass
    return gorkov_laplacian


def second_order_force(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """Create a second_order_force calculation function.

    Creates a function, which calculates the radiation force on a sphere
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    This is more suitable than the Gor'kov formulation for use with progressive
    wave fiends, e.g. single sided arrays, see https://doi.org/10.1121/1.4773924.

    Modes:
        1) weights = None: returns the x, y and z forces as a numpy array
        2) weights = False: returns the forces as an array and the jacobian as a 3 x num_transducers 2darray as a tuple
        3) else: returns the weighted sum of forces and the corresponding jacobian

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the laplacian at
    weights : bool, (float, float, float) or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones
    c_sphere : float, optional, default 2350
        Speed of sound in Polystyrene (the material used to be floated)
    rho_sphere : float, optional, default 25
        Density of the Styrofoam balls.
    radius_sphere : float, optional, default 1e-3
        Radius of the Styrofoam balls used

    Returns
    -------
    second_order_force : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location, orders=2)

    compressibility_air = 1 / (Air.rho * Air.c**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    f_1 = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    f_2 = 2 * (rho_sphere / Air.rho - 1) / (2 * rho_sphere / Air.rho + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * compressibility_air

    def calc_values(tot_der):
        Fx = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(tot_der[1]) +  # G Gx
                                 psi_1 * tot_der[1] * np.conj(tot_der[0])) +  # Gx G
              1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[4]) +  # Gx Gxx
                                tot_der[2] * np.conj(tot_der[7]) +  # Gy Gxy
                                tot_der[3] * np.conj(tot_der[8]))  # Gz Gxz
              ).real * force_coeff
        Fy = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(tot_der[2]) +  # G Gy
                                 psi_1 * tot_der[2] * np.conj(tot_der[0])) +  # Gy G
              1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[7]) +  # Gx Gxy
                                tot_der[2] * np.conj(tot_der[5]) +  # Gy Gyy
                                tot_der[3] * np.conj(tot_der[9]))  # Gz Gyz
              ).real * force_coeff
        Fz = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(tot_der[3]) +  # G Gz
                                 psi_1 * tot_der[3] * np.conj(tot_der[0])) +  # Gz G
              1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[8]) +  # Gx Gxz
                                tot_der[2] * np.conj(tot_der[9]) +  # Gy Gyz
                                tot_der[3] * np.conj(tot_der[6]))  # Gz Gzz
              ).real * force_coeff
        return np.array((Fx, Fy, Fz))

    def calc_jacobian(tot_der, ind_der):
        dFx = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[1]) - np.conj(psi_0) * tot_der[1] * np.conj(ind_der[0]) +  # G Gx(i) - Gx G(i)
                                  psi_1 * tot_der[1] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[1])) +  # Gx G(i) - G Gx(i)
               1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[4]) - np.conj(psi_1) * tot_der[4] * np.conj(ind_der[1]) +  # Gx Gxz(i) - Gxz Gx(i)
                         psi_1 * tot_der[2] * np.conj(ind_der[7]) - np.conj(psi_1) * tot_der[7] * np.conj(ind_der[2]) +  # Gy Gxy(i) - Gxy Gy(i)
                         psi_1 * tot_der[3] * np.conj(ind_der[8]) - np.conj(psi_1) * tot_der[8] * np.conj(ind_der[3]))  # Gz Gxz(i) - Gxz Gz(i)
               ) * force_coeff
        dFy = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[2]) - np.conj(psi_0) * tot_der[2] * np.conj(ind_der[0]) +  # G Gy(i) - Gy G(i)
                                  psi_1 * tot_der[2] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[2])) +  # Gy G(i) - G Gy(i)
               1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[7]) - np.conj(psi_1) * tot_der[7] * np.conj(ind_der[1]) +  # Gx Gxy(i) - Gxy Gx(i)
                         psi_1 * tot_der[2] * np.conj(ind_der[5]) - np.conj(psi_1) * tot_der[5] * np.conj(ind_der[2]) +  # Gy Gyy(i) - Gyy Gy(i)
                         psi_1 * tot_der[3] * np.conj(ind_der[9]) - np.conj(psi_1) * tot_der[9] * np.conj(ind_der[3]))  # Gz Gyz(i) - Gyz Gz(i)
               ) * force_coeff
        dFz = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[3]) - np.conj(psi_0) * tot_der[3] * np.conj(ind_der[0]) +   # G Gz(i) - Gz G(i)
                                  psi_1 * tot_der[3] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[3])) +   # Gz G(i) - G Gz(i)
               1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[8]) - np.conj(psi_1) * tot_der[8] * np.conj(ind_der[1]) +   # Gx Gxz(i) - Gxz Gx(i)
                         psi_1 * tot_der[2] * np.conj(ind_der[9]) - np.conj(psi_1) * tot_der[9] * np.conj(ind_der[2]) +   # Gy Gyz(i) - Gyz Gy(i)
                         psi_1 * tot_der[3] * np.conj(ind_der[6]) - np.conj(psi_1) * tot_der[6] * np.conj(ind_der[3]))   # Gz Gzz(i) - Gzz Gz(i)
               ) * force_coeff
        return np.array((dFx, dFy, dFz))
    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def second_order_force(*args, **kwargs):
        """
        Calculates the radiation force accounting for both standing and travelling waves.
        """
        pass
    return second_order_force


def second_order_stiffness(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """Create a second_order_stiffness calculation function.

    Creates a function, which calculates the radiation stiffness on a sphere
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    This is more suitable than the Gor'kov formulation for use with progressive
    wave fiends, e.g. single sided arrays, see https://doi.org/10.1121/1.4773924.

    Modes:
        1) weights = None: returns the x, y and z stiffness as a numpy array
        2) weights = False: returns the stiffnesses as an array and the jacobian as a 3 x num_transducers 2darray as a tuple
        3) else: returns the weighted the stiffness and the corresponding jacobian

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the laplacian at
    weights : bool, (float, float, float) or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones
    c_sphere : float, optional, default 2350
        Speed of sound in Polystyrene (the material used to be floated)
    rho_sphere : float, optional, default 25
        Density of the Styrofoam balls.
    radius_sphere : float, optional, default 1e-3
        Radius of the Styrofoam balls used

    Returns
    -------
    second_order_stiffness : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location, orders=3)

    compressibility_air = 1 / (Air.rho * Air.c**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    f_1 = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    f_2 = 2 * (rho_sphere / Air.rho - 1) / (2 * rho_sphere / Air.rho + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * compressibility_air

    def calc_values(tot_der):
        Fxx = (1j * array.k**2 * (psi_0 * (tot_der[0] * np.conj(tot_der[4]) + tot_der[1] * np.conj(tot_der[1])) +  # G Gxx + Gx Gx
                                  psi_1 * (tot_der[4] * np.conj(tot_der[0]) + tot_der[1] * np.conj(tot_der[1]))) +  # Gxx G + Gx Gx
               1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[10]) + tot_der[4] * np.conj(tot_der[4]) +  # Gx Gxxx + Gxx Gxx
                                 tot_der[2] * np.conj(tot_der[13]) + tot_der[7] * np.conj(tot_der[7]) +  # Gy Gxxy + Gxy Gxy
                                 tot_der[3] * np.conj(tot_der[14]) + tot_der[8] * np.conj(tot_der[8]))  # Gz Gxxz + Gxz Gxz
               ).real * force_coeff
        Fyy = (1j * array.k**2 * (psi_0 * (tot_der[0] * np.conj(tot_der[5]) + tot_der[2] * np.conj(tot_der[2])) +  # G Gyy + Gy Gy
                                  psi_1 * (tot_der[5] * np.conj(tot_der[0]) + tot_der[2] * np.conj(tot_der[2]))) +  # Gyy G + Gy Gy
               1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[15]) + tot_der[7] * np.conj(tot_der[7]) +  # Gx Gyyx + Gxy Gxy
                                 tot_der[2] * np.conj(tot_der[11]) + tot_der[5] * np.conj(tot_der[5]) +  # Gy Gyyy + Gyy Gyy
                                 tot_der[3] * np.conj(tot_der[16]) + tot_der[9] * np.conj(tot_der[9]))  # Gz Gyyz + Gyz Gyz
               ).real * force_coeff
        Fzz = (1j * array.k**2 * (psi_0 * (tot_der[0] * np.conj(tot_der[6]) + tot_der[3] * np.conj(tot_der[3])) +  # G Gzz + Gz Gz
                                  psi_1 * (tot_der[6] * np.conj(tot_der[0]) + tot_der[3] * np.conj(tot_der[3]))) +  # Gzz G + Gz Gz
               1j * 3 * psi_1 * (tot_der[1] * np.conj(tot_der[17]) + tot_der[8] * np.conj(tot_der[8]) +  # Gx Gzzx + Gxz Gxz
                                 tot_der[2] * np.conj(tot_der[18]) + tot_der[9] * np.conj(tot_der[9]) +  # Gy Gzzy + Gyz Gyz
                                 tot_der[3] * np.conj(tot_der[12]) + tot_der[6] * np.conj(tot_der[6]))  # Gz Gzzx + Gzz Gzz
               ).real * force_coeff
        return np.array((Fxx, Fyy, Fzz))

    def calc_jacobian(tot_der, ind_der):
        dFxx = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[4]) - np.conj(psi_0) * tot_der[4] * np.conj(ind_der[0]) + (psi_0 - np.conj(psi_0)) * tot_der[1] * np.conj(ind_der[1]) +  # G Gxx(i) - Gxx G(i) + Gx Gx(i)
                                   psi_1 * tot_der[4] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[4]) + (psi_1 - np.conj(psi_1)) * tot_der[1] * np.conj(ind_der[1])) +  # Gxx G(i) - G Gxx(i) + Gx Gx(i)
                1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[10]) - np.conj(psi_1) * tot_der[10] * np.conj(ind_der[1]) + (psi_1 - np.conj(psi_1)) * tot_der[4] * np.conj(ind_der[4]) +  # Gx Gxxx(i) - Gxxx Gx(i) + Gxx Gxx(i)
                          psi_1 * tot_der[2] * np.conj(ind_der[13]) - np.conj(psi_1) * tot_der[13] * np.conj(ind_der[2]) + (psi_1 - np.conj(psi_1)) * tot_der[7] * np.conj(ind_der[7]) +  # Gy Gxxy(i) - Gxxy Gy(i) + Gxy Gxy(i)
                          psi_1 * tot_der[3] * np.conj(ind_der[14]) - np.conj(psi_1) * tot_der[14] * np.conj(ind_der[3]) + (psi_1 - np.conj(psi_1)) * tot_der[8] * np.conj(ind_der[8]))  # Gz Gxxz(i) - Gxxz Gz(i) + Gxz Gxz(i)
                ) * force_coeff
        dFyy = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[5]) - np.conj(psi_0) * tot_der[5] * np.conj(ind_der[0]) + (psi_0 - np.conj(psi_0)) * tot_der[2] * np.conj(ind_der[2]) +  # G Gyy(i) - Gyy G(i) + Gy Gy(i)
                                   psi_1 * tot_der[5] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[5]) + (psi_1 - np.conj(psi_1)) * tot_der[2] * np.conj(ind_der[2])) +  # Gyy G(i) - G Gyy(i) + Gy Gy(i)
                1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[15]) - np.conj(psi_1) * tot_der[15] * np.conj(ind_der[1]) + (psi_1 - np.conj(psi_1)) * tot_der[7] * np.conj(ind_der[7]) +  # Gx Gyyx(i) - Gyyx Gx(i) + Gxy Gxy(i)
                          psi_1 * tot_der[2] * np.conj(ind_der[11]) - np.conj(psi_1) * tot_der[11] * np.conj(ind_der[2]) + (psi_1 - np.conj(psi_1)) * tot_der[5] * np.conj(ind_der[5]) +  # Gy Gyyy(i) - Gyyy Gy(i) + Gyy Gyy(i)
                          psi_1 * tot_der[3] * np.conj(ind_der[16]) - np.conj(psi_1) * tot_der[16] * np.conj(ind_der[3]) + (psi_1 - np.conj(psi_1)) * tot_der[9] * np.conj(ind_der[9]))  # Gz Gyyz(i) - Gyyz Gz(i) + Gyz Gyz(i)
                ) * force_coeff
        dFzz = (1j * array.k**2 * (psi_0 * tot_der[0] * np.conj(ind_der[6]) - np.conj(psi_0) * tot_der[6] * np.conj(ind_der[0]) + (psi_0 - np.conj(psi_0)) * tot_der[3] * np.conj(ind_der[3]) +  # G Gzz(i) - Gzz G(i) + Gz Gz(i)
                                   psi_1 * tot_der[6] * np.conj(ind_der[0]) - np.conj(psi_1) * tot_der[0] * np.conj(ind_der[6]) + (psi_1 - np.conj(psi_1)) * tot_der[3] * np.conj(ind_der[3])) +  # Gzz G(i) - G Gzz(i) + Gz Gz(i)
                1j * 3 * (psi_1 * tot_der[1] * np.conj(ind_der[17]) - np.conj(psi_1) * tot_der[17] * np.conj(ind_der[1]) + (psi_1 - np.conj(psi_1)) * tot_der[8] * np.conj(ind_der[8]) +  # Gx Gzzx(i) - Gzzx Gx(i) + Gxz Gxz(i)
                          psi_1 * tot_der[2] * np.conj(ind_der[18]) - np.conj(psi_1) * tot_der[18] * np.conj(ind_der[2]) + (psi_1 - np.conj(psi_1)) * tot_der[9] * np.conj(ind_der[9]) +  # Gy Gzzy(i) - Gzzy Gy(i) + Gyz Gyz(i)
                          psi_1 * tot_der[3] * np.conj(ind_der[12]) - np.conj(psi_1) * tot_der[12] * np.conj(ind_der[3]) + (psi_1 - np.conj(psi_1)) * tot_der[6] * np.conj(ind_der[6]))  # Gz Gzzz(i) - Gzzz Gz(i) + Gzz Gzz(i)
                ) * force_coeff
        return np.array((dFxx, dFyy, dFzz))

    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def second_order_stiffness(*args, **kwargs):
        """
        Calculates the radiation stiffness accounting for both standing and traveling waves.
        """
        pass
    return second_order_stiffness


def amplitude_limiting(array, bounds=(1e-3, 1 - 1e-3), order=4, scaling=10):
    """Create an amplitude limiting cost function.

    Creates a function which can apply additional cost for amplitudes outside
    a certain range. This can be used with unbounded optimizers to enforce
    virtual bounds.
    The limiting is implemented as a polynomial soft-limiter. Amplitudes
    outside the specified range will be multiplied with a scaling, then
    raised to a certain order. The total cost is evaluated over all transducer
    elements.

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    bounds : array_like
        Specifies the bounds to which the amplitude should be limited.
        Default (0.001, 0.999).
    order : int
        The order of the polynomial, default 4.
    scaling : float
        The scaling of the cost, default 10.

    Returns
    -------
    amplitude_limiting : func
        The function described above.

    """
    num_transducers = array.num_transducers
    lower_bound = np.asarray(bounds).min()
    upper_bound = np.asarray(bounds).max()

    def amplitude_limiting(phases, amplitudes):
        """Cost function to limit the amplitudes of an array.

        Note that this only makes sense as a cost function for minimization,
        and only for variable amplitudes.
        """
        under_idx = np.where(amplitudes < lower_bound)[0]
        over_idx = np.where(amplitudes > upper_bound)[0]
        under = scaling * (lower_bound - amplitudes[under_idx])
        over = scaling * (amplitudes[over_idx] - upper_bound)

        value = np.sum(under**order) + np.sum(over**order)
        jacobian = np.zeros(num_transducers)
        jacobian[under_idx] = under**(order - 1) * order
        jacobian[over_idx] = over**(order - 1) * order

        return value, np.zeros(num_transducers), jacobian
    return amplitude_limiting


def pressure(array, location=None, weight=None, spatial_derivatives=None):
    """Create a pressure calculation function.

    Creates a function, which calculates the pressure and the jacobian of the field
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    Modes:
        1) weight = None: returns the complex pressure as a numpy array
        2) weight = value: returns the squared magnitude of the pressure, weighted by the weight, and the corresponding jacobians

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the pressure at
    weight : bool, float or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones

    Returns
    -------
    pressure : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location, orders=0)
    spatial_derivatives = spatial_derivatives[np.newaxis, 0]  # Get only the first underivated component, but keep the axis

    if weight is not None and weight is not False:
        weight = np.atleast_1d(weight)

    if weight is None:
        def calc_values(tot_der):
            return tot_der
    else:
        def calc_values(tot_der):
            return np.abs(tot_der)**2

    def calc_jacobian(tot_der, ind_der):
        return 2 * np.einsum('i...,i...->i...', tot_der, np.conj(ind_der))

    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weight)
    def pressure(*args, **kwargs):
        """
        Calculates the pressure in a sound field.
        """
        pass
    return pressure


def velocity(array, location=None, weights=None, spatial_derivatives=None):
    """Create a velocity calculation function.

    Creates a function, which calculates the sound particle velocity and the
    jacobian of the field generated by the array at the given location when
    given the phases and optional the amplitudes and return them according to mode.

    Modes:
        1) weights = None: returns the complex velocity as a numpy array
        2) weights = False: returns the squared magnitude of the velocity components, and the corresponding jacobians
        3) else: As 2), but the weighted sum of the components

    Parameters
    ----------
    array : TransducerArray
        The object modeling the array
    location : ndarray
        Point(s) to calculate the pressure at
    weights : bool, float or None, optional, default None
        Variable used for mode selection and providing of weights if they apply
    spatial_derivatives : ndarray, optional, default None
        Derivatives to be used if not the default ones

    Returns
    -------
    velocity : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.pressure_derivs(location, orders=1)
    pre_grad_2_vel = 1 / (1j * Air.rho * array.omega)
    spatial_derivatives = spatial_derivatives[num_pressure_derivs[0]:num_pressure_derivs[1]] * pre_grad_2_vel

    if weights is None:
        def calc_values(tot_der):
            return tot_der  # * pre_grad_2_vel
    else:
        def calc_values(tot_der):
            return np.abs(tot_der)**2  # * pre_grad_2_vel

    def calc_jacobian(tot_der, ind_der):
        return 2 * np.einsum('i...,i...->i...', tot_der, np.conj(ind_der))  # * np.abs(pre_grad_2_vel)**2

    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def velocity(*args, **kwargs):
        """
        Calculates the velocity in the sound field.
        """
        pass
    return velocity
