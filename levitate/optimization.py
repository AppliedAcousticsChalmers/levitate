import numpy as np
import scipy.optimize
import itertools


# TODO: Create a named tuple for the combination of (calc_values, calc_jacobian, weights), i.e. a single const function description
# Idea: Should there be a separate function with the inputs ...((calc_values, calc_jacobian), weights)?

class CostFunctionPoint:

    def __init__(self, position, array, *funcs):
        # Determine which spatial derivatives are needed and calculate them from the array
        # Each function stores the required derivaties (and other needed things?) in some defined properties
        self.position = position
        self.array = array
        self._requirements = {}
        self._spatial_derivatives = None
        # self._calculate_jacobians = False
        self._calc_values = []
        self._calc_jacobians = []
        # Here is a good place to add some documenting properties to the cost function, like the position,
        # the included cost functions
        if len(funcs):
            self.append(*funcs)

    def __call__(self, complex_transducer_weights):
        # This is the top-level function which is called by the minimizer
        # Determine the input type, e.g. phases_amplitudes, (phases, amplitudes), etc
        # Pre-compute the amplitude-phase weighted spatial derivatives, and the amplitude-phase weigeted-summed derivatives
        # Call each cost function and return the sum of values and jacobians, weighted with the cost function weights
        individual_derivs = np.einsum('i,ji...->ji...', complex_transducer_weights, self._spatial_derivatives)
        summed_derivs = np.sum(individual_derivs, axis=1)

        value = sum(np.einsum('i..., i', np.atleast_1d(calc_values(summed_derivs)), np.atleast_1d(calc_values.weights)) for calc_values in self._calc_values)
        jacobians = sum(np.einsum('i..., i', np.atleast_1d(calc_jacobians(summed_derivs, individual_derivs)), np.atleast_1d(calc_jacobians.weights)) for calc_jacobians in self._calc_jacobians)

        return value, jacobians

    def _parse_requirements(self, *funcs):
        updated_requirements = {}
        # Go through all requirements in all functions and get the highest requirement
        # of the parsed functions and the already existing functions
        for func in funcs:
            for key, value in func.requires.items():
                if value > self._requirements.get(key, -1):
                    updated_requirements[key] = value
                    self._requirements[key] = value
        for key, value in updated_requirements.items():
            if key.find('pressure_orders') > -1:
                self._spatial_derivatives = self.array.spatial_derivatives(self.position, orders=value)
            else:
                raise ValueError("Unknown requirement '{}'".format(key))

    def append(self, *funcs):
        calc_values, calc_jacobians = zip(*funcs)
        self._parse_requirements(*calc_values, *calc_jacobians)
        self._calc_values.extend(calc_values)
        self._calc_jacobians.extend(calc_jacobians)


def _minimize_sequence(function_sequence, array,
                       start_values, use_real_imag,
                       constrain_transducers, variable_amplitudes,
                       callback, precall,
                       basinhopping, minimize_kwargs,
                       return_optim_status,
                       ):
    try:
        iter(use_real_imag)
    except TypeError:
        use_real_imag = itertools.repeat(use_real_imag)

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
        iter(variable_amplitudes)
    except TypeError:
        variable_amplitudes = itertools.repeat(variable_amplitudes)

    try:
        iter(callback)
    except TypeError:
        callback = itertools.repeat(callback)

    try:
        iter(precall)
    except TypeError:
        precall = itertools.repeat(precall)

    try:
        iter(basinhopping)
    except TypeError:
        basinhopping = itertools.repeat(basinhopping)

    try:
        iter(minimize_kwargs)  # Exception for None
        if type(next(iter(minimize_kwargs))) is not dict:
            raise TypeError
    except TypeError:
        minimize_kwargs = itertools.repeat(minimize_kwargs)

    results = []
    opt_results = []
    for idx, (functions, real_imag, var_amp, const_trans, basinhop, clbck, precl, min_kwarg) in enumerate(zip(function_sequence, use_real_imag, variable_amplitudes, constrain_transducers, basinhopping, callback, precall, minimize_kwargs)):
        start_values = precl(start_values, idx).copy()
        result, opt_res = minimize(functions, array, use_real_imag=real_imag, variable_amplitudes=var_amp, constrain_transducers=const_trans,
                                   basinhopping=basinhop, return_optim_status=True, minimize_kwargs=min_kwarg)
        results.append(result.copy())
        opt_results.append(opt_res)
        if clbck(array=array, result=result, optim_status=opt_res, idx=idx) is False:
            break

    if return_optim_status:
        return np.asarray(results), opt_results
    else:
        return np.asarray(results)


def _minimize_phase_amplitude(functions, array, start_values,
                              constrain_transducers, variable_amplitudes,
                              basinhopping, minimize_kwargs,
                              return_optim_status):
    if variable_amplitudes == 'phases first':
        return minimize([functions, functions], array, start_values=start_values, constrain_transducers=constrain_transducers,
                        variable_amplitudes=[False, True], basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                        return_optim_status=return_optim_status)[-1]
        # raise NotImplementedError('Lazy setting for phases first optimization not implemented')

    num_total_transducers = len(start_values)
    unconstrained_transducers = np.delete(np.arange(num_total_transducers), constrain_transducers)
    num_unconstrained_transducers = len(unconstrained_transducers)
    call_phases = np.angle(start_values)
    call_amplitudes = np.abs(start_values)

    if variable_amplitudes is True:
        start = np.concatenate((call_phases[unconstrained_transducers], call_amplitudes[unconstrained_transducers])).copy()
        bounds = [(None, None)] * num_unconstrained_transducers + [(1e-3, 1)] * num_unconstrained_transducers
    else:
        start = call_phases[unconstrained_transducers].copy()
        bounds = [(None, None)] * num_unconstrained_transducers  # Use bounds even if the problem is unbounded since the L-BFGS-B is faster than normal BFGS

    opt_args = {'jac': True, 'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'gtol': 1e-9, 'ftol': 1e-15}}
    if minimize_kwargs is not None:
        opt_args.update(minimize_kwargs)

    if opt_args['jac'] and variable_amplitudes:
        def func(phases_amplitudes):
            call_phases[unconstrained_transducers] = phases_amplitudes[:num_unconstrained_transducers]
            call_amplitudes[unconstrained_transducers] = phases_amplitudes[num_unconstrained_transducers:]
            call_complex = call_amplitudes * np.exp(1j * call_phases)

            value = 0
            jacobians = np.zeros(num_total_transducers, dtype=complex)
            for f in functions:
                out = f(call_complex)
                value += out[0]
                jacobians += out[1]
            jacobians = np.concatenate((-np.imag(jacobians[unconstrained_transducers]), np.real(jacobians[unconstrained_transducers])))
            return value, jacobians
    elif opt_args['jac']:
        def func(phases):
            call_phases[unconstrained_transducers] = phases[:num_unconstrained_transducers]
            call_complex = call_amplitudes * np.exp(1j * call_phases)

            value = 0
            jacobians = np.zeros(num_total_transducers, dtype=complex)
            for f in functions:
                out = f(call_complex)
                value += out[0]
                jacobians += out[1]
            jacobians = -np.imag(jacobians[unconstrained_transducers])
            return value, jacobians
    else:
        raise NotImplementedError('Minimiation without jacobians currently not supported')

    if basinhopping:
        if basinhopping is True:
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


def _minimize_real_imag(functions, start_values,
                        constrain_transducers, variable_amplitudes,
                        basinhopping, minimize_kwargs,
                        return_optim_status):
    raise NotImplementedError('Optimization using real and imag not implemented')


def minimize(functions, array,
             start_values=None, use_real_imag=False,
             constrain_transducers=None, variable_amplitudes=False,
             callback=None, precall=None,
             basinhopping=False, minimize_kwargs=None,
             return_optim_status=False,
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
    if start_values is None:
        start_values = array.complex_amplitudes.copy()
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
        if use_real_imag is True:
            return _minimize_real_imag(functions=functions, array=array, start_values=start_values,
                                       constrain_transducers=constrain_transducers, variable_amplitudes=variable_amplitudes,
                                       basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                                       return_optim_status=return_optim_status)
        elif use_real_imag is False:
            return _minimize_phase_amplitude(functions=functions, array=array, start_values=start_values,
                                             constrain_transducers=constrain_transducers, variable_amplitudes=variable_amplitudes,
                                             basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                                             return_optim_status=return_optim_status)
        else:
            raise ValueError("Argument 'use_real_imag' can only take bools, not `{}`".format(use_real_imag))
    else:
        if callback is None:
            def callback(**kwargs):
                return True
        if precall is None:
            def precall(start_values, idx):
                return start_values
        return _minimize_sequence(function_sequence=functions, array=array,
                                  start_values=start_values, use_real_imag=use_real_imag,
                                  constrain_transducers=constrain_transducers, variable_amplitudes=variable_amplitudes,
                                  callback=callback, precall=precall,
                                  basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                                  return_optim_status=return_optim_status)
