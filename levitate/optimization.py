"""Procedures and algorithms for numerical optimization.

The main method currently in use for acoustic levitation (in this package)
is nonlinear numerical minimization of a cost function. The cost funcion
should be constructed using the `~levitate.fields` module.
"""

import numpy as np
import scipy.optimize
import itertools


def phase_alignment(*states, method='parallel', output='states'):
    r"""Align independet states with respect to the global phase.

    Parameters
    ----------
    *states : arrary_like
        The states to align. THis can be passed in as a sequence of arguments,
        of as a (P, N)-shaped array, where P is the number of states and N is
        the number of elements in each state.
    method : str, optional, keyword-only
        Which method to use for the alignment, default 'parallel'.
        Should be one of 'parallel' or 'sequential', see below for description.
    output : str, optional, keyword-only
        What the function should return, default 'states'.
        The string should contain some combination of 'states' and/or 'phases'.
        The funtion will return the aligned states and/or the obtained phases
        in the order found in the string. If only one of 'states' or 'phases'
        is found, only that one will be returned.


    Notes
    -----
    A single state for an array is only unique up to a global phase.
    When multiple states are considered, the global phase of each state can be
    shifted to minimize the difference between the states.
    This function takes a number of states and finds the optimal phase shifts
    for each of the states. This can operate in two distinct modes,
    a parallel mode and a sequential mode.

    Method `'parallel'` minimizes the sum of all magnitude differences of the states

    .. math::
        \sum_k \sum_l || S_k e^{i\phi_k} -  S_l e^{i\phi_l} ||^2

    or equivalently, maximizes the sum of the states

    .. math::
        || \sum_k S_k e^{i\phi_k} ||^2.

    Explicitly, this is done by numerically minimizing the cost function

    .. math::
        O = \Re\{c A c^*\}

    where :math:`c` is a vector with the phases written on complex form, and
    :math:`A[i,j] = -\sum_n S_i[n] S_j^*[n] (1 - \delta_{ij})`.
    This is suitable for superposition, where we want the states to have the
    most power output.

    Method `sequential` minimizes the difference between consecutive states in
    an iterative fashion. In each step, the difference

    .. math::
        ||S_k e^{i\phi_k} - S_{k-1} e^{i\phi_{k-1}}||

    is minimized. This is done explicitly as

    .. math::
        \phi_k = \phi_{k-1} - \arg\{ \sum_n S_k[n] S_{n-1}^*[n] \}

    with :math:`\phi_0 = 0`. This procedure is suitable for state transitions,
    where the difference between non-consecutive states is irrelevant.
    """
    if len(states) == 1 and np.ndim(states[0]) == 2:
        states = np.asarray(states[0])
    else:
        states = np.stack(states, axis=0)
    num_states = states.shape[0]

    if 'parallel' in method.lower():
        dots = - np.inner(states, states.conjugate())
        dots[np.diag_indices(num_states)] = 0

        def func_and_grad(phases):
            cplx = np.exp(1j * phases)
            tmp = cplx @ dots * cplx.conjugate()
            func = tmp.real.sum()
            grad = 2 * tmp.imag
            return func, grad

        start = np.zeros(num_states)
        result = scipy.optimize.minimize(func_and_grad, start, jac=True)
        phases = result.x
        phases -= phases[0]
    elif 'sequential' in method.lower():
        phases = np.zeros(num_states)
        for state_idx in range(1, num_states):
            phases[state_idx] = phases[state_idx - 1] - np.angle(
                np.sum(states[state_idx] * states[state_idx - 1].conjugate())
            )
    else:
        raise ValueError(f'Method `{method}` is not a known method for phase alignment.')

    if 'state' in output.lower():
        states = np.exp(1j * phases[:, None]) * states

    if 'state' in output.lower() and 'phase' in output.lower():
        if output.lower().find('state') < output.lower().find('phase'):
            return states, phases
        else:
            return phases, states

    if 'state' in output.lower():
        return states
    if 'phase' in output.lower():
        return phases


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
        if type(variable_amplitudes) == str:
            # Strings are iterable, but 'Phases first' should be repeated
            raise TypeError
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
    result = start_values
    for idx, (functions, real_imag, var_amp, const_trans, basinhop, clbck, precl, min_kwarg) in enumerate(zip(function_sequence, use_real_imag, variable_amplitudes, constrain_transducers, basinhopping, callback, precall, minimize_kwargs)):
        start_values = precl(result, idx).copy()
        result, opt_res = minimize(functions, array, use_real_imag=real_imag, variable_amplitudes=var_amp, constrain_transducers=const_trans,
                                   basinhopping=basinhop, return_optim_status=True, minimize_kwargs=min_kwarg, start_values=start_values)
        results.append(result.copy())
        opt_results.append(opt_res)
        if clbck(array=array, result=result, optim_status=opt_res, idx=idx) is False:
            break

    if return_optim_status:
        return np.asarray(results), opt_results
    else:
        return np.asarray(results)


def _minimize_phase_amplitude(function, array, start_values,
                              constrain_transducers, variable_amplitudes,
                              basinhopping, minimize_kwargs,
                              return_optim_status):
    if variable_amplitudes == [False, True]:
        result, status = minimize([function, function], array, start_values=start_values, constrain_transducers=constrain_transducers,
                                  variable_amplitudes=[False, True], basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                                  return_optim_status=True)
        if return_optim_status:
            return result[-1], status[-1]
        else:
            return result[-1]
    elif not isinstance(variable_amplitudes, bool):
        raise TypeError('the `variable_amplitudes` argument should be a bool or "phases first"')

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

            value, jacobians = function(call_complex)
            jacobians = np.concatenate((
                -np.imag(jacobians[unconstrained_transducers]),
                np.einsum('i, i...->i...', 1 / call_amplitudes[unconstrained_transducers], np.real(jacobians[unconstrained_transducers]))
            ))
            return value, jacobians
    elif opt_args['jac']:
        def func(phases):
            call_phases[unconstrained_transducers] = phases[:num_unconstrained_transducers]
            call_complex = call_amplitudes * np.exp(1j * call_phases)

            value, jacobians = function(call_complex)
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

    The cost function should have the signature `f(complex_amplitudes)`
    where `complex_amplitudes` is an ndarray with weight of each element in the transducer array.
    The function should return `value, jacobians` where the jacobians are the
    derivatives of the value w.r.t the transducers as defined in the full documentation.
    Also see the documentation of the field wrappers for further details.

    This function supports minimization sequences. Pass an iterable of functions
    to start sequenced minimization, e.g. a list of cost functions.
    The arguments: `use_real_imag`, `variable_amplitudes`, `constrain_transducers`,
    `callback`, `precall`, `basinhopping`, and  `minimize_kwargs` can be given as single
    values or as iterables of the same length as `functions`.

    Parameters
    ----------
    functions
        The cost function that should be minimized. A single callable, or an
        iterable of callables, as described above.
    array : `TransducerArray`
        The array from which the cost functions are created.
    start_values : complex ndarray, optional
        The start values for the optimization. Will default to 1 for all
        transducers if not given. Note that the precall for minimization
        sequences can overrule this value.
    use_real_imag : bool, default False
        Toggles if the optimization should run using the phase-amplitude formulation
        or the real-imag formulation.
    constrain_transducers : array_like
        Specifies a number of transducers which are constant elements in the
        minimization. Will be used as the second argument in `np.delete`
    variable_amplitudes : bool
        Toggles the usage of varying amplitudes in the minimization.
        If `use_real_imag` is False 'phases first' is also a valid argument for this
        parameter. The minimizer will then automatically sequence to optimize first with
        fixed then with variable amplitudes, returning only the last result.
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
        `precall(complex_amplitudes, idx)`
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
        Stacks sequenced results in the first dimension.
    optim_status : `OptimizeResult`
        Scipy optimization result structure. Optional output,
        toggle with the corresponding input argument.


    """
    if start_values is None:
        start_values = np.ones(array.num_transducers, dtype=complex)
    else:
        start_values = np.asarray(start_values)
    if constrain_transducers is None or constrain_transducers is False:
        constrain_transducers = []
    # Check if we should do sequenced optimization
    try:
        iter(functions)
    except TypeError:
        do_sequence = False
    else:
        do_sequence = True

    if not do_sequence:
        if use_real_imag is True:
            return _minimize_real_imag(function=functions, array=array, start_values=start_values,
                                       constrain_transducers=constrain_transducers, variable_amplitudes=variable_amplitudes,
                                       basinhopping=basinhopping, minimize_kwargs=minimize_kwargs,
                                       return_optim_status=return_optim_status)
        elif use_real_imag is False:
            return _minimize_phase_amplitude(function=functions, array=array, start_values=start_values,
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
