import numpy as np
# from scipy.optimize import minimize, basinhopping
import scipy.optimize
import logging
import itertools

from .materials import Air
from . import num_spatial_derivatives

logger = logging.getLogger(__name__)


def minimize(functions, array, variable_amplitudes=False,
             constrain_transducers=None, callback=None, precall=None,
             basinhopping=False, return_optim_status=False, minimize_kwargs=None,
             ):
    """
    Minimizes a set of cost functions.

    This function supports minimization sequeces. Pass an iterable of iterables
    of cost functions to start sequenced minimization.

    Parameters
    ----------
    functions
        The const functions that should be minimized. A single callable, an
        iterable of callables, or an iterable of iterables of callables, as
        described above.
    array : `TransducerArray`
        The array from which the const functions are created.
    variable_amplitudes : bool
        Toggles the usage of varying amplitudes in the minimization.
    constrain_transducers : array_like
        Specifies a number of transducers which are constant elements in the
        minimization. Will be used as the second argument in `np.delete`
    callback : callable, ``callback(array=array, retult=result, optim_status=opt_res, idx=idx)``
        A callback function which will be called after each step in sequenced
        minimization. Return false from the callback to break the sequence.
    precall : callable, ``precall(phases, amplitudes, idx)``
        Initialization function which will be called with the array phases,
        amplitudes, and the sequence index before each sequence step.
        Must return the initial phases and amplitudes for the sequence step.
        Default sets the phases and amplitudes to the solution of the previous
        sequence step, or the original state for the first iteration.
    basinhopping : bool or int
        Specifies if basinhopping should be used. Pass an int to specify the
        number of basinhopping interations, or True to use default value.
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
        Scipy optimization result structure. Optional ourput,
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

        if variable_amplitudes:
            unconstrained_variables = np.concatenate((unconstrained_transducers, unconstrained_transducers + array.num_transducers))
        else:
            unconstrained_variables = unconstrained_transducers
        call_values = array.phases_amplitudes.copy()
        start = call_values[unconstrained_variables].copy()

        bounds = [(None, None)] * num_unconstrained_transducers
        if variable_amplitudes:
            bounds += [(1e-3, 1)] * num_unconstrained_transducers
        opt_args = {'jac': True, 'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'gtol': 1e-9, 'ftol': 1e-15}}
        if minimize_kwargs is not None:
            opt_args.update(minimize_kwargs)

        if opt_args['jac']:
            def func(phases_amplitudes):
                call_values[unconstrained_variables] = phases_amplitudes
                results = [f(call_values) for f in functions]
                value = np.sum(result[0] for result in results)
                jacobian = np.sum(result[1] for result in results)[unconstrained_variables]

                return value, jacobian
        else:
            def func(phases_amplitudes):
                call_values[unconstrained_variables] = phases_amplitudes
                value = np.sum([f(call_values) for f in functions])

                return value

        if basinhopping:
            if basinhopping is True:
                # It's not a number, use default value
                basinhopping = 20
            opt_result = scipy.optimize.basinhopping(func, start, T=1e-7, minimizer_kwargs=opt_args, niter=basinhopping)
        else:
            opt_result = scipy.optimize.minimize(func, start, **opt_args)

        call_values[unconstrained_variables] = opt_result.x
        if return_optim_status:
            return call_values, opt_result
        else:
            return call_values
    else:
        # ====================================================
        # Sequenced minimization of cost functions start here!
        # ====================================================
        initial_array_state = array.phases_amplitudes
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
            array.phases_amplitudes = result
            if clbck(array=array, retult=result, optim_status=opt_res, idx=idx) is False:
                break

        array.phases_amplitudes = initial_array_state
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


def _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False):
    if np.iscomplexobj(phases_amplitudes):
        if allow_complex:
            phases = np.angle(phases_amplitudes)
            amplitudes = np.abs(phases_amplitudes)
            variable_amplitudes = None
        else:
            raise NotImplementedError('Jacobian does not exist for complex inputs!')
    elif phases_amplitudes.size == num_transducers:
        phases = phases_amplitudes
        amplitudes = np.ones(num_transducers)
        variable_amplitudes = False
    elif phases_amplitudes.size == 2 * num_transducers:
        phases = phases_amplitudes.ravel()[:num_transducers]
        amplitudes = phases_amplitudes.ravel()[num_transducers:]
        variable_amplitudes = True
    return phases, amplitudes, variable_amplitudes


def vector_target(vector_calculator, target_vector=(0, 0, 0), weights=(1, 1, 1)):
    """
    Creates a function which calculates the weighted squared difference between a
    target vector and a varying vector.

    This can create cost functions representing :math:`||(v - v_0)||^2_w`, i.e.
    the weighted square norm between a varying vector and a fixed vector.
    Note that the values in the weights will not be squared.

    Parameters
    ----------
    vector_calculator : callable
        A function which calculates the varying vector from array phases and
        optional amplitudes, along with the jacobian of said varying vector.
        This function must return `(v, dv)`, where `v` is a 3 element ndarray,
        and `dv` is a shape 3xn ndarray.
        Suitable functions can be created by passing `False` as weights to other
        cost function generators in this module.
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

    def vector_target(phases_amplitudes):
        v, dv = vector_calculator(phases_amplitudes)
        difference = v - target_vector
        value = np.sum(np.abs(difference)**2 * weights)
        jacobian = (2 * weights * difference).dot(dv)
        return value, jacobian
    return vector_target


def create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights):
    num_transducers = spatial_derivatives.shape[1]

    # def parse_inputs(*args, **kwargs):
    #     if len(args) == 1 and np.iscomplexobj(args[0]) and len(args[0]) == num_transducers:
    #         # Single input with complex value
    #         return np.angle(args[0]), np.abs(args[0])
    #     elif len(args) == 2 and len(args[0]) == num_transducers and len(args[1]) == num_transducers:
    #         # Two inputs, should be phase, amplitude
    #         return args
    #     elif 'amplitudes' in kwargs and 'phases' in kwargs:
    #         # Keyword input
    #         return kwargs['phases'], kwargs['amplitudes']
    def wrapper(f):
        try:
            len(weights)
        except TypeError:
            if weights is None:
                def func(phases_amplitudes):
                    """Returns the values as is."""
                    phases, amplitudes, _ = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
                    complex_coeff = amplitudes * np.exp(1j * phases)
                    return calc_values(np.einsum('i,ji...->j...', complex_coeff, spatial_derivatives))
            else:
                def func(phases_amplitudes):
                    """Returns the unweighted values as well as the unweighted jacobian."""
                    phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
                    complex_coeff = amplitudes * np.exp(1j * phases)
                    ind_der = np.einsum('i,ji...->ji...', complex_coeff, spatial_derivatives)
                    tot_der = np.sum(ind_der, axis=1)
                    value = calc_values(tot_der)
                    jacobian = calc_jacobian(tot_der, ind_der)
                    if variable_amplitudes:
                        return value, np.concatenate((jacobian.imag, np.einsum('i,ji...->ji...', 1 / amplitudes, jacobian.real)), axis=1)
                    else:
                        return value, jacobian.imag
        else:
            def func(phases_amplitudes):
                """Returns the weighted sum of the values and the weighted sum of the jacobian."""
                phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
                complex_coeff = amplitudes * np.exp(1j * phases)
                ind_der = np.einsum('i,ji...->ji...', complex_coeff, spatial_derivatives)
                tot_der = np.sum(ind_der, axis=1)

                values = calc_values(tot_der)
                jacobians = calc_jacobian(tot_der, ind_der)
                value = np.einsum('i, i...', weights, values)
                jacobian = np.einsum('i, i...', weights, jacobians)
                # value, jacobian = f(np.einsum('i, i...', weights, values), np.einsum('i, i...', weights, jacobians))

                if variable_amplitudes:
                    return value, np.concatenate((jacobian.imag, np.einsum('i,i...->i...', 1 / amplitudes, jacobian.real)), axis=0)
                else:
                    return value, jacobian.imag
        func.__name__ = f.__name__
        func.__qualname__ = f.__qualname__
        func.__module__ = f.__module__
        if f.__doc__ is not None:
            func.__doc__ = f.__doc__ + '\n' + func.__doc__
        return func
    return wrapper


def gorkov_divergence(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """
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
        spatial_derivatives = array.spatial_derivatives(location, orders=2)

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
        """ Calculate the divergence of the Gor'kov potential. """
        pass
    return gorkov_divergence


def gorkov_laplacian(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """
    Creates a function, which calculates the laplacian and the jacobian of the field
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    Modes:
        1) weights = None: returns the x, y and z second derivatives as a numpy array
        2) weights = False: returns the second derivatives as an array and the jacobian as a 3 x num_transducers 2darray as a tuple
        3) else: returns the weighted laplacian and the corresponding jacobian

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
        spatial_derivatives = array.spatial_derivatives(location)

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
        """ Calculates the cartesian parts of the Laplacian of the Gor'kov potential."""
        pass
    return gorkov_laplacian


def second_order_force(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """
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
        spatial_derivatives = array.spatial_derivatives(location, orders=2)

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
        """Calculates the radiation force accounting for both standing and travelling waves."""
        pass
    return second_order_force


def second_order_stiffness(array, location=None, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
    """
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
        spatial_derivatives = array.spatial_derivatives(location, orders=3)

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
        """Calculates the radiation stiffness accounting for both standing and travelling waves."""
        pass
    return second_order_stiffness


def amplitude_limiting(array, bounds=(1e-3, 1 - 1e-3), order=4, scaling=10):
    """
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

    def amplitude_limiting(phases_amplitudes):
        """Cost function to limit the amplitudes of an array.
=
        Note that this only makes sense as a cost function for minimzation,
        and only for variable amplitudes.
        """
        _, amplitudes, variable_amps = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
        if not variable_amps:
            return 0, np.zeros(num_transducers)
        under_idx = np.where(amplitudes < lower_bound)[0]
        over_idx = np.where(amplitudes > upper_bound)[0]
        under = scaling * (lower_bound - amplitudes[under_idx])
        over = scaling * (amplitudes[over_idx] - upper_bound)

        value = np.sum(under**order) + np.sum(over**order)
        jacobian = np.zeros(2 * num_transducers)
        jacobian[num_transducers + under_idx] = under**(order - 1) * order
        jacobian[num_transducers + over_idx] = over**(order - 1) * order

        return value, jacobian
    return amplitude_limiting


def pressure_null(array, location=None, weights=None, spatial_derivatives=None):
    """
    Creates a function, which calculates the pressure derivatives and the jacobian of the field
    generated by the array at the given location when given the phases and optional
    the amplitudes and return them according to mode.

    Modes:
        1) weights = None: returns the x, y and z derivatives and p as a numpy array
        2) weights = False: returns the derivatives (including squared pressure) as an array and the jacobian as a 4 x num_transducers 2darray as a tuple
        3) else: returns the weighted derivatives including squared pressure and the corresponding jacobian

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

    Returns
    -------
    pressure_null : func
        The function described above
    """
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location, orders=1)
    if weights is None or weights is False:
        spatial_derivatives = spatial_derivatives[:num_spatial_derivatives[1]]
    else:
        weights = np.atleast_1d(weights)
        if len(weights) == 4:
            # weights = np.asarray(weights)
            spatial_derivatives = spatial_derivatives[:num_spatial_derivatives[1]]
        elif len(weights) == 3:
            # weights = np.concatenate(([0], weights))
            spatial_derivatives = spatial_derivatives[1:num_spatial_derivatives[1]]
        elif len(weights):
            spatial_derivatives = spatial_derivatives[np.newaxis, 0]
            # weights = np.array((weights, 0, 0, 0))

    def calc_values(tot_der):
        if weights is None:
            return tot_der
        else:
            return np.abs(tot_der)**2

    def calc_jacobian(tot_der, ind_der):
        return 2 * np.einsum('i..., i... -> i...', tot_der, np.conj(ind_der))

    @create_weighted_cost_function(calc_values, calc_jacobian, spatial_derivatives, weights)
    def pressure_null(*args, **kwargs):
        """Calculates the pressure and possibly pressure gradient in a sound field."""
        pass
    return pressure_null
