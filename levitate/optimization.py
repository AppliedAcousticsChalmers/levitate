import numpy as np
# from scipy.optimize import minimize, basinhopping
import scipy.optimize
import logging
import itertools

from . import models

logger = logging.getLogger(__name__)


class Optimizer:
    """ Optimizer for transducer array optimizations

    Attributes
    ----------
    array : levitate.models.TransducerArray
        The transducer array for which to optimize.
    objectives : list of callables
        A list of cost functions which should be minimized.
        A valid cost function has the signature
        `value, jacobian = fun(phases_amplitudes)`.
        If the amplitudes should be varied in the optimization, the arguments
        for cost functions will be of shape `2 * num_transduces` where all
        phases come first, then all amplitudes. If the amplitudes are kept
        constant, the input is of shape `num_transducers` with only the phases.
    variable_amplitudes : bool
        Toggles the optimization of amplitudes.
    basinhopping : bool
        Toggles the use of basinhopping in the optimization.
    complex_amplitudes : complex numpy.ndarray
        The result after optimization, on complex valued form.

    """

    def __init__(self, array=None):
        if array is None:
            self.array = models.TransducerArray()
        else:
            self.array = array
        self.objectives = []
        self.basinhopping = False
        self.variable_amplitudes = False

    @property
    def complex_amplitudes(self):
        return self.amplitudes * np.exp(1j * self.phases)

    @complex_amplitudes.setter
    def complex_amplitudes(self, value):
        self.amplitudes = np.abs(value)
        self.phases = np.angle(value)

    def func_and_jac(self, phases_amplitudes):
        """ Evaluates all cost functions

        Parameters
        ----------
        phases_amplitudes : numpy.ndarray
            Representation of the array state, formatted according to the
            `variable_amplitudes` attribute.

        Returns
        -------
        value : float
            The total sum of the cost functions.
        jacobian : ndarray
            The jacobian of the sum of cost functions, wrt the input arguments.
            This if up to individual cost functions to implement properly.


        """
        results = [f(phases_amplitudes) for f in self.objectives]
        value = np.sum(result[0] for result in results)
        jac = np.sum(result[1] for result in results)
        return value, jac

    def __call__(self):
        """ Runs the optimization.

        """
        # Initialize all parts of the objective function
        # Assemble objective function and jacobian function
        # Start optimization
        # Basin hopping? Check number of iterations?
        # Return phases?
        # self.initialize()
        # Set starting points
        if self.variable_amplitudes:
            start = np.concatenate((self.array.phases, self.array.amplitudes))
        else:
            start = self.array.phases
        # Set bounds for L-BFGS-B
        bounds = [(None, None)] * self.array.num_transducers
        if self.variable_amplitudes:
            bounds += [(1e-3, 1)] * self.array.num_transducers
        # TODO: The method selection should be configureable
        args = {'jac': True,  # self.objective_list[0][1]
                # 'method': 'BFGS', 'options': {'return_all': False, 'gtol': 5e-5, 'norm': 2, 'disp': True}}
                'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'gtol': 1e-9, 'ftol': 1e-15}}
        if self.basinhopping:
            take_step = RadndomDisplacer(self.array.num_transducers, self.variable_amplitudes, stepsize=0.1)
            self.result = scipy.optimize.basinhopping(self.func_and_jac, start, T=1e-7, take_step=None, minimizer_kwargs=args, disp=True)
        else:
            self.result = scipy.optimize.minimize(self.func_and_jac, start, callback=None, **args)

        if self.variable_amplitudes:
            self.phases = self.result.x[:self.array.num_transducers]
            self.amplitudes = self.result.x[self.array.num_transducers:]
        else:
            self.phases = self.result.x
            self.amplitudes = self.array.amplitudes

        # self.result = minimize(self.function, self.array.phases, jac=self.jacobian, callback=None,
        # method='L-BFGS-B', bounds=[(-3*np.pi, 3*np.pi)]*self.array.num_transducers, options={'gtol': 1e-7, 'ftol': 1e-12})
        # method='BFGS', options={'return_all': True, 'gtol': 1e-5, 'norm': 2})


def minimize_objectives(functions, array, variable_amplitudes=False,
                        constrain_transducers=None, passover_callable=None,
                        basinhopping=False, status_return=None, minimize_kwargs=None,
                        ):
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
        pass
    else:
        # This is where we end up if we are sequencing optimizations
        initial_array_state = array.phases_amplitudes
        try:
            iter(variable_amplitudes)
        except TypeError:
            variable_amplitudes = itertools.repeat(variable_amplitudes)
        if passover_callable is None:
            def passover_callable(arg): return arg
        try:
            iter(passover_callable)
        except TypeError:
            passover_callable = itertools.repeat(passover_callable)
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
        for function, var_amp, const_trans, basinhop, pass_call, min_kwarg in zip(functions, variable_amplitudes, constrain_transducers, basinhopping, passover_callable, minimize_kwargs):
            results.append(minimize_objectives(function, array, variable_amplitudes=var_amp,
                constrain_transducers=const_trans, basinhopping=basinhop, status_return=status_return, minimize_kwargs=min_kwarg))
            array.phases_amplitudes = pass_call(results[-1])
        array.phases_amplitudes = initial_array_state
        return results

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
    if status_return is not None:
        status_return.append(opt_result)

    call_values[unconstrained_variables] = opt_result.x
    return call_values


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
    Note that the values in the norm will not be squared.

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


def gorkov_divergence(array, location, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
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
    num_transducers = array.num_transducers
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location, orders=2)

    V = 4 / 3 * np.pi * radius_sphere**3
    rho_air = models.rho_air
    c_air = models.c_air
    compressibility_air = 1 / (rho_air * c_air**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    monopole_coefficient = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (rho_sphere / rho_air - 1) / (2 * rho_sphere / rho_air + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * rho_air)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * compressibility_air * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * rho_air

    def calc_values(tot_der):
        Ux = (pressure_coefficient * (tot_der['x'] * np.conj(tot_der[''])).real -
              gradient_coefficient * (tot_der['xx'] * np.conj(tot_der['x'])).real -
              gradient_coefficient * (tot_der['xy'] * np.conj(tot_der['y'])).real -
              gradient_coefficient * (tot_der['xz'] * np.conj(tot_der['z'])).real) * 2
        Uy = (pressure_coefficient * (tot_der['y'] * np.conj(tot_der[''])).real -
              gradient_coefficient * (tot_der['xy'] * np.conj(tot_der['x'])).real -
              gradient_coefficient * (tot_der['yy'] * np.conj(tot_der['y'])).real -
              gradient_coefficient * (tot_der['yz'] * np.conj(tot_der['z'])).real) * 2
        Uz = (pressure_coefficient * (tot_der['z'] * np.conj(tot_der[''])).real -
              gradient_coefficient * (tot_der['xz'] * np.conj(tot_der['x'])).real -
              gradient_coefficient * (tot_der['yz'] * np.conj(tot_der['y'])).real -
              gradient_coefficient * (tot_der['zz'] * np.conj(tot_der['z'])).real) * 2

        return np.array((Ux, Uy, Uz))

    def calc_jacobian(tot_der, ind_der):
        dUx = (pressure_coefficient * (tot_der['x'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['x'])) -
               gradient_coefficient * (tot_der['xx'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['xx'])) -
               gradient_coefficient * (tot_der['xy'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['xy'])) -
               gradient_coefficient * (tot_der['xz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['xz']))) * 2
        dUy = (pressure_coefficient * (tot_der['y'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['y'])) -
               gradient_coefficient * (tot_der['xy'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['xy'])) -
               gradient_coefficient * (tot_der['yy'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['yy'])) -
               gradient_coefficient * (tot_der['yz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['yz']))) * 2
        dUz = (pressure_coefficient * (tot_der['z'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['z'])) -
               gradient_coefficient * (tot_der['xz'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['xz'])) -
               gradient_coefficient * (tot_der['yz'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['yz'])) -
               gradient_coefficient * (tot_der['zz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['zz']))) * 2

        return np.array((dUx, dUy, dUz))

    if weights is None:
        def gorkov_divergence(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            tot_der = {}
            for key, value in spatial_derivatives.items():
                tot_der[key] = np.sum(complex_coeff * value)
            return calc_values(tot_der)
    elif weights is False:
        def gorkov_divergence(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            value = calc_values(tot_der)
            jacobian = calc_jacobian(tot_der, ind_der)

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes), axis=-1)
            else:
                return value, jacobian.imag
    else:
        wx, wy, wz = weights

        def gorkov_divergence(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            # Tried to keep values and jacobian as single arrays and converting
            # weights to an array, but this seems to be the fastest implementation.
            Ux, Uy, Uz = calc_values(tot_der)
            dUx, dUy, dUz = calc_jacobian(tot_der, ind_der)
            value = wx * Ux + wy * Uy + wz * Uz
            jacobian = wx * dUx + wy * dUy + wz * dUz

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag

    return gorkov_divergence


def gorkov_laplacian(array, location, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
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
    num_transducers = array.num_transducers
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location)

    c_air = models.c_air
    rho_air = models.rho_air
    V = 4 / 3 * np.pi * radius_sphere**3
    compressibility_air = 1 / (rho_air * c_air**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    monopole_coefficient = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    dipole_coefficient = 2 * (rho_sphere / rho_air - 1) / (2 * rho_sphere / rho_air + 1)   # f_2 in H. Bruus 2012
    preToVel = 1 / (array.omega * rho_air)  # Converting velocity to pressure gradient using equation of motion
    pressure_coefficient = V / 4 * compressibility_air * monopole_coefficient
    gradient_coefficient = V * 3 / 8 * dipole_coefficient * preToVel**2 * rho_air

    def calc_values(tot_der):
        Uxx = (pressure_coefficient * (tot_der['xx'] * np.conj(tot_der['']) + tot_der['x'] * np.conj(tot_der['x'])).real -
               gradient_coefficient * (tot_der['xxx'] * np.conj(tot_der['x']) + tot_der['xx'] * np.conj(tot_der['xx'])).real -
               gradient_coefficient * (tot_der['xxy'] * np.conj(tot_der['y']) + tot_der['xy'] * np.conj(tot_der['xy'])).real -
               gradient_coefficient * (tot_der['xxz'] * np.conj(tot_der['z']) + tot_der['xz'] * np.conj(tot_der['xz'])).real) * 2
        Uyy = (pressure_coefficient * (tot_der['yy'] * np.conj(tot_der['']) + tot_der['y'] * np.conj(tot_der['y'])).real -
               gradient_coefficient * (tot_der['yyx'] * np.conj(tot_der['x']) + tot_der['xy'] * np.conj(tot_der['xy'])).real -
               gradient_coefficient * (tot_der['yyy'] * np.conj(tot_der['y']) + tot_der['yy'] * np.conj(tot_der['yy'])).real -
               gradient_coefficient * (tot_der['yyz'] * np.conj(tot_der['z']) + tot_der['yz'] * np.conj(tot_der['yz'])).real) * 2
        Uzz = (pressure_coefficient * (tot_der['zz'] * np.conj(tot_der['']) + tot_der['z'] * np.conj(tot_der['z'])).real -
               gradient_coefficient * (tot_der['zzx'] * np.conj(tot_der['x']) + tot_der['xz'] * np.conj(tot_der['xz'])).real -
               gradient_coefficient * (tot_der['zzy'] * np.conj(tot_der['y']) + tot_der['yz'] * np.conj(tot_der['yz'])).real -
               gradient_coefficient * (tot_der['zzz'] * np.conj(tot_der['z']) + tot_der['zz'] * np.conj(tot_der['zz'])).real) * 2
        return np.array((Uxx, Uyy, Uzz))

    def calc_jacobian(tot_der, ind_der):
        dUxx = (pressure_coefficient * (tot_der['xx'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['xx']) + 2 * tot_der['x'] * np.conj(ind_der['x'])) -
                gradient_coefficient * (tot_der['xxx'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['xxx']) + 2 * tot_der['xx'] * np.conj(ind_der['xx'])) -
                gradient_coefficient * (tot_der['xxy'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['xxy']) + 2 * tot_der['xy'] * np.conj(ind_der['xy'])) -
                gradient_coefficient * (tot_der['xxz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['xxz']) + 2 * tot_der['xz'] * np.conj(ind_der['xz']))) * 2
        dUyy = (pressure_coefficient * (tot_der['yy'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['yy']) + 2 * tot_der['y'] * np.conj(ind_der['y'])) -
                gradient_coefficient * (tot_der['yyx'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['yyx']) + 2 * tot_der['xy'] * np.conj(ind_der['xy'])) -
                gradient_coefficient * (tot_der['yyy'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['yyy']) + 2 * tot_der['yy'] * np.conj(ind_der['yy'])) -
                gradient_coefficient * (tot_der['yyz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['yyz']) + 2 * tot_der['yz'] * np.conj(ind_der['yz']))) * 2
        dUzz = (pressure_coefficient * (tot_der['zz'] * np.conj(ind_der['']) + tot_der[''] * np.conj(ind_der['zz']) + 2 * tot_der['z'] * np.conj(ind_der['z'])) -
                gradient_coefficient * (tot_der['zzx'] * np.conj(ind_der['x']) + tot_der['x'] * np.conj(ind_der['zzx']) + 2 * tot_der['xz'] * np.conj(ind_der['xz'])) -
                gradient_coefficient * (tot_der['zzy'] * np.conj(ind_der['y']) + tot_der['y'] * np.conj(ind_der['zzy']) + 2 * tot_der['yz'] * np.conj(ind_der['yz'])) -
                gradient_coefficient * (tot_der['zzz'] * np.conj(ind_der['z']) + tot_der['z'] * np.conj(ind_der['zzz']) + 2 * tot_der['zz'] * np.conj(ind_der['zz']))) * 2
        return np.array((dUxx, dUyy, dUzz))

    if weights is None:
        def gorkov_laplacian(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            tot_der = {}
            for key, value in spatial_derivatives.items():
                tot_der[key] = np.sum(complex_coeff * value)

            return calc_values(tot_der)
    elif weights is False:
        def gorkov_laplacian(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            value = calc_values(tot_der)
            jacobian = calc_jacobian(tot_der, ind_der)

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes), axis=-1)
            else:
                return value, jacobian.imag
    else:
        wx, wy, wz = weights

        def gorkov_laplacian(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            # Tried to keep values and jacobian as single arrays and converting
            # weights to an array, but this seems to be the fastest implementation.
            Uxx, Uyy, Uzz = calc_values(tot_der)
            dUxx, dUyy, dUzz = calc_jacobian(tot_der, ind_der)
            value = wx * Uxx + wy * Uyy + wz * Uzz
            jacobian = wx * dUxx + wy * dUyy + wz * dUzz

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag

    return gorkov_laplacian


def second_order_force(array, location, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
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
    num_transducers = array.num_transducers
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location, orders=2)

    c_air = models.c_air
    rho_air = models.rho_air
    compressibility_air = 1 / (rho_air * c_air**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    f_1 = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    f_2 = 2 * (rho_sphere / rho_air - 1) / (2 * rho_sphere / rho_air + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * compressibility_air

    def calc_values(tot_der):
        Fx = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(tot_der['x']) +
                                 psi_1 * tot_der['x'] * np.conj(tot_der[''])) +
              1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['xx']) +
                                tot_der['y'] * np.conj(tot_der['xy']) +
                                tot_der['z'] * np.conj(tot_der['xz']))
              ).real * force_coeff
        Fy = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(tot_der['y']) +
                                 psi_1 * tot_der['y'] * np.conj(tot_der[''])) +
              1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['xy']) +
                                tot_der['y'] * np.conj(tot_der['yy']) +
                                tot_der['z'] * np.conj(tot_der['yz']))
              ).real * force_coeff
        Fz = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(tot_der['z']) +
                                 psi_1 * tot_der['z'] * np.conj(tot_der[''])) +
              1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['xz']) +
                                tot_der['y'] * np.conj(tot_der['yz']) +
                                tot_der['z'] * np.conj(tot_der['zz']))
              ).real * force_coeff
        return np.array((Fx, Fy, Fz))

    def calc_jacobian(tot_der, ind_der):
        dFx = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['x']) - np.conj(psi_0) * tot_der['x'] * np.conj(ind_der['']) +
                                  psi_1 * tot_der['x'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['x'])) +
               1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['xx']) - np.conj(psi_1) * tot_der['xx'] * np.conj(ind_der['x']) +
                         psi_1 * tot_der['y'] * np.conj(ind_der['xy']) - np.conj(psi_1) * tot_der['xy'] * np.conj(ind_der['y']) +
                         psi_1 * tot_der['z'] * np.conj(ind_der['xz']) - np.conj(psi_1) * tot_der['xz'] * np.conj(ind_der['z']))
               ) * force_coeff
        dFy = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['y']) - np.conj(psi_0) * tot_der['y'] * np.conj(ind_der['']) +
                                  psi_1 * tot_der['y'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['y'])) +
               1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['xy']) - np.conj(psi_1) * tot_der['xy'] * np.conj(ind_der['x']) +
                         psi_1 * tot_der['y'] * np.conj(ind_der['yy']) - np.conj(psi_1) * tot_der['yy'] * np.conj(ind_der['y']) +
                         psi_1 * tot_der['z'] * np.conj(ind_der['yz']) - np.conj(psi_1) * tot_der['yz'] * np.conj(ind_der['z']))
               ) * force_coeff
        dFz = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['z']) - np.conj(psi_0) * tot_der['z'] * np.conj(ind_der['']) +
                                  psi_1 * tot_der['z'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['z'])) +
               1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['xz']) - np.conj(psi_1) * tot_der['xz'] * np.conj(ind_der['x']) +
                         psi_1 * tot_der['y'] * np.conj(ind_der['yz']) - np.conj(psi_1) * tot_der['yz'] * np.conj(ind_der['y']) +
                         psi_1 * tot_der['z'] * np.conj(ind_der['zz']) - np.conj(psi_1) * tot_der['zz'] * np.conj(ind_der['z']))
               ) * force_coeff
        return np.array((dFx, dFy, dFz))

    if weights is None:
        def second_order_force(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            tot_der = {}
            for key, value in spatial_derivatives.items():
                tot_der[key] = np.sum(complex_coeff * value)
            return calc_values(tot_der)
    elif weights is False:
        def second_order_force(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            value = calc_values(tot_der)
            jacobian = calc_jacobian(tot_der, ind_der)

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag
    else:
        wx, wy, wz = weights

        def second_order_force(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            # Tried to keep values and jacobian as single arrays and converting
            # weights to an array, but this seems to be the fastest implementation.
            Fx, Fy, Fz = calc_values(tot_der)
            dFx, dFy, dFz = calc_jacobian(tot_der, ind_der)
            value = wx * Fx + wy * Fy + wz * Fz
            jacobian = wx * dFx + wy * dFy + wz * dFz

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag
    return second_order_force


def second_order_stiffness(array, location, weights=None, spatial_derivatives=None, c_sphere=2350, rho_sphere=25, radius_sphere=1e-3):
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
    num_transducers = array.num_transducers
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location, orders=3)

    c_air = models.c_air
    rho_air = models.rho_air
    compressibility_air = 1 / (rho_air * c_air**2)
    compressibility_sphere = 1 / (rho_sphere * c_sphere**2)
    f_1 = 1 - compressibility_sphere / compressibility_air  # f_1 in H. Bruus 2012
    f_2 = 2 * (rho_sphere / rho_air - 1) / (2 * rho_sphere / rho_air + 1)   # f_2 in H. Bruus 2012

    ka = array.k * radius_sphere
    psi_0 = -2 * ka**6 / 9 * (f_1**2 + f_2**2 / 4 + f_1 * f_2) - 1j * ka**3 / 3 * (2 * f_1 + f_2)
    psi_1 = -ka**6 / 18 * f_2**2 + 1j * ka**3 / 3 * f_2
    force_coeff = -np.pi / array.k**5 * compressibility_air

    def calc_values(tot_der):
        Fxx = (1j * array.k**2 * (psi_0 * (tot_der[''] * np.conj(tot_der['xx']) + tot_der['x'] * np.conj(tot_der['x'])) +
                                  psi_1 * (tot_der['xx'] * np.conj(tot_der['']) + tot_der['x'] * np.conj(tot_der['x']))) +
               1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['xxx']) + tot_der['xx'] * np.conj(tot_der['xx']) +
                                 tot_der['y'] * np.conj(tot_der['xxy']) + tot_der['xy'] * np.conj(tot_der['xy']) +
                                 tot_der['z'] * np.conj(tot_der['xxz']) + tot_der['xz'] * np.conj(tot_der['xz']))
               ).real * force_coeff
        Fyy = (1j * array.k**2 * (psi_0 * (tot_der[''] * np.conj(tot_der['yy']) + tot_der['y'] * np.conj(tot_der['y'])) +
                                  psi_1 * (tot_der['yy'] * np.conj(tot_der['']) + tot_der['y'] * np.conj(tot_der['y']))) +
               1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['yyx']) + tot_der['xy'] * np.conj(tot_der['xy']) +
                                 tot_der['y'] * np.conj(tot_der['yyy']) + tot_der['yy'] * np.conj(tot_der['yy']) +
                                 tot_der['z'] * np.conj(tot_der['yyz']) + tot_der['yz'] * np.conj(tot_der['yz']))
               ).real * force_coeff
        Fzz = (1j * array.k**2 * (psi_0 * (tot_der[''] * np.conj(tot_der['zz']) + tot_der['z'] * np.conj(tot_der['z'])) +
                                  psi_1 * (tot_der['zz'] * np.conj(tot_der['']) + tot_der['z'] * np.conj(tot_der['z']))) +
               1j * 3 * psi_1 * (tot_der['x'] * np.conj(tot_der['zzx']) + tot_der['xz'] * np.conj(tot_der['xz']) +
                                 tot_der['y'] * np.conj(tot_der['zzy']) + tot_der['yz'] * np.conj(tot_der['yz']) +
                                 tot_der['z'] * np.conj(tot_der['zzz']) + tot_der['zz'] * np.conj(tot_der['zz']))
               ).real * force_coeff
        return np.array((Fxx, Fyy, Fzz))

    def calc_jacobian(tot_der, ind_der):
        dFxx = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['xx']) - np.conj(psi_0) * tot_der['xx'] * np.conj(ind_der['']) + (psi_0 - np.conj(psi_0)) * tot_der['x'] * np.conj(ind_der['x']) +
                                   psi_1 * tot_der['xx'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['xx']) + (psi_1 - np.conj(psi_1)) * tot_der['x'] * np.conj(ind_der['x'])) +
                1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['xxx']) - np.conj(psi_1) * tot_der['xxx'] * np.conj(ind_der['x']) + (psi_1 - np.conj(psi_1)) * tot_der['xx'] * np.conj(ind_der['xx']) +
                          psi_1 * tot_der['y'] * np.conj(ind_der['xxy']) - np.conj(psi_1) * tot_der['xxy'] * np.conj(ind_der['y']) + (psi_1 - np.conj(psi_1)) * tot_der['xy'] * np.conj(ind_der['xy']) +
                          psi_1 * tot_der['z'] * np.conj(ind_der['xxz']) - np.conj(psi_1) * tot_der['xxz'] * np.conj(ind_der['z']) + (psi_1 - np.conj(psi_1)) * tot_der['xz'] * np.conj(ind_der['xz']))
                ) * force_coeff
        dFyy = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['yy']) - np.conj(psi_0) * tot_der['yy'] * np.conj(ind_der['']) + (psi_0 - np.conj(psi_0)) * tot_der['y'] * np.conj(ind_der['y']) +
                                   psi_1 * tot_der['yy'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['yy']) + (psi_1 - np.conj(psi_1)) * tot_der['y'] * np.conj(ind_der['y'])) +
                1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['yyx']) - np.conj(psi_1) * tot_der['yyx'] * np.conj(ind_der['x']) + (psi_1 - np.conj(psi_1)) * tot_der['xy'] * np.conj(ind_der['xy']) +
                          psi_1 * tot_der['y'] * np.conj(ind_der['yyy']) - np.conj(psi_1) * tot_der['yyy'] * np.conj(ind_der['y']) + (psi_1 - np.conj(psi_1)) * tot_der['yy'] * np.conj(ind_der['yy']) +
                          psi_1 * tot_der['z'] * np.conj(ind_der['yyz']) - np.conj(psi_1) * tot_der['yyz'] * np.conj(ind_der['z']) + (psi_1 - np.conj(psi_1)) * tot_der['yz'] * np.conj(ind_der['yz']))
                ) * force_coeff
        dFzz = (1j * array.k**2 * (psi_0 * tot_der[''] * np.conj(ind_der['zz']) - np.conj(psi_0) * tot_der['zz'] * np.conj(ind_der['']) + (psi_0 - np.conj(psi_0)) * tot_der['z'] * np.conj(ind_der['z']) +
                                   psi_1 * tot_der['zz'] * np.conj(ind_der['']) - np.conj(psi_1) * tot_der[''] * np.conj(ind_der['zz']) + (psi_1 - np.conj(psi_1)) * tot_der['z'] * np.conj(ind_der['z'])) +
                1j * 3 * (psi_1 * tot_der['x'] * np.conj(ind_der['zzx']) - np.conj(psi_1) * tot_der['zzx'] * np.conj(ind_der['x']) + (psi_1 - np.conj(psi_1)) * tot_der['xz'] * np.conj(ind_der['xz']) +
                          psi_1 * tot_der['y'] * np.conj(ind_der['zzy']) - np.conj(psi_1) * tot_der['zzy'] * np.conj(ind_der['y']) + (psi_1 - np.conj(psi_1)) * tot_der['yz'] * np.conj(ind_der['yz']) +
                          psi_1 * tot_der['z'] * np.conj(ind_der['zzz']) - np.conj(psi_1) * tot_der['zzz'] * np.conj(ind_der['z']) + (psi_1 - np.conj(psi_1)) * tot_der['zz'] * np.conj(ind_der['zz']))
                ) * force_coeff
        return np.array((dFxx, dFyy, dFzz))

    if weights is None:
        def second_order_stiffness(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            tot_der = {}
            for key, value in spatial_derivatives.items():
                tot_der[key] = np.sum(complex_coeff * value)

            return calc_values(tot_der)
    elif weights is False:
        def second_order_stiffness(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            value = calc_values(tot_der)
            jacobian = calc_jacobian(tot_der, ind_der)

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag
    else:
        wx, wy, wz = weights

        def second_order_stiffness(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)
            ind_der = {}
            tot_der = {}
            for key, value in spatial_derivatives.items():
                ind_der[key] = complex_coeff * value
                tot_der[key] = np.sum(ind_der[key])

            # Tried to keep values and jacobian as single arrays and converting
            # weights to an array, but this seems to be the fastest implementation.
            Fxx, Fyy, Fzz = calc_values(tot_der)
            dFxx, dFyy, dFzz = calc_jacobian(tot_der, ind_der)
            value = wx * Fxx + wy * Fyy + wz * Fzz
            jacobian = wx * dFxx + wy * dFyy + wz * dFzz

            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag
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
        # Note that this only makes sense as a cost function, and only for variable amplitudes,
        # so no implementation for complex inputs is needed.
        _, amplitudes, variable_amps = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
        if not variable_amps:
            return 0, np.zeros(num_transducers)
        under_idx = amplitudes < lower_bound
        over_idx = amplitudes > upper_bound
        under = scaling * (lower_bound - amplitudes[under_idx])
        over = scaling * (amplitudes[over_idx] - upper_bound)

        value = (under**order + over**order).sum()
        jacobian = np.zeros(2 * num_transducers)
        jacobian[num_transducers + under_idx] = under**(order - 1) * order
        jacobian[num_transducers + over_idx] = over**(order - 1) * order

        return value, jacobian
    return amplitude_limiting


def pressure_null(array, location, weights=None, spatial_derivatives=None):
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
    num_transducers = array.num_transducers
    if spatial_derivatives is None:
        spatial_derivatives = array.spatial_derivatives(location, orders=1)
    else:
        input_ders = spatial_derivatives
        spatial_derivatives = {'': input_ders[''],
                               'x': input_ders['x'],
                               'y': input_ders['y'],
                               'z': input_ders['z']}

    def calc_values(complex_coeff):
        p = np.sum(complex_coeff * spatial_derivatives[''])
        px = np.sum(complex_coeff * spatial_derivatives['x'])
        py = np.sum(complex_coeff * spatial_derivatives['y'])
        pz = np.sum(complex_coeff * spatial_derivatives['y'])

        return np.array((p, px, py, pz))

    def calc_jacobian(complex_coeff, values):
        p, px, py, pz = values
        dp = 2 * p * np.conj(complex_coeff * spatial_derivatives[''])
        dpx = 2 * px * np.conj(complex_coeff * spatial_derivatives['x'])
        dpy = 2 * py * np.conj(complex_coeff * spatial_derivatives['y'])
        dpz = 2 * pz * np.conj(complex_coeff * spatial_derivatives['z'])

        return np.array((dp, dpx, dpy, dpz))

    if weights is None:
        def pressure_null(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=True)
            complex_coeff = amplitudes * np.exp(1j * phases)
            return calc_values(complex_coeff)
    elif weights is False:
        def pressure_null(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)

            complex_value = calc_values(complex_coeff)
            jacobian = calc_jacobian(complex_coeff, complex_value)
            value = np.abs(complex_value)**2
            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes), axis=-1)
            else:
                return value, jacobian.imag
    else:
        try:
            if len(weights) == 4:
                weights = np.asarray(weights)
            elif len(weights) == 3:
                weights = np.concatenate(([0], weights))
        except TypeError:
            weights = np.array((weights, 0, 0, 0))

        def pressure_null(phases_amplitudes):
            phases, amplitudes, variable_amplitudes = _phase_and_amplitude_input(phases_amplitudes, num_transducers, allow_complex=False)
            complex_coeff = amplitudes * np.exp(1j * phases)

            complex_vals = calc_values(complex_coeff)
            jacobian = np.sum(calc_jacobian(complex_coeff, complex_vals) * weights[:, np.newaxis], axis=0)
            value = np.sum(np.abs(complex_vals)**2 * weights)
            if variable_amplitudes:
                return value, np.concatenate((jacobian.imag, jacobian.real / amplitudes))
            else:
                return value, jacobian.imag
    return pressure_null
