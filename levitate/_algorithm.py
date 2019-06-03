import numpy as np
import functools
import textwrap


def algorithm(ndim):
    def algorithm(func):
        func.__doc__ = func.__doc__ or 'Parameters\n----------\n'

        @functools.wraps(func)
        def wrapper(array, *args, weight=None, position=None, **kwargs):
            outputs = func(array, *args, **kwargs)
            try:
                values, jacobians = outputs
            except TypeError:
                values = outputs
                jacobians = None
            if weight is None and position is None:
                obj = Algorithm(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, ndim=ndim)
            elif weight is None:
                obj = BoundAlgorithm(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, ndim=ndim, position=position)
            elif position is None:
                obj = UnboundCostFunction(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, ndim=ndim, weight=weight)
            elif weight is not None and position is not None:
                obj = CostFunction(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, ndim=ndim, weight=weight, position=position)
            return obj
        wrapper.__doc__ = textwrap.dedent(wrapper.__doc__).rstrip('\n') + textwrap.dedent("""
            weight : numeric, optional
                Directly converting the algorithm to a cost function for use in optimizations.
            position : 3 element numeric, optional
                Directly binds the algorithm to one or more points in space for efficient execution.

            Returns
            -------
            algorithm : `Algorithm`
                An Algorithm object or a subclass thereof, depending on whether weight or position
                was supplied in the call.
            """)
        return wrapper
    return algorithm


class AlgorithmImplementation:
    def __new__(cls, array, *cls_args, weight=None, position=None, _nowrap=False, **cls_kwargs):
        obj = object.__new__(cls)  # Call __new__ from object to use a "normal" __new__ method. calling cls.__new__ would give infinite recursion
        if _nowrap is True:
            return obj
        obj.__init__(array, *cls_args, **cls_kwargs)  # Manually instantiate the object since the automatic __init__ call is skipped
        if weight is None and position is None:
            alg = Algorithm(algorithm=obj)
        elif weight is None:
            alg = BoundAlgorithm(algorithm=obj, position=position)
        elif position is None:
            alg = UnboundCostFunction(algorithm=obj, weight=weight)
        elif weight is not None and position is not None:
            alg = CostFunction(algorithm=obj, weight=weight, position=position)
        return alg

    def __init__(self, array, *args, **kwargs):
        self.array = array

    def __getnewargs_ex__(self):
        return (self.array,), {'_nowrap': True}


def requires(**requirements):
    possible_requirements = [
        'complex_transducer_amplitudes',
        'pressure_derivs_summed', 'pressure_derivs_individual',
        'spherical_harmonics_summed', 'spherical_harmonics_individual',
    ]
    for requirement in requirements:
        if requirement not in possible_requirements:
            raise NotImplementedError("Requirement '{}' is not implemented for an algorithm. The possible requests are: {}".format(requirement, possible_requirements))

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        wrapped.requires = requirements
        return wrapped
    return wrapper


class Algorithm:
    _str_format_spec = '{:%cls%name}'

    def __init__(self, algorithm):
        self.algorithm = algorithm
        value_indices = ''.join(chr(ord('i') + idx) for idx in range(self.ndim))
        self._sum_str = value_indices + ', ' + value_indices + '...'
        self.requires = self.calc_values.requires.copy()

    @property
    def name(self):
        return self.algorithm.__class__.__name__
    @property
    def calc_values(self):
        return self.algorithm.calc_values
    @property
    def calc_jacobians(self):
        return self.algorithm.calc_jacobians
    @property
    def ndim(self):
        return self.algorithm.ndim
    @property
    def array(self):
        return self.algorithm.array

    def __call__(self, complex_transducer_amplitudes, position):
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})

    def _evaluate_requirements(self, complex_transducer_amplitudes, spatial_structures):
        requirements = {}
        if 'complex_transducer_amplitudes' in self.requires:
            requirements['complex_transducer_amplitudes'] = complex_transducer_amplitudes
        if 'pressure_derivs' in spatial_structures:
            requirements['pressure_derivs_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, spatial_structures['pressure_derivs'])
            requirements['pressure_derivs_summed'] = np.sum(requirements['pressure_derivs_individual'], axis=1)
        if 'spherical_harmonics' in spatial_structures:
            requirements['spherical_harmonics_individual'] = np.einsum('i,ji...->ji...', complex_transducer_amplitudes, spatial_structures['spherical_harmonics'])
            requirements['spherical_harmonics_summed'] = np.sum(requirements['spherical_harmonics_individual'], axis=1)
        return requirements

    def _spatial_structures(self, position):
        # Check what spatial structures we need from the array to fulfill the requirements
        spatial_structures = {}
        for key, value in self.requires.items():
            if key.find('pressure_derivs') > -1:
                spatial_structures['pressure_derivs'] = max(value, spatial_structures.get('pressure_derivs', -1))
            elif key.find('spherical_harmonics') > -1:
                spatial_structures['spherical_harmonics'] = max(value, spatial_structures.get('spherical_harmonics', -1))
            elif key != 'complex_transducer_amplitudes':
                raise ValueError("Unknown requirement '{}'".format(key))
        # Replace the requets with values calculated by the array
        if 'pressure_derivs' in spatial_structures:
            spatial_structures['pressure_derivs'] = self.array.pressure_derivs(position, orders=spatial_structures['pressure_derivs'])
        if 'spherical_harmonics' in spatial_structures:
            spatial_structures['spherical_harmonics'] = self.array.spherical_harmonics(position, orders=spatial_structures['spherical_harmonics'])
        return spatial_structures

    def __mul__(self, weight):
        return UnboundCostFunction(weight=weight, algorithm=self.algorithm)

    def __rmul__(self, weight):
        return self.__mul__(weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return BoundAlgorithm(position=position, algorithm=self.algorithm)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return AlgorithmPoint(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, vector):
        return VectorAlgorithm(algorithm=self, target_vector=vector)

    def __str__(self, not_api_call=True):
        return self._str_format_spec.format(self)

    def __format__(self, format_spec):
        cls = self.__class__.__name__ + ': '
        name = getattr(self, 'name', None) or 'Unknown'
        weight = getattr(self, 'weight', None)
        pos = getattr(self, 'position', None)
        weight = ' * ' + str(weight) if weight is not None else ''
        pos = ' @ ' + str(pos) if pos is not None else ''
        return format_spec.replace('%cls', cls).replace('%name', name).replace('%weight', weight).replace('%position', pos).replace('%algorithms', '')


class BoundAlgorithm(Algorithm):
    _str_format_spec = '{:%cls%name%position}'

    def __init__(self, algorithm, position, **kwargs):
        super().__init__(algorithm=algorithm, **kwargs)
        self.position = position

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})

    def _spatial_structures(self):
        try:
            return self._cashed_spatial_structures
        except AttributeError:
            self._cashed_spatial_structures = super()._spatial_structures(self.position)
            return self._cashed_spatial_structures

    def __add__(self, other):
        if other == 0:
            return self
        try:
            if np.allclose(self.position, other.position):
                return super().__add__(other)
            else:
                return AlgorithmCollection(self, other)
        except AttributeError:
            return NotImplemented

    def __sub__(self, vector):
        return VectorBoundAlgorithm(algorithm=self, target_vector=vector, position=self.position)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostFunction(weight=weight, position=self.position, algorithm=self.algorithm)


class UnboundCostFunction(Algorithm):
    _str_format_spec = '{:%cls%name%weight}'

    def __init__(self, algorithm, weight, **kwargs):
        super().__init__(algorithm=algorithm, **kwargs)
        self.weight = np.asarray(weight)
        if self.weight.ndim < self.ndim:
            extra_dims = self.ndim - self.weight.ndim
            self.weight.shape = (1,) * extra_dims + self.weight.shape
        for key, value in self.calc_jacobians.requires.items():
            self.requires[key] = max(value, self.requires.get(key, -1))

    def __call__(self, complex_transducer_amplitudes, position):
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})
        jacobians = self.calc_jacobians(**{key: requirements[key] for key in self.calc_jacobians.requires})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __mul__(self, weight):
        return super().__mul__(self.weight * weight)

    def __sub__(self, vector):
        return VectorUnboundCostFunction(algorithm=self, target_vector=vector, weight=self.weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return CostFunction(weight=self.weight, position=position, algorithm=self.algorithm)


class CostFunction(UnboundCostFunction, BoundAlgorithm):
    _str_format_spec = '{:%cls%name%weight%position}'

    # Inharitance order is important here, we need to resolve to UnboundCostFunction.__mul__ and not BoundAlgorithm.__mul__
    def __init__(self, algorithm, weight, position, **kwargs):
        super().__init__(algorithm=algorithm, weight=weight, position=position, **kwargs)

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})
        jacobians = self.calc_jacobians(**{key: requirements[key] for key in self.calc_jacobians.requires})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __sub__(self, vector):
        return VectorCostFunction(algorithm=self, target_vector=vector, weight=self.weight, position=self.position)


class VectorBase(Algorithm):
    def __init__(self, algorithm, target_vector, **kwargs):
        if type(self) == VectorBase:
            raise AssertionError('`VectorBase` should never be directly instantiated!')
        self.calc_values.requires.update(algorithm.calc_values.requires)
        self.calc_jacobians.requires.update(algorithm.calc_jacobians.requires)
        for key, value in algorithm.calc_values.requires.items():
            self.calc_jacobians.requires[key] = max(value, self.calc_jacobians.requires.get(key, -1))
        super().__init__(algorithm=algorithm, **kwargs)
        target_vector = np.asarray(target_vector)
        self.target_vector = target_vector

    @property
    def name(self):
        return self.algorithm.name

    @requires()
    def calc_values(self, **kwargs):
        values = self.algorithm.calc_values(**kwargs)
        values -= self.target_vector.reshape([-1] + (values.ndim - 1) * [1])
        return np.real(values * np.conj(values))

    @requires()
    def calc_jacobians(self, **kwargs):
        values = self.algorithm.calc_values(**{key: kwargs[key] for key in self.algorithm.calc_values.requires})
        values -= self.target_vector.reshape([-1] + (values.ndim - 1) * [1])
        jacobians = self.algorithm.calc_jacobians(**{key: kwargs[key] for key in self.algorithm.calc_jacobians.requires})
        return 2 * jacobians * values.reshape(values.shape[:self.ndim] + (1,) + values.shape[self.ndim:])

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if VectorBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            return AlgorithmPoint(self, other)
        else:
            return NotImplemented

    def __format__(self, format_spec):
        format_spec = format_spec.replace('%name', '||%name - %vector||^2').replace('%vector', str(self.target_vector))
        return super().__format__(format_spec)


class VectorAlgorithm(VectorBase, Algorithm):
    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorBoundAlgorithm(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorUnboundCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight)


class VectorBoundAlgorithm(VectorBase, BoundAlgorithm):
    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorBoundAlgorithm(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight, position=algorithm.position)


class VectorUnboundCostFunction(VectorBase, UnboundCostFunction):
    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position, weight=algorithm.weight)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorUnboundCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight)


class VectorCostFunction(VectorBase, CostFunction):
    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position, weight=algorithm.weight)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight, position=algorithm.position)


class AlgorithmPoint(Algorithm):
    _str_format_spec = '{:%cls%algorithms%position}'

    def __new__(cls, *algorithms):
        is_bound = isinstance(algorithms[0], BoundAlgorithm)
        is_cost = isinstance(algorithms[0], UnboundCostFunction)
        is_collection = False
        if is_bound:
            pos_0 = algorithms[0].position
            for algorithm in algorithms:
                if not np.allclose(pos_0, algorithm.position):
                    is_collection = True
                    break

        if is_collection and is_cost:
            new_cls = CostFunctionCollection
        elif is_collection:
            new_cls = AlgorithmCollection
        elif is_cost and is_bound:
            new_cls = CostFunctionPoint
        elif is_cost:
            new_cls = UnboundCostFunctionPoint
        elif is_bound:
            new_cls = BoundAlgorithmPoint
        elif not (is_collection and is_bound and is_cost):
            new_cls = AlgorithmPoint
        obj = object.__new__(new_cls)
        return obj

    def __init__(self, *algorithms):
        self.algorithms = []
        self.requires = {}
        for algorithm in algorithms:
            self += algorithm

    def __getnewargs__(self):
        return tuple(self.algorithms)

    @property
    def array(self):
        return self.algorithms[0].array

    def __call__(self, complex_transducer_amplitudes, position):
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return [algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if VectorBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type in type(self).__bases__:
            new = type(self)(*self.algorithms, other)
        elif other_type == type(self):
            new = type(self)(*self.algorithms, *other.algorithms)
        else:
            return NotImplemented
        return new

    def __iadd__(self, other):
        add_element = False
        add_point = False
        if type(other) in type(self).__bases__:
            add_element = True
        elif VectorBase in type(other).__bases__ and type(other).__bases__[1] in type(self).__bases__:
            add_element = True
        elif type(other) == type(self):
            add_point = True

        if add_element:
            for key, value in other.requires.items():
                self.requires[key] = max(value, self.requires.get(key, -1))
            self.algorithms.append(other)
        elif add_point:
            for algorithm in other.algorithms:
                self += algorithm
        else:
            return NotImplemented
        return self

    def __mul__(self, weight):
        return UnboundCostFunctionPoint(*[algorithm * weight for algorithm in self.algorithms])

    def __matmul__(self, position):
        return BoundAlgorithmPoint(*[algorithm @ position for algorithm in self.algorithms])

    def __format__(self, format_spec):
        if '%algorithms' in format_spec:
            alg_start = format_spec.find('%algorithms')
            if len(format_spec) > alg_start + 11 and format_spec[alg_start + 11] == ':':
                alg_spec_len = format_spec[alg_start + 12].find(':')
                alg_spec = format_spec[alg_start + 12:alg_start + 12 + alg_spec_len]
                pre = format_spec[:alg_start + 10]
                post = format_spec[alg_start + 13 + alg_spec_len:]
                format_spec = pre + post
            else:
                alg_spec = '{:%name%weight}'
            alg_str = '('
            for algorithm in self.algorithms:
                alg_str += alg_spec.format(algorithm) + ' + '
            format_spec = format_spec.replace('%algorithms', alg_str.rstrip(' + ') + ')')
        return super().__format__(format_spec.replace('%name', ''))

    def __sub__(self, other):
        return type(self)(*[algorithm - other for algorithm in self.algorithms])


class BoundAlgorithmPoint(AlgorithmPoint, BoundAlgorithm):
    def __init__(self, *algorithms):
        self.position = algorithms[0].position
        super().__init__(*algorithms)

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return [algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}) for algorithm in self.algorithms]

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return AlgorithmCollection(self, other)
        except AttributeError:
            return NotImplemented

    def __mul__(self, weight):
        return CostFunctionPoint(*[algorithm * weight for algorithm in self.algorithms])


class UnboundCostFunctionPoint(AlgorithmPoint, UnboundCostFunction):
    def __call__(self, complex_transducer_amplitudes, position):
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.calc_jacobians(**{key: requirements[key] for key in algorithm.calc_jacobians.requires}))
        return value, jacobians

    def __matmul__(self, position):
        return CostFunctionPoint(*[algorithm @ position for algorithm in self.algorithms])


class CostFunctionPoint(UnboundCostFunctionPoint, BoundAlgorithmPoint, CostFunction):
    def __call__(self, complex_transducer_amplitudes,):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.calc_jacobians(**{key: requirements[key] for key in algorithm.calc_jacobians.requires}))
        return value, jacobians


class AlgorithmCollection(BoundAlgorithmPoint):
    _str_format_spec = '{:%cls%points}'

    def __new__(cls, *algorithms):
        if isinstance(algorithms[0], CostFunction):
            new_cls = CostFunctionCollection
        elif isinstance(algorithms[0], BoundAlgorithm):
            new_cls = AlgorithmCollection
        obj = object.__new__(new_cls)
        return obj

    def __init__(self, *algorithms):
        self.algorithms = []
        for algorithm in algorithms:
            self += algorithm

    def __call__(self, complex_transducer_amplitudes):
        values = []
        for point in self.algorithms:
            values.append(point(complex_transducer_amplitudes))
        return values

    def __add__(self, other):
        if other == 0:
            return self
        elif isinstance(self, CostFunctionCollection) and not isinstance(other, CostFunction):
            # Make sure that we are not adding bound algorithms to cost function collections
            return NotImplemented
        elif isinstance(other, CostFunction) and not isinstance(self, CostFunctionCollection):
            # Make sure that we are not adding cost functions to algorithm collections
            return NotImplemented
        try:
            other.position
        except AttributeError:
            return NotImplemented
        else:
            return AlgorithmCollection(*self.algorithms, other)

    def __iadd__(self, other):
        if type(other) == type(self):
            for algorithm in other.algorithms:
                self += algorithm
            return self
        elif isinstance(self, CostFunctionCollection) and not isinstance(other, CostFunction):
            # Make sure that we are not adding bound algorithms to cost function collections
            return NotImplemented
        elif isinstance(other, CostFunction) and not isinstance(self, CostFunctionCollection):
            # Make sure that we are not adding cost functions to algorithm collections
            return NotImplemented
        try:
            other_pos = other.position
        except AttributeError:
            return NotImplemented
        else:
            for idx, point in enumerate(self.algorithms):
                if np.allclose(point.position, other_pos):
                    # Mutating `point` will not update the contents in the list!
                    self.algorithms[idx] += other
                    break
            else:
                self.algorithms.append(other)
            return self

    def __format__(self, format_spec):
        if '%points' in format_spec:
            points_start = format_spec.find('%points')
            if len(format_spec) > points_start + 7 and format_spec[points_start + 7] == ':':
                points_spec_len = format_spec[points_start + 8].rind(':')
                points_spec = format_spec[points_start + 8:points_start + 8 + points_spec_len]
                pre = format_spec[:points_start + 6]
                post = format_spec[points_start + 9 + points_spec_len:]
                format_spec = pre + post
            else:
                points_spec = '\t{:%cls%name%algorithms%weight%position}\n'
            points_str = '[\n'
            for algorithm in self.algorithms:
                points_str += points_spec.format(algorithm)
            format_spec = format_spec.replace('%points', points_str + ']')
        return super().__format__(format_spec)


class CostFunctionCollection(AlgorithmCollection, CostFunctionPoint):
    def __call__(self, complex_transducer_amplitudes):
        values = 0
        jacobians = 0
        for algorithm in self.algorithms:
            val, jac = algorithm(complex_transducer_amplitudes)
            values += val
            jacobians += jac
        return values, jacobians
