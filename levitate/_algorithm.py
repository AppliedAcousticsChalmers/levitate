import numpy as np
import functools
import textwrap


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
            obj = Algorithm(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__)
        elif weight is None:
            obj = BoundAlgorithm(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, position=position)
        elif position is None:
            obj = UnboundCostFunction(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, weight=weight)
        elif weight is not None and position is not None:
            obj = CostFunction(array, calc_values=values, calc_jacobians=jacobians, name=func.__name__, weight=weight, position=position)
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


def requires(**requirements):
    possible_requirements = [
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

    def __init__(self, array, *, calc_values, calc_jacobians=None, name=None):
        self.name = name
        self.calc_values = calc_values
        self.calc_jacobians = calc_jacobians
        self.array = array
        self.requires = calc_values.requires.copy()

    def __call__(self, complex_transducer_amplitudes, position):
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})

    def _evaluate_requirements(self, complex_transducer_amplitudes, spatial_structures):
        requirements = {}
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
            else:
                raise ValueError("Unknown requirement '{}'".format(key))
        # Replace the requets with values calculated by the array
        if 'pressure_derivs' in spatial_structures:
            spatial_structures['pressure_derivs'] = self.array.pressure_derivs(position, orders=spatial_structures['pressure_derivs'])
        if 'spherical_harmonics' in spatial_structures:
            spatial_structures['spherical_harmonics'] = self.array.spherical_harmonics(position, orders=spatial_structures['spherical_harmonics'])
        return spatial_structures

    def __mul__(self, weight):
        return UnboundCostFunction(array=self.array, name=self.name, weight=weight,
                                   calc_values=self.calc_values, calc_jacobians=self.calc_jacobians)

    def __rmul__(self, weight):
        return self.__mul__(weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return BoundAlgorithm(array=self.array, name=self.name, position=position,
                              calc_values=self.calc_values, calc_jacobians=self.calc_jacobians)

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
        return VectorBase(algorithm=self, target_vector=vector)

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

    def __init__(self, array, *, calc_values, position, calc_jacobians=None, name=None, **kwargs):
        super().__init__(array=array, calc_values=calc_values, calc_jacobians=calc_jacobians, name=name, **kwargs)
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
            other.weights  # If other has weights it's a cost function type object and should not be addable with self
            if np.allclose(self.position, other.position):
                return super().__add__(other)
            else:
                return AlgorithmCollection(self, other)
        except AttributeError:
            return NotImplemented

    def __mul__(self, weight):
        weight = np.atleast_1d(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostFunction(array=self.array, name=self.name, weight=weight, position=self.position,
                            calc_values=self.calc_values, calc_jacobians=self.calc_jacobians)


class UnboundCostFunction(Algorithm):
    _str_format_spec = '{:%cls%name%weight}'

    def __init__(self, array, *, calc_values, calc_jacobians, weight, name=None, **kwargs):
        super().__init__(array=array, calc_values=calc_values, calc_jacobians=calc_jacobians, name=name, **kwargs)
        self.weight = np.atleast_1d(weight)
        for key, value in calc_jacobians.requires.items():
            self.requires[key] = max(value, self.requires.get(key, -1))

    def __call__(self, complex_transducer_amplitudes, position):
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})
        jacobians = self.calc_jacobians(**{key: requirements[key] for key in self.calc_jacobians.requires})
        return np.einsum('i, i...', self.weight, values), np.einsum('i, i...', self.weight, jacobians)

    def __mul__(self, weight):
        return super().__mul__(self.weight * weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return CostFunction(array=self.array, name=self.name, weight=self.weight, position=position,
                            calc_values=self.calc_values, calc_jacobians=self.calc_jacobians)


class CostFunction(UnboundCostFunction, BoundAlgorithm):
    _str_format_spec = '{:%cls%name%weight%position}'

    # Inharitance order is important here, we need to resolve to UnboundCostFunction.__mul__ and not BoundAlgorithm.__mul__
    def __init__(self, array, *, calc_values, calc_jacobians, weight, position, name=None, **kwargs):
        super().__init__(array=array, calc_values=calc_values, calc_jacobians=calc_jacobians, name=name, weight=weight, position=position, **kwargs)

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.calc_values(**{key: requirements[key] for key in self.calc_values.requires})
        jacobians = self.calc_jacobians(**{key: requirements[key] for key in self.calc_jacobians.requires})
        return np.einsum('i, i...', self.weight, values), np.einsum('i, i...', self.weight, jacobians)


class VectorBase:

    def __new__(cls, algorithm, *, target_vector, **kwargs):
        alg_type = type(algorithm)
        obj = alg_type.__new__(alg_type)
        obj.__class__ = type('Vector{}'.format(alg_type.__name__), (VectorBase, alg_type), {})
        return obj

    def __init__(self, algorithm, *, target_vector, **kwargs):
        target_vector = np.asarray(target_vector)
        self.target_vector = target_vector
        if hasattr(algorithm, 'weight'):
            kwargs['weight'] = algorithm.weight
        if hasattr(algorithm, 'position'):
            kwargs['position'] = algorithm.position

        @functools.wraps(algorithm.calc_values)
        def calc_values(**kwargs):
            values = algorithm.calc_values(**kwargs)
            values -= target_vector.reshape([-1] + (values.ndim - 1) * [1])
            return np.real(values * np.conj(values))
        calc_values.requires = algorithm.calc_values.requires.copy()
        calc_values.is_vector = True

        # This form would be better since it does not rely on the explicit arguments,
        # but it requires that the arguments of all algorithms are in line with the requirements
        @functools.wraps(algorithm.calc_jacobians)
        def calc_jacobians(**kwargs):
            values = algorithm.calc_values(**{key: kwargs[key] for key in algorithm.calc_values.requires})
            values -= self.target_vector.reshape([-1] + (values.ndim - 1) * [1])
            jacobians = algorithm.calc_jacobians(**{key: kwargs[key] for key in algorithm.calc_jacobians.requires})
            return 2 * np.einsum('ij..., i...->ij...', jacobians, values)
        calc_jacobians.requires = algorithm.calc_jacobians.requires.copy()
        for key, value in algorithm.calc_values.requires.items():
            calc_jacobians.requires[key] = max(value, calc_jacobians.requires.get(key, -1))
        calc_jacobians.is_vector = True

        super().__init__(array=algorithm.array, calc_values=calc_values, calc_jacobians=calc_jacobians, name=algorithm.name, **kwargs)

    def __matmul__(self, position):
        obj = super().__matmul__(position)
        alg_type = type(obj)
        obj.__class__ = type('Vector{}'.format(alg_type.__name__), (VectorBase, alg_type), {})
        obj.target_vector = self.target_vector
        return obj

    def __mul__(self, weight):
        obj = super().__mul__(weight)
        alg_type = type(obj)
        obj.__class__ = type('Vector{}'.format(alg_type.__name__), (VectorBase, alg_type), {})
        obj.target_vector = self.target_vector
        return obj

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other) or type(other) == type(self).__bases__[1]:
            return AlgorithmPoint(self, other)
        else:
            return NotImplemented

    def __format__(self, format_spec):
        format_spec = format_spec.replace('%name', '||%name - %vector||^2').replace('%vector', str(self.target_vector))
        return super().__format__(format_spec)


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
        self.array = algorithms[0].array
        self.algorithms = []
        self.requires = {}
        for algorithm in algorithms:
            self += algorithm

    def __call__(self, complex_transducer_amplitudes, position):
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return [algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        if type(other) in type(self).__bases__:
            new = type(self)(*self.algorithms, other)
        elif type(other) == type(self):
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
            value += np.einsum('i,i...', algorithm.weight, algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}))
            jacobians += np.einsum('i, i...', algorithm.weight, algorithm.calc_jacobians(**{key: requirements[key] for key in algorithm.calc_jacobians.requires}))
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
            value += np.einsum('i,i...', algorithm.weight, algorithm.calc_values(**{key: requirements[key] for key in algorithm.calc_values.requires}))
            jacobians += np.einsum('i, i...', algorithm.weight, algorithm.calc_jacobians(**{key: requirements[key] for key in algorithm.calc_jacobians.requires}))
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
