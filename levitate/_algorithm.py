import numpy as np


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

    def __eq__(self, other):
        return type(self) == type(other) and self.array == other.array

    def __getnewargs_ex__(self):
        return (self.array,), {'_nowrap': True}


def requirement(**requirements):
    possible_requirements = [
        'complex_transducer_amplitudes',
        'pressure_derivs_summed', 'pressure_derivs_individual',
        'spherical_harmonics_summed', 'spherical_harmonics_individual',
    ]
    for requirement in requirements:
        if requirement not in possible_requirements:
            raise NotImplementedError("Requirement '{}' is not implemented for an algorithm. The possible requests are: {}".format(requirement, possible_requirements))
    return requirements


class AlgorithmMeta(type):
    # This class makes sure that _type is available at the class level.
    @property
    def _type(cls):
        return cls._is_bound, cls._is_cost


class AlgorithmBase(metaclass=AlgorithmMeta):

    @property
    def _type(self):
        # Binds the _type instance property to the class property.
        return type(self)._type

    def __eq__(self, other):
        return type(self) == type(other)

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

    def _spatial_structures(self, position=None):
        # If called without a position we are using a bound algorithm, check the cache and calculate it if needed
        if position is None:
            try:
                return self._cached_spatial_structures
            except AttributeError:
                self._cached_spatial_structures = self._spatial_structures(self.position)
                return self._cached_spatial_structures
        # Check what spatial structures we need from the array to fulfill the requirements
        spatial_structures = {}
        for key, value in self.requires.items():
            if key.find('pressure_derivs') > -1:
                spatial_structures['pressure_derivs'] = max(value, spatial_structures.get('pressure_derivs', -1))
            elif key.find('spherical_harmonics') > -1:
                spatial_structures['spherical_harmonics'] = max(value, spatial_structures.get('spherical_harmonics', -1))
            elif key != 'complex_transducer_amplitudes':
                raise ValueError("Unknown requirement '{}'".format(key))
        # Replace the requests with values calculated by the array
        if 'pressure_derivs' in spatial_structures:
            spatial_structures['pressure_derivs'] = self.array.pressure_derivs(position, orders=spatial_structures['pressure_derivs'])
        if 'spherical_harmonics' in spatial_structures:
            spatial_structures['spherical_harmonics'] = self.array.spherical_harmonics(position, orders=spatial_structures['spherical_harmonics'])
        return spatial_structures

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, weight):
        return self.__mul__(weight)

    def __str__(self, not_api_call=True):
        return self._str_format_spec.format(self)

    def __format__(self, format_spec):
        cls = self.__class__.__name__ + ': '
        name = getattr(self, 'name', None) or 'Unknown'
        weight = getattr(self, 'weight', None)
        pos = getattr(self, 'position', None)
        weight = ' * ' + str(weight) if weight is not None else ''
        pos = ' @ ' + str(pos) if pos is not None else ''
        return format_spec.replace('%cls', cls).replace('%weight', weight).replace('%position', pos)


class Algorithm(AlgorithmBase):
    _str_format_spec = '{:%cls%name}'
    _is_bound = False
    _is_cost = False

    def __init__(self, algorithm):
        self.algorithm = algorithm
        value_indices = ''.join(chr(ord('i') + idx) for idx in range(self.ndim))
        self._sum_str = value_indices + ', ' + value_indices + '...'
        self.requires = self.algorithm.values_require.copy()

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.algorithm == other.algorithm
            and self.array == other.array
        )

    @property
    def name(self):
        return self.algorithm.__class__.__name__
    @property
    def values(self):
        return self.algorithm.values
    @property
    def jacobians(self):
        return self.algorithm.jacobians
    @property
    def values_require(self):
        return self.algorithm.values_require
    @property
    def jacobians_require(self):
        return self.algorithm.jacobians_require
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
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return AlgorithmPoint(self, other)
        else:
            return NotImplemented

    def __sub__(self, vector):
        return VectorAlgorithm(algorithm=self, target_vector=vector)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return UnboundCostFunction(weight=weight, algorithm=self.algorithm)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return BoundAlgorithm(position=position, algorithm=self.algorithm)

    def __format__(self, format_spec):
        name = getattr(self, 'name', None) or 'Unknown'
        return super().__format__(format_spec.replace('%name', name))


class BoundAlgorithm(Algorithm):
    _str_format_spec = '{:%cls%name%position}'
    _is_bound = True
    _is_cost = False

    def __init__(self, algorithm, position, **kwargs):
        super().__init__(algorithm=algorithm, **kwargs)
        self.position = position

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.position, other.position)
        )

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return self.values(**{key: requirements[key] for key in self.values_require})

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return BoundAlgorithmPoint(self, other)
            else:
                return AlgorithmCollection(self, other)
        else:
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
    _is_bound = False
    _is_cost = True

    def __init__(self, algorithm, weight, **kwargs):
        super().__init__(algorithm=algorithm, **kwargs)
        self.weight = np.asarray(weight)
        if self.weight.ndim < self.ndim:
            extra_dims = self.ndim - self.weight.ndim
            self.weight.shape = (1,) * extra_dims + self.weight.shape
        for key, value in self.jacobians_require.items():
            self.requires[key] = max(value, self.requires.get(key, -1))

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.weight, other.weight)
        )

    def __call__(self, complex_transducer_amplitudes, position):
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return UnboundCostFunctionPoint(self, other)
        else:
            return NotImplemented

    def __sub__(self, vector):
        return VectorUnboundCostFunction(algorithm=self, target_vector=vector, weight=self.weight)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return UnboundCostFunction(self.algorithm, self.weight * weight)

    def __matmul__(self, position):
        position = np.asarray(position)
        if position.ndim < 1 or position.shape[0] != 3:
            return NotImplemented
        return CostFunction(weight=self.weight, position=position, algorithm=self.algorithm)


class CostFunction(UnboundCostFunction, BoundAlgorithm):
    _str_format_spec = '{:%cls%name%weight%position}'
    _is_bound = True
    _is_cost = True

    # Inheritance order is important here, we need to resolve to UnboundCostFunction.__mul__ and not BoundAlgorithm.__mul__
    def __init__(self, algorithm, weight, position, **kwargs):
        super().__init__(algorithm=algorithm, weight=weight, position=position, **kwargs)

    def __eq__(self, other):
        return super().__eq__(other)

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        values = self.values(**{key: requirements[key] for key in self.values_require})
        jacobians = self.jacobians(**{key: requirements[key] for key in self.jacobians_require})
        return np.einsum(self._sum_str, self.weight, values), np.einsum(self._sum_str, self.weight, jacobians)

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            if np.allclose(self.position, other.position):
                return CostFunctionPoint(self, other)
            else:
                return CostFunctionCollection(self, other)
        else:
            return NotImplemented

    def __sub__(self, vector):
        return VectorCostFunction(algorithm=self, target_vector=vector, weight=self.weight, position=self.position)

    def __mul__(self, weight):
        weight = np.asarray(weight)
        if weight.dtype == object:
            return NotImplemented
        return CostFunction(self.algorithm, self.weight * weight, self.position)


class VectorBase(Algorithm):
    def __init__(self, algorithm, target_vector, **kwargs):
        if type(self) == VectorBase:
            raise AssertionError('`VectorBase` should never be directly instantiated!')
        self.values_require = algorithm.values_require.copy()
        self.jacobians_require = algorithm.jacobians_require.copy()
        for key, value in algorithm.values_require.items():
            self.jacobians_require[key] = max(value, self.jacobians_require.get(key, -1))
        super().__init__(algorithm=algorithm, **kwargs)
        target_vector = np.asarray(target_vector)
        self.target_vector = target_vector

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.allclose(self.target_vector, other.target_vector)
        )

    @property
    def name(self):
        return self.algorithm.name

    def values(self, **kwargs):
        values = self.algorithm.values(**kwargs)
        values -= self.target_vector.reshape([-1] + (values.ndim - 1) * [1])
        return np.real(values * np.conj(values))

    def jacobians(self, **kwargs):
        values = self.algorithm.values(**{key: kwargs[key] for key in self.algorithm.values_require})
        values -= self.target_vector.reshape([-1] + (values.ndim - 1) * [1])
        jacobians = self.algorithm.jacobians(**{key: kwargs[key] for key in self.algorithm.jacobians_require})
        return 2 * jacobians * values.reshape(values.shape[:self.ndim] + (1,) + values.shape[self.ndim:])

    # These properties are needed to not overwrite the requirements defined in the algorithm implementations.
    @property
    def values_require(self):
        return self._values_require

    @values_require.setter
    def values_require(self, val):
        self._values_require = val

    @property
    def jacobians_require(self):
        return self._jacobians_require

    @jacobians_require.setter
    def jacobians_require(self, val):
        self._jacobians_require = val

    def __sub__(self, vector):
        kwargs = {}
        if self._is_bound:
            kwargs['position'] = self.position
        if self._is_cost:
            kwargs['weight'] = self.weight
        return type(self)(self.algorithm, self.target_vector + vector, **kwargs)

    def __format__(self, format_spec):
        format_spec = format_spec.replace('%name', '||%name - %vector||^2').replace('%vector', str(self.target_vector))
        return super().__format__(format_spec)


class VectorAlgorithm(VectorBase, Algorithm):
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

    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorBoundAlgorithm(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorUnboundCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight)


class VectorBoundAlgorithm(VectorBase, BoundAlgorithm):
    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if VectorBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            if np.allclose(self.position, other.position):
                return BoundAlgorithmPoint(self, other)
            else:
                return AlgorithmCollection(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorBoundAlgorithm(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight, position=algorithm.position)


class VectorUnboundCostFunction(VectorBase, UnboundCostFunction):
    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if VectorBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            return UnboundCostFunctionPoint(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position, weight=algorithm.weight)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorUnboundCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight)


class VectorCostFunction(VectorBase, CostFunction):
    def __add__(self, other):
        if other == 0:
            return self
        other_type = type(other)
        if VectorBase in other_type.__bases__:
            other_type = other_type.__bases__[1]
        if other_type == type(self).__bases__[1]:
            if np.allclose(self.position, other.position):
                return CostFunctionPoint(self, other)
            else:
                return CostFunctionCollection(self, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        algorithm = self.algorithm @ position
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, position=algorithm.position, weight=algorithm.weight)

    def __mul__(self, weight):
        algorithm = self.algorithm * weight
        return VectorCostFunction(algorithm=algorithm, target_vector=self.target_vector, weight=algorithm.weight, position=algorithm.position)


class AlgorithmPoint(AlgorithmBase):
    _str_format_spec = '{:%cls%algorithms%position}'
    _is_bound = False
    _is_cost = False

    def __init__(self, *algorithms):
        self.algorithms = []
        self.requires = {}
        for algorithm in algorithms:
            self += algorithm

    def __eq__(self, other):
        return super().__eq__(other) and self.algorithms == other.algorithms

    @property
    def array(self):
        return self.algorithms[0].array

    def __call__(self, complex_transducer_amplitudes, position):
        # Prepare the requirements dict
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        # Call the function with the correct arguments
        return [algorithm.values(**{key: requirements[key] for key in algorithm.values_require}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return AlgorithmPoint(*self.algorithms, *other.algorithms)
        elif self._type == other._type:
            return AlgorithmPoint(*self.algorithms, other)
        else:
            return NotImplemented

    def __iadd__(self, other):
        add_element = False
        add_point = False
        if type(self) == type(other):
            add_point = True
        elif self._type == other._type:
            add_element = True
        old_requires = self.requires.copy()
        if add_element:
            for key, value in other.requires.items():
                self.requires[key] = max(value, self.requires.get(key, -1))
            self.algorithms.append(other)
        elif add_point:
            for algorithm in other.algorithms:
                self += algorithm
        else:
            return NotImplemented
        if self.requires != old_requires:
            # We have new requirements, if there are cached spatial structures they will
            # need to be recalculated at next call.
            try:
                del self._cached_spatial_structures
            except AttributeError:
                pass
        return self

    def __sub__(self, other):
        return type(self)(*[algorithm - other for algorithm in self.algorithms])

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


class BoundAlgorithmPoint(AlgorithmPoint):
    _is_bound = True
    _is_cost = False

    def __init__(self, *algorithms):
        self.position = algorithms[0].position
        super().__init__(*algorithms)

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        return [algorithm.values(**{key: requirements[key] for key in algorithm.values_require}) for algorithm in self.algorithms]

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == BoundAlgorithmPoint and np.allclose(self.position, other.position):
            return BoundAlgorithmPoint(*self.algorithms, *other.algorithms)
        elif isinstance(other, BoundAlgorithm) and np.allclose(self.position, other.position):
            return BoundAlgorithmPoint(*self.algorithms, other)
        else:
            return AlgorithmCollection(self, other)

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


class UnboundCostFunctionPoint(AlgorithmPoint):
    _is_bound = False
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes, position):
        spatial_structures = self._spatial_structures(position)
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.values(**{key: requirements[key] for key in algorithm.values_require}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.jacobians(**{key: requirements[key] for key in algorithm.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if type(self) == type(other):
            return UnboundCostFunctionPoint(*self.algorithms, *other.algorithms)
        elif self._type == other._type:
            return UnboundCostFunctionPoint(*self.algorithms, other)
        else:
            return NotImplemented

    def __matmul__(self, position):
        return CostFunctionPoint(*[algorithm @ position for algorithm in self.algorithms])


class CostFunctionPoint(UnboundCostFunctionPoint, BoundAlgorithmPoint):
    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        spatial_structures = self._spatial_structures()
        requirements = self._evaluate_requirements(complex_transducer_amplitudes, spatial_structures)
        value = 0
        jacobians = 0
        for algorithm in self.algorithms:
            value += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.values(**{key: requirements[key] for key in algorithm.values_require}))
            jacobians += np.einsum(algorithm._sum_str, algorithm.weight, algorithm.jacobians(**{key: requirements[key] for key in algorithm.jacobians_require}))
        return value, jacobians

    def __add__(self, other):
        if other == 0:
            return self
        if self._type != other._type:
            return NotImplemented
        if type(other) == CostFunctionPoint and np.allclose(self.position, other.position):
            return CostFunctionPoint(*self.algorithms, *other.algorithms)
        elif isinstance(other, CostFunction) and np.allclose(self.position, other.position):
            return CostFunctionPoint(*self.algorithms, other)
        else:
            return CostFunctionCollection(self, other)

    def __iadd__(self, other):
        try:
            if np.allclose(other.position, self.position):
                return super().__iadd__(other)
            else:
                return CostFunctionCollection(self, other)
        except AttributeError:
            return NotImplemented


class AlgorithmCollection(AlgorithmBase):
    _str_format_spec = '{:%cls%points}'
    _is_bound = True
    _is_cost = False

    def __init__(self, *algorithms):
        self.algorithms = []
        for algorithm in algorithms:
            self += algorithm

    def __eq__(self, other):
        return super().__eq__(other) and self.algorithms == other.algorithms

    def __call__(self, complex_transducer_amplitudes):
        values = []
        for point in self.algorithms:
            values.append(point(complex_transducer_amplitudes))
        return values

    def __add__(self, other):
        if other == 0:
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            return type(self)(*self.algorithms, other)

    def __iadd__(self, other):
        if type(other) == type(self):
            for algorithm in other.algorithms:
                self += algorithm
            return self
        elif self._type != other._type:
            return NotImplemented
        else:
            for idx, point in enumerate(self.algorithms):
                if np.allclose(point.position, other.position):
                    # Mutating `point` will not update the contents in the list!
                    self.algorithms[idx] += other
                    break
            else:
                self.algorithms.append(other)
            return self

    def __mul__(self, weight):
        return CostFunctionCollection(*[algorithm * weight for algorithm in self.algorithms])

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
                points_str += points_spec.format(algorithm).replace('%algorithms', '')
            format_spec = format_spec.replace('%points', points_str + ']')
        return super().__format__(format_spec)


class CostFunctionCollection(AlgorithmCollection, CostFunctionPoint):
    _is_bound = True
    _is_cost = True

    def __call__(self, complex_transducer_amplitudes):
        values = 0
        jacobians = 0
        for algorithm in self.algorithms:
            val, jac = algorithm(complex_transducer_amplitudes)
            values += val
            jacobians += jac
        return values, jacobians
