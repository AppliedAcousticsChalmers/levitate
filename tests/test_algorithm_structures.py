import pytest
import numpy as np
import levitate

classes = levitate._algorithm

pos = np.array([0.1, 0.2, 0.3])
pos_b = np.array([-0.15, 1.27, 0.001])

array = levitate.arrays.RectangularArray(shape=(4, 5))

algorithm = levitate.algorithms.GorkovPotential(array)
bound_algorithm = algorithm @ pos
bound_algorithm_b = algorithm @ pos_b
unbound_cost_function = algorithm * 1
cost_function = algorithm @ pos * 1
cost_function_b = algorithm @ pos_b * 1

vector_algorithm = algorithm - 0
vector_bound_algorithm = bound_algorithm - 0
vector_bound_algorithm_b = bound_algorithm_b - 0
vector_unbound_cost_function = unbound_cost_function - 0
vector_cost_function = cost_function - 0
vector_cost_function_b = cost_function_b - 0

algorithm_point = algorithm + algorithm
bound_algorithm_point = bound_algorithm + bound_algorithm
bound_algorithm_point_b = bound_algorithm_b + bound_algorithm_b
unbound_cost_function_point = unbound_cost_function + unbound_cost_function
cost_function_point = cost_function + cost_function
cost_function_point_b = cost_function_b + cost_function_b

algorithm_collection = bound_algorithm + bound_algorithm_b
cost_function_collection = cost_function + cost_function_b

algorithms = [algorithm, vector_algorithm, algorithm_point]
bound_algorithms = [bound_algorithm, bound_algorithm_b, vector_bound_algorithm, vector_bound_algorithm_b, bound_algorithm_point, bound_algorithm_point_b, algorithm_collection]
unbound_cost_functions = [unbound_cost_function, vector_unbound_cost_function, unbound_cost_function_point]
cost_functions = [cost_function, cost_function_b, vector_cost_function, vector_cost_function_b, cost_function_point, cost_function_point_b, cost_function_collection]


def test_basics():
    # Make sure that all the above actually are what they should
    assert type(algorithm) == classes.Algorithm
    assert type(bound_algorithm) == classes.BoundAlgorithm
    assert type(unbound_cost_function) == classes.UnboundCostFunction
    assert type(cost_function) == classes.CostFunction

    assert type(vector_algorithm) == classes.VectorAlgorithm
    assert type(vector_bound_algorithm) == classes.VectorBoundAlgorithm
    assert type(vector_unbound_cost_function) == classes.VectorUnboundCostFunction
    assert type(vector_cost_function) == classes.VectorCostFunction

    assert type(algorithm_point) == classes.AlgorithmPoint
    assert type(bound_algorithm_point) == classes.BoundAlgorithmPoint
    assert type(unbound_cost_function_point) == classes.UnboundCostFunctionPoint
    assert type(cost_function_point) == classes.CostFunctionPoint

    assert type(algorithm_collection) == classes.AlgorithmCollection
    assert type(cost_function_collection) == classes.CostFunctionCollection


def addable(a, b, result_cls):
    try:
        assert type(a + b) == result_cls
    except TypeError:
        raise TypeError('{.__name__} and {.__name__} addition failed'.format(type(a), type(b)))
    except AssertionError:
        raise TypeError('Addition of {.__name__} and {.__name__} returned {.__name__}, not {.__name__}'.format(type(a), type(b), type(a + b), result_cls))


def not_addable(a, b):
    if hasattr(b, '__len__'):
        for item in b:
            not_addable(a, item)
    else:
        try:
            a + b
        except TypeError:
            pass
        else:
            raise TypeError('Addition of {.__name__} and {.__name__} returned {.__name__}, should fail'.format(type(a), type(b), type(a + b)))


def mult(obj, result_cls):
    try:
        assert type(obj * 1) == result_cls
    except TypeError:
        raise TypeError('Mult of {.__name__} failed'.format(type(obj)))
    except AssertionError:
        raise TypeError('Mult of {.__name__} returned {.__name__}, not {.__name__}'.format(type(obj), type(obj * 1), result_cls))

    try:
        obj * obj
    except TypeError as e:
        if not str(e).startswith('unsupported operand type(s) for *:'):
            raise e
    else:
        raise TypeError('{.__name__} can multiply with itself'.format(type(obj)))


def bind(obj, result_cls):
    try:
        assert type(obj @ pos) == result_cls
    except TypeError:
        raise TypeError('Bind of {.__name__} failed'.format(type(obj)))
    except AssertionError:
        raise TypeError('Bind of {.__name__} returned {.__name__}, not {.__name__}'.format(type(obj), type(obj @ pos), result_cls))
    with pytest.raises((TypeError, ValueError)):
        obj @ np.array(0)
    with pytest.raises((TypeError, ValueError)):
        obj @ np.array([0])
    with pytest.raises((TypeError, ValueError)):
        obj @ np.array([0, 1])
    with pytest.raises((TypeError, ValueError)):
        obj @ np.array([1, 2, 3, 4])
    with pytest.raises((TypeError, ValueError)):
        obj @ np.array([[0, 0, 0]])


def sub(obj, result_cls):
    try:
        assert type(obj - 0) == result_cls
    except TypeError:
        raise TypeError('Sub of {.__name__} failed'.format(type(obj)))
    except AssertionError:
        raise TypeError('Sub of {.__name__} returned {.__name__}, not {.__name__}'.format(type(obj), type(obj - 0), result_cls))


def test_algorithm():
    # Test for addition
    addable(algorithm, algorithm, classes.AlgorithmPoint)
    addable(algorithm, vector_algorithm, classes.AlgorithmPoint)
    addable(algorithm, algorithm_point, classes.AlgorithmPoint)
    not_addable(algorithm, bound_algorithms)
    not_addable(algorithm, unbound_cost_functions)
    not_addable(algorithm, cost_functions)

    # Test of other morphing
    mult(algorithm, classes.UnboundCostFunction)
    bind(algorithm, classes.BoundAlgorithm)
    sub(algorithm, classes.VectorAlgorithm)

    # Test misc
    str(algorithm)


def test_bound_algorithm():
    # Test for addition
    addable(bound_algorithm, bound_algorithm, classes.BoundAlgorithmPoint)
    addable(bound_algorithm, bound_algorithm_b, classes.AlgorithmCollection)
    addable(bound_algorithm, vector_bound_algorithm, classes.BoundAlgorithmPoint)
    addable(bound_algorithm, vector_bound_algorithm_b, classes.AlgorithmCollection)
    addable(bound_algorithm, bound_algorithm_point, classes.BoundAlgorithmPoint)
    addable(bound_algorithm, bound_algorithm_point_b, classes.AlgorithmCollection)
    addable(bound_algorithm, algorithm_collection, classes.AlgorithmCollection)
    not_addable(bound_algorithm, algorithms)
    not_addable(bound_algorithm, unbound_cost_functions)
    not_addable(bound_algorithm, cost_functions)

    # Test of other morphing
    mult(bound_algorithm, classes.CostFunction)
    bind(bound_algorithm, classes.BoundAlgorithm)
    sub(bound_algorithm, classes.VectorBoundAlgorithm)

    # Test misc
    str(bound_algorithm)


def test_unbound_cost_function():
    # Test for addition
    addable(unbound_cost_function, unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(unbound_cost_function, vector_unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(unbound_cost_function, unbound_cost_function_point, classes.UnboundCostFunctionPoint)
    not_addable(unbound_cost_function, algorithms)
    not_addable(unbound_cost_function, bound_algorithms)
    not_addable(unbound_cost_function, cost_functions)

    # Test of other morphing
    mult(unbound_cost_function, classes.UnboundCostFunction)
    bind(unbound_cost_function, classes.CostFunction)
    sub(unbound_cost_function, classes.VectorUnboundCostFunction)

    # Test misc
    str(unbound_cost_function)


def test_cost_function():
    # Test for addition
    addable(cost_function, cost_function, classes.CostFunctionPoint)
    addable(cost_function, cost_function_b, classes.CostFunctionCollection)
    addable(cost_function, vector_cost_function, classes.CostFunctionPoint)
    addable(cost_function, vector_cost_function_b, classes.CostFunctionCollection)
    addable(cost_function, cost_function_point, classes.CostFunctionPoint)
    addable(cost_function, cost_function_point_b, classes.CostFunctionCollection)
    addable(cost_function, cost_function_collection, classes.CostFunctionCollection)
    not_addable(cost_function, algorithms)
    not_addable(cost_function, unbound_cost_functions)
    not_addable(cost_function, bound_algorithms)

    # Test of other morphing
    mult(cost_function, classes.CostFunction)
    bind(cost_function, classes.CostFunction)
    sub(cost_function, classes.VectorCostFunction)

    # Test misc
    str(cost_function)


def test_vector_algorithm():
    # Test for addition
    addable(vector_algorithm, algorithm, classes.AlgorithmPoint)
    addable(vector_algorithm, vector_algorithm, classes.AlgorithmPoint)
    addable(vector_algorithm, algorithm_point, classes.AlgorithmPoint)
    not_addable(vector_algorithm, bound_algorithms)
    not_addable(vector_algorithm, unbound_cost_functions)
    not_addable(vector_algorithm, cost_functions)

    # Test of other morphing
    mult(vector_algorithm, classes.VectorUnboundCostFunction)
    bind(vector_algorithm, classes.VectorBoundAlgorithm)
    sub(vector_algorithm, classes.VectorAlgorithm)

    # Test misc
    str(vector_algorithm)


def test_vector_bound_algorithm():
    # Test for addition
    addable(vector_bound_algorithm, bound_algorithm, classes.BoundAlgorithmPoint)
    addable(vector_bound_algorithm, bound_algorithm_b, classes.AlgorithmCollection)
    addable(vector_bound_algorithm, vector_bound_algorithm, classes.BoundAlgorithmPoint)
    addable(vector_bound_algorithm, vector_bound_algorithm_b, classes.AlgorithmCollection)
    addable(vector_bound_algorithm, bound_algorithm_point, classes.BoundAlgorithmPoint)
    addable(vector_bound_algorithm, bound_algorithm_point_b, classes.AlgorithmCollection)
    addable(vector_bound_algorithm, algorithm_collection, classes.AlgorithmCollection)
    not_addable(vector_bound_algorithm, algorithms)
    not_addable(vector_bound_algorithm, unbound_cost_functions)
    not_addable(vector_bound_algorithm, cost_functions)

    # Test of other morphing
    mult(vector_bound_algorithm, classes.VectorCostFunction)
    bind(vector_bound_algorithm, classes.VectorBoundAlgorithm)
    sub(vector_bound_algorithm, classes.VectorBoundAlgorithm)

    # Test misc
    str(vector_bound_algorithm)


def test_vector_unbound_cost_function():
    # Test for addition
    addable(vector_unbound_cost_function, unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(vector_unbound_cost_function, vector_unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(vector_unbound_cost_function, unbound_cost_function_point, classes.UnboundCostFunctionPoint)
    not_addable(vector_unbound_cost_function, algorithms)
    not_addable(vector_unbound_cost_function, bound_algorithms)
    not_addable(vector_unbound_cost_function, cost_functions)

    # Test of other morphing
    mult(vector_unbound_cost_function, classes.VectorUnboundCostFunction)
    bind(vector_unbound_cost_function, classes.VectorCostFunction)
    sub(vector_unbound_cost_function, classes.VectorUnboundCostFunction)

    # Test misc
    str(vector_unbound_cost_function)


def test_vector_cost_function():
    # Test for addition
    addable(vector_cost_function, cost_function, classes.CostFunctionPoint)
    addable(vector_cost_function, cost_function_b, classes.CostFunctionCollection)
    addable(vector_cost_function, vector_cost_function, classes.CostFunctionPoint)
    addable(vector_cost_function, vector_cost_function_b, classes.CostFunctionCollection)
    addable(vector_cost_function, cost_function_point, classes.CostFunctionPoint)
    addable(vector_cost_function, cost_function_point_b, classes.CostFunctionCollection)
    addable(vector_cost_function, cost_function_collection, classes.CostFunctionCollection)
    not_addable(vector_cost_function, algorithms)
    not_addable(vector_cost_function, unbound_cost_functions)
    not_addable(vector_cost_function, bound_algorithms)

    # Test of other morphing
    mult(vector_cost_function, classes.VectorCostFunction)
    bind(vector_cost_function, classes.VectorCostFunction)
    sub(vector_cost_function, classes.VectorCostFunction)

    # Test misc
    str(vector_cost_function)


def test_algorithm_point():
    # Test for addition
    addable(algorithm_point, algorithm, classes.AlgorithmPoint)
    addable(algorithm_point, vector_algorithm, classes.AlgorithmPoint)
    addable(algorithm_point, algorithm_point, classes.AlgorithmPoint)
    not_addable(algorithm_point, bound_algorithms)
    not_addable(algorithm_point, unbound_cost_functions)
    not_addable(algorithm_point, cost_functions)

    # Test of other morphing
    mult(algorithm_point, classes.UnboundCostFunctionPoint)
    bind(algorithm_point, classes.BoundAlgorithmPoint)
    sub(algorithm_point, classes.AlgorithmPoint)

    # Test misc
    str(algorithm_point)


def test_bound_algorithm_point():
    # Test for addition
    addable(bound_algorithm_point, bound_algorithm, classes.BoundAlgorithmPoint)
    addable(bound_algorithm_point, bound_algorithm_b, classes.AlgorithmCollection)
    addable(bound_algorithm_point, vector_bound_algorithm, classes.BoundAlgorithmPoint)
    addable(bound_algorithm_point, vector_bound_algorithm_b, classes.AlgorithmCollection)
    addable(bound_algorithm_point, bound_algorithm_point, classes.BoundAlgorithmPoint)
    addable(bound_algorithm_point, bound_algorithm_point_b, classes.AlgorithmCollection)
    addable(bound_algorithm_point, algorithm_collection, classes.AlgorithmCollection)
    not_addable(bound_algorithm_point, algorithms)
    not_addable(bound_algorithm_point, unbound_cost_functions)
    not_addable(bound_algorithm_point, cost_functions)

    # Test of other morphing
    mult(bound_algorithm_point, classes.CostFunctionPoint)
    bind(bound_algorithm_point, classes.BoundAlgorithmPoint)
    sub(bound_algorithm_point, classes.BoundAlgorithmPoint)

    # Test misc
    str(bound_algorithm_point)


def test_unbound_cost_function_point():
    # Test for addition
    addable(unbound_cost_function_point, unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(unbound_cost_function_point, vector_unbound_cost_function, classes.UnboundCostFunctionPoint)
    addable(unbound_cost_function_point, unbound_cost_function_point, classes.UnboundCostFunctionPoint)
    not_addable(unbound_cost_function_point, algorithms)
    not_addable(unbound_cost_function_point, bound_algorithms)
    not_addable(unbound_cost_function_point, cost_functions)

    # Test of other morphing
    mult(unbound_cost_function_point, classes.UnboundCostFunctionPoint)
    bind(unbound_cost_function_point, classes.CostFunctionPoint)
    sub(unbound_cost_function_point, classes.UnboundCostFunctionPoint)

    # Test misc
    str(unbound_cost_function_point)


def test_cost_function_point():
    # Test for addition
    addable(cost_function_point, cost_function, classes.CostFunctionPoint)
    addable(cost_function_point, cost_function_b, classes.CostFunctionCollection)
    addable(cost_function_point, vector_cost_function, classes.CostFunctionPoint)
    addable(cost_function_point, vector_cost_function_b, classes.CostFunctionCollection)
    addable(cost_function_point, cost_function_point, classes.CostFunctionPoint)
    addable(cost_function_point, cost_function_point_b, classes.CostFunctionCollection)
    addable(cost_function_point, cost_function_collection, classes.CostFunctionCollection)
    not_addable(cost_function_point, algorithms)
    not_addable(cost_function_point, unbound_cost_functions)
    not_addable(cost_function_point, bound_algorithms)

    # Test of other morphing
    mult(cost_function_point, classes.CostFunctionPoint)
    bind(cost_function_point, classes.CostFunctionPoint)
    sub(cost_function_point, classes.CostFunctionPoint)

    # Test misc
    str(cost_function_point)


def test_algorithm_collection():
    # Test for addition
    addable(algorithm_collection, bound_algorithm, classes.AlgorithmCollection)
    addable(algorithm_collection, bound_algorithm_b, classes.AlgorithmCollection)
    addable(algorithm_collection, vector_bound_algorithm, classes.AlgorithmCollection)
    addable(algorithm_collection, vector_bound_algorithm_b, classes.AlgorithmCollection)
    addable(algorithm_collection, bound_algorithm_point, classes.AlgorithmCollection)
    addable(algorithm_collection, bound_algorithm_point_b, classes.AlgorithmCollection)
    addable(algorithm_collection, algorithm_collection, classes.AlgorithmCollection)
    not_addable(algorithm_collection, algorithms)
    not_addable(algorithm_collection, unbound_cost_functions)
    not_addable(algorithm_collection, cost_functions)

    mult(algorithm_collection, classes.CostFunctionCollection)

    # Test misc
    str(algorithm_collection)


def test_cost_function_collection():
    # Test for addition
    addable(cost_function_collection, cost_function, classes.CostFunctionCollection)
    addable(cost_function_collection, cost_function_b, classes.CostFunctionCollection)
    addable(cost_function_collection, vector_cost_function, classes.CostFunctionCollection)
    addable(cost_function_collection, vector_cost_function_b, classes.CostFunctionCollection)
    addable(cost_function_collection, cost_function_point, classes.CostFunctionCollection)
    addable(cost_function_collection, cost_function_point_b, classes.CostFunctionCollection)
    addable(cost_function_collection, cost_function_collection, classes.CostFunctionCollection)
    not_addable(cost_function_collection, algorithms)
    not_addable(cost_function_collection, unbound_cost_functions)
    not_addable(cost_function_collection, bound_algorithms)

    mult(cost_function_collection, classes.CostFunctionCollection)

    # Test misc
    str(cost_function_collection)
