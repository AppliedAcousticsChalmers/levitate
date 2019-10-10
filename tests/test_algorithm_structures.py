import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

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

magnitude_squared_algorithm = algorithm - 0
magnitude_squared_bound_algorithm = bound_algorithm - 0
magnitude_squared_bound_algorithm_b = bound_algorithm_b - 0
magnitude_squared_unbound_cost_function = unbound_cost_function - 0
magnitude_squared_cost_function = cost_function - 0
magnitude_squared_cost_function_b = cost_function_b - 0

algorithm_point = algorithm + algorithm
bound_algorithm_point = bound_algorithm + bound_algorithm
bound_algorithm_point_b = bound_algorithm_b + bound_algorithm_b
unbound_cost_function_point = unbound_cost_function + unbound_cost_function
cost_function_point = cost_function + cost_function
cost_function_point_b = cost_function_b + cost_function_b

algorithm_collection = bound_algorithm + bound_algorithm_b
cost_function_collection = cost_function + cost_function_b

algorithms = [algorithm, magnitude_squared_algorithm, algorithm_point]
bound_algorithms = [bound_algorithm, bound_algorithm_b, magnitude_squared_bound_algorithm, magnitude_squared_bound_algorithm_b, bound_algorithm_point, bound_algorithm_point_b, algorithm_collection]
unbound_cost_functions = [unbound_cost_function, magnitude_squared_unbound_cost_function, unbound_cost_function_point]
cost_functions = [cost_function, cost_function_b, magnitude_squared_cost_function, magnitude_squared_cost_function_b, cost_function_point, cost_function_point_b, cost_function_collection]


def test_basics():
    # Make sure that all the above actually are what they should
    assert type(algorithm) == classes.Field
    assert type(bound_algorithm) == classes.FieldPoint
    assert type(unbound_cost_function) == classes.CostField
    assert type(cost_function) == classes.CostFieldPoint

    assert type(magnitude_squared_algorithm) == classes.SquaredField
    assert type(magnitude_squared_bound_algorithm) == classes.SquaredFieldPoint
    assert type(magnitude_squared_unbound_cost_function) == classes.SquaredCostField
    assert type(magnitude_squared_cost_function) == classes.SquaredCostFieldPoint

    assert type(algorithm_point) == classes.MultiField
    assert type(bound_algorithm_point) == classes.MultiFieldPoint
    assert type(unbound_cost_function_point) == classes.MultiCostField
    assert type(cost_function_point) == classes.MultiCostFieldPoint

    assert type(algorithm_collection) == classes.MultiFieldMultiPoint
    assert type(cost_function_collection) == classes.MultiCostFieldMultiPoint


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
    addable(algorithm, algorithm, classes.MultiField)
    addable(algorithm, magnitude_squared_algorithm, classes.MultiField)
    addable(algorithm, algorithm_point, classes.MultiField)
    not_addable(algorithm, bound_algorithms)
    not_addable(algorithm, unbound_cost_functions)
    not_addable(algorithm, cost_functions)

    # Test of other morphing
    mult(algorithm, classes.CostField)
    bind(algorithm, classes.FieldPoint)
    sub(algorithm, classes.SquaredField)

    # Test misc
    str(algorithm)


def test_bound_algorithm():
    # Test for addition
    addable(bound_algorithm, bound_algorithm, classes.MultiFieldPoint)
    addable(bound_algorithm, bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm, magnitude_squared_bound_algorithm, classes.MultiFieldPoint)
    addable(bound_algorithm, magnitude_squared_bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm, bound_algorithm_point, classes.MultiFieldPoint)
    addable(bound_algorithm, bound_algorithm_point_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm, algorithm_collection, classes.MultiFieldMultiPoint)
    not_addable(bound_algorithm, algorithms)
    not_addable(bound_algorithm, unbound_cost_functions)
    not_addable(bound_algorithm, cost_functions)

    # Test of other morphing
    mult(bound_algorithm, classes.CostFieldPoint)
    bind(bound_algorithm, classes.FieldPoint)
    sub(bound_algorithm, classes.SquaredFieldPoint)

    # Test misc
    str(bound_algorithm)


def test_unbound_cost_function():
    # Test for addition
    addable(unbound_cost_function, unbound_cost_function, classes.MultiCostField)
    addable(unbound_cost_function, magnitude_squared_unbound_cost_function, classes.MultiCostField)
    addable(unbound_cost_function, unbound_cost_function_point, classes.MultiCostField)
    not_addable(unbound_cost_function, algorithms)
    not_addable(unbound_cost_function, bound_algorithms)
    not_addable(unbound_cost_function, cost_functions)

    # Test of other morphing
    mult(unbound_cost_function, classes.CostField)
    bind(unbound_cost_function, classes.CostFieldPoint)
    sub(unbound_cost_function, classes.SquaredCostField)

    # Test misc
    str(unbound_cost_function)


def test_cost_function():
    # Test for addition
    addable(cost_function, cost_function, classes.MultiCostFieldPoint)
    addable(cost_function, cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function, magnitude_squared_cost_function, classes.MultiCostFieldPoint)
    addable(cost_function, magnitude_squared_cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function, cost_function_point, classes.MultiCostFieldPoint)
    addable(cost_function, cost_function_point_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function, cost_function_collection, classes.MultiCostFieldMultiPoint)
    not_addable(cost_function, algorithms)
    not_addable(cost_function, unbound_cost_functions)
    not_addable(cost_function, bound_algorithms)

    # Test of other morphing
    mult(cost_function, classes.CostFieldPoint)
    bind(cost_function, classes.CostFieldPoint)
    sub(cost_function, classes.SquaredCostFieldPoint)

    # Test misc
    str(cost_function)


def test_magnitude_squared_algorithm():
    # Test for addition
    addable(magnitude_squared_algorithm, algorithm, classes.MultiField)
    addable(magnitude_squared_algorithm, magnitude_squared_algorithm, classes.MultiField)
    addable(magnitude_squared_algorithm, algorithm_point, classes.MultiField)
    not_addable(magnitude_squared_algorithm, bound_algorithms)
    not_addable(magnitude_squared_algorithm, unbound_cost_functions)
    not_addable(magnitude_squared_algorithm, cost_functions)

    # Test of other morphing
    mult(magnitude_squared_algorithm, classes.SquaredCostField)
    bind(magnitude_squared_algorithm, classes.SquaredFieldPoint)
    sub(magnitude_squared_algorithm, classes.SquaredField)

    # Test misc
    str(magnitude_squared_algorithm)


def test_magnitude_squared_bound_algorithm():
    # Test for addition
    addable(magnitude_squared_bound_algorithm, bound_algorithm, classes.MultiFieldPoint)
    addable(magnitude_squared_bound_algorithm, bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(magnitude_squared_bound_algorithm, magnitude_squared_bound_algorithm, classes.MultiFieldPoint)
    addable(magnitude_squared_bound_algorithm, magnitude_squared_bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(magnitude_squared_bound_algorithm, bound_algorithm_point, classes.MultiFieldPoint)
    addable(magnitude_squared_bound_algorithm, bound_algorithm_point_b, classes.MultiFieldMultiPoint)
    addable(magnitude_squared_bound_algorithm, algorithm_collection, classes.MultiFieldMultiPoint)
    not_addable(magnitude_squared_bound_algorithm, algorithms)
    not_addable(magnitude_squared_bound_algorithm, unbound_cost_functions)
    not_addable(magnitude_squared_bound_algorithm, cost_functions)

    # Test of other morphing
    mult(magnitude_squared_bound_algorithm, classes.SquaredCostFieldPoint)
    bind(magnitude_squared_bound_algorithm, classes.SquaredFieldPoint)
    sub(magnitude_squared_bound_algorithm, classes.SquaredFieldPoint)

    # Test misc
    str(magnitude_squared_bound_algorithm)


def test_magnitude_squared_unbound_cost_function():
    # Test for addition
    addable(magnitude_squared_unbound_cost_function, unbound_cost_function, classes.MultiCostField)
    addable(magnitude_squared_unbound_cost_function, magnitude_squared_unbound_cost_function, classes.MultiCostField)
    addable(magnitude_squared_unbound_cost_function, unbound_cost_function_point, classes.MultiCostField)
    not_addable(magnitude_squared_unbound_cost_function, algorithms)
    not_addable(magnitude_squared_unbound_cost_function, bound_algorithms)
    not_addable(magnitude_squared_unbound_cost_function, cost_functions)

    # Test of other morphing
    mult(magnitude_squared_unbound_cost_function, classes.SquaredCostField)
    bind(magnitude_squared_unbound_cost_function, classes.SquaredCostFieldPoint)
    sub(magnitude_squared_unbound_cost_function, classes.SquaredCostField)

    # Test misc
    str(magnitude_squared_unbound_cost_function)


def test_magnitude_squared_cost_function():
    # Test for addition
    addable(magnitude_squared_cost_function, cost_function, classes.MultiCostFieldPoint)
    addable(magnitude_squared_cost_function, cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(magnitude_squared_cost_function, magnitude_squared_cost_function, classes.MultiCostFieldPoint)
    addable(magnitude_squared_cost_function, magnitude_squared_cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(magnitude_squared_cost_function, cost_function_point, classes.MultiCostFieldPoint)
    addable(magnitude_squared_cost_function, cost_function_point_b, classes.MultiCostFieldMultiPoint)
    addable(magnitude_squared_cost_function, cost_function_collection, classes.MultiCostFieldMultiPoint)
    not_addable(magnitude_squared_cost_function, algorithms)
    not_addable(magnitude_squared_cost_function, unbound_cost_functions)
    not_addable(magnitude_squared_cost_function, bound_algorithms)

    # Test of other morphing
    mult(magnitude_squared_cost_function, classes.SquaredCostFieldPoint)
    bind(magnitude_squared_cost_function, classes.SquaredCostFieldPoint)
    sub(magnitude_squared_cost_function, classes.SquaredCostFieldPoint)

    # Test misc
    str(magnitude_squared_cost_function)


def test_algorithm_point():
    # Test for addition
    addable(algorithm_point, algorithm, classes.MultiField)
    addable(algorithm_point, magnitude_squared_algorithm, classes.MultiField)
    addable(algorithm_point, algorithm_point, classes.MultiField)
    not_addable(algorithm_point, bound_algorithms)
    not_addable(algorithm_point, unbound_cost_functions)
    not_addable(algorithm_point, cost_functions)

    # Test of other morphing
    mult(algorithm_point, classes.MultiCostField)
    bind(algorithm_point, classes.MultiFieldPoint)
    sub(algorithm_point, classes.MultiField)

    # Test misc
    str(algorithm_point)


def test_bound_algorithm_point():
    # Test for addition
    addable(bound_algorithm_point, bound_algorithm, classes.MultiFieldPoint)
    addable(bound_algorithm_point, bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm_point, magnitude_squared_bound_algorithm, classes.MultiFieldPoint)
    addable(bound_algorithm_point, magnitude_squared_bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm_point, bound_algorithm_point, classes.MultiFieldPoint)
    addable(bound_algorithm_point, bound_algorithm_point_b, classes.MultiFieldMultiPoint)
    addable(bound_algorithm_point, algorithm_collection, classes.MultiFieldMultiPoint)
    not_addable(bound_algorithm_point, algorithms)
    not_addable(bound_algorithm_point, unbound_cost_functions)
    not_addable(bound_algorithm_point, cost_functions)

    # Test of other morphing
    mult(bound_algorithm_point, classes.MultiCostFieldPoint)
    bind(bound_algorithm_point, classes.MultiFieldPoint)
    sub(bound_algorithm_point, classes.MultiFieldPoint)

    # Test misc
    str(bound_algorithm_point)


def test_unbound_cost_function_point():
    # Test for addition
    addable(unbound_cost_function_point, unbound_cost_function, classes.MultiCostField)
    addable(unbound_cost_function_point, magnitude_squared_unbound_cost_function, classes.MultiCostField)
    addable(unbound_cost_function_point, unbound_cost_function_point, classes.MultiCostField)
    not_addable(unbound_cost_function_point, algorithms)
    not_addable(unbound_cost_function_point, bound_algorithms)
    not_addable(unbound_cost_function_point, cost_functions)

    # Test of other morphing
    mult(unbound_cost_function_point, classes.MultiCostField)
    bind(unbound_cost_function_point, classes.MultiCostFieldPoint)
    sub(unbound_cost_function_point, classes.MultiCostField)

    # Test misc
    str(unbound_cost_function_point)


def test_cost_function_point():
    # Test for addition
    addable(cost_function_point, cost_function, classes.MultiCostFieldPoint)
    addable(cost_function_point, cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_point, magnitude_squared_cost_function, classes.MultiCostFieldPoint)
    addable(cost_function_point, magnitude_squared_cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_point, cost_function_point, classes.MultiCostFieldPoint)
    addable(cost_function_point, cost_function_point_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_point, cost_function_collection, classes.MultiCostFieldMultiPoint)
    not_addable(cost_function_point, algorithms)
    not_addable(cost_function_point, unbound_cost_functions)
    not_addable(cost_function_point, bound_algorithms)

    # Test of other morphing
    mult(cost_function_point, classes.MultiCostFieldPoint)
    bind(cost_function_point, classes.MultiCostFieldPoint)
    sub(cost_function_point, classes.MultiCostFieldPoint)

    # Test misc
    str(cost_function_point)


def test_algorithm_collection():
    # Test for addition
    addable(algorithm_collection, bound_algorithm, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, magnitude_squared_bound_algorithm, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, magnitude_squared_bound_algorithm_b, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, bound_algorithm_point, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, bound_algorithm_point_b, classes.MultiFieldMultiPoint)
    addable(algorithm_collection, algorithm_collection, classes.MultiFieldMultiPoint)
    not_addable(algorithm_collection, algorithms)
    not_addable(algorithm_collection, unbound_cost_functions)
    not_addable(algorithm_collection, cost_functions)

    mult(algorithm_collection, classes.MultiCostFieldMultiPoint)

    # Test misc
    str(algorithm_collection)


def test_cost_function_collection():
    # Test for addition
    addable(cost_function_collection, cost_function, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, magnitude_squared_cost_function, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, magnitude_squared_cost_function_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, cost_function_point, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, cost_function_point_b, classes.MultiCostFieldMultiPoint)
    addable(cost_function_collection, cost_function_collection, classes.MultiCostFieldMultiPoint)
    not_addable(cost_function_collection, algorithms)
    not_addable(cost_function_collection, unbound_cost_functions)
    not_addable(cost_function_collection, bound_algorithms)

    mult(cost_function_collection, classes.MultiCostFieldMultiPoint)

    # Test misc
    str(cost_function_collection)
