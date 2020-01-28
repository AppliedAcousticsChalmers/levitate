import pytest
import numpy as np
import levitate

# Tests created with these air properties
from levitate.materials import air
air.c = 343
air.rho = 1.2

classes = levitate._field_wrappers

pos = np.array([0.1, 0.2, 0.3])
pos_b = np.array([-0.15, 1.27, 0.001])

array = levitate.arrays.RectangularArray(shape=(4, 5))

field = levitate.fields.GorkovPotential(array)
field_point_a = field @ pos
field_point_b = field @ pos_b
cost_field = field * 1
cost_field_a = field @ pos * 1
cost_field_b = field @ pos_b * 1

squared_field = field - 0
squared_field_point_a = field_point_a - 0
squared_field_point_b = field_point_b - 0
squared_cost_field = cost_field - 0
squared_cost_field_point_a = cost_field_a - 0
squared_cost_field_point_b = cost_field_b - 0

multi_field = field + field
multi_field_point_a = field_point_a + field_point_a
multi_field_point_b = field_point_b + field_point_b
multi_cost_field = cost_field + cost_field
multi_cost_field_point_a = cost_field_a + cost_field_a
multi_cost_field_point_b = cost_field_b + cost_field_b

multi_field_multi_point = field_point_a + field_point_b
multi_cost_field_multi_point = cost_field_a + cost_field_b

fields = [field, squared_field, multi_field]
field_points = [field_point_a, field_point_b, squared_field_point_a, squared_field_point_b, multi_field_point_a, multi_field_point_b, multi_field_multi_point]
cost_fields = [cost_field, squared_cost_field, multi_cost_field]
cost_field_points = [cost_field_a, cost_field_b, squared_cost_field_point_a, squared_cost_field_point_b, multi_cost_field_point_a, multi_cost_field_point_b, multi_cost_field_multi_point]


def test_basics():
    # Make sure that all the above actually are what they should
    assert type(field) == classes.Field
    assert type(field_point_a) == classes.FieldPoint
    assert type(cost_field) == classes.CostField
    assert type(cost_field_a) == classes.CostFieldPoint

    assert type(squared_field) == classes.SquaredField
    assert type(squared_field_point_a) == classes.SquaredFieldPoint
    assert type(squared_cost_field) == classes.SquaredCostField
    assert type(squared_cost_field_point_a) == classes.SquaredCostFieldPoint

    assert type(multi_field) == classes.MultiField
    assert type(multi_field_point_a) == classes.MultiFieldPoint
    assert type(multi_cost_field) == classes.MultiCostField
    assert type(multi_cost_field_point_a) == classes.MultiCostFieldPoint

    assert type(multi_field_multi_point) == classes.MultiFieldMultiPoint
    assert type(multi_cost_field_multi_point) == classes.MultiCostFieldMultiPoint


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


def test_field():
    # Test for addition
    addable(field, field, classes.MultiField)
    addable(field, squared_field, classes.MultiField)
    addable(field, multi_field, classes.MultiField)
    not_addable(field, field_points)
    not_addable(field, cost_fields)
    not_addable(field, cost_field_points)

    # Test of other morphing
    mult(field, classes.CostField)
    bind(field, classes.FieldPoint)
    sub(field, classes.SquaredField)

    # Test misc
    str(field)


def test_field_point():
    # Test for addition
    addable(field_point_a, field_point_a, classes.MultiFieldPoint)
    addable(field_point_a, field_point_b, classes.MultiFieldMultiPoint)
    addable(field_point_a, squared_field_point_a, classes.MultiFieldPoint)
    addable(field_point_a, squared_field_point_b, classes.MultiFieldMultiPoint)
    addable(field_point_a, multi_field_point_a, classes.MultiFieldPoint)
    addable(field_point_a, multi_field_point_b, classes.MultiFieldMultiPoint)
    addable(field_point_a, multi_field_multi_point, classes.MultiFieldMultiPoint)
    not_addable(field_point_a, fields)
    not_addable(field_point_a, cost_fields)
    not_addable(field_point_a, cost_field_points)

    # Test of other morphing
    mult(field_point_a, classes.CostFieldPoint)
    bind(field_point_a, classes.FieldPoint)
    sub(field_point_a, classes.SquaredFieldPoint)

    # Test misc
    str(field_point_a)


def test_cost_field():
    # Test for addition
    addable(cost_field, cost_field, classes.MultiCostField)
    addable(cost_field, squared_cost_field, classes.MultiCostField)
    addable(cost_field, multi_cost_field, classes.MultiCostField)
    not_addable(cost_field, fields)
    not_addable(cost_field, field_points)
    not_addable(cost_field, cost_field_points)

    # Test of other morphing
    mult(cost_field, classes.CostField)
    bind(cost_field, classes.CostFieldPoint)
    sub(cost_field, classes.SquaredCostField)

    # Test misc
    str(cost_field)


def test_cost_field_point():
    # Test for addition
    addable(cost_field_a, cost_field_a, classes.MultiCostFieldPoint)
    addable(cost_field_a, cost_field_b, classes.MultiCostFieldMultiPoint)
    addable(cost_field_a, squared_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(cost_field_a, squared_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(cost_field_a, multi_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(cost_field_a, multi_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(cost_field_a, multi_cost_field_multi_point, classes.MultiCostFieldMultiPoint)
    not_addable(cost_field_a, fields)
    not_addable(cost_field_a, cost_fields)
    not_addable(cost_field_a, field_points)

    # Test of other morphing
    mult(cost_field_a, classes.CostFieldPoint)
    bind(cost_field_a, classes.CostFieldPoint)
    sub(cost_field_a, classes.SquaredCostFieldPoint)

    # Test misc
    str(cost_field_a)


def test_squared_field():
    # Test for addition
    addable(squared_field, field, classes.MultiField)
    addable(squared_field, squared_field, classes.MultiField)
    addable(squared_field, multi_field, classes.MultiField)
    not_addable(squared_field, field_points)
    not_addable(squared_field, cost_fields)
    not_addable(squared_field, cost_field_points)

    # Test of other morphing
    mult(squared_field, classes.SquaredCostField)
    bind(squared_field, classes.SquaredFieldPoint)
    sub(squared_field, classes.SquaredField)

    # Test misc
    str(squared_field)


def test_squared_field_point():
    # Test for addition
    addable(squared_field_point_a, field_point_a, classes.MultiFieldPoint)
    addable(squared_field_point_a, field_point_b, classes.MultiFieldMultiPoint)
    addable(squared_field_point_a, squared_field_point_a, classes.MultiFieldPoint)
    addable(squared_field_point_a, squared_field_point_b, classes.MultiFieldMultiPoint)
    addable(squared_field_point_a, multi_field_point_a, classes.MultiFieldPoint)
    addable(squared_field_point_a, multi_field_point_b, classes.MultiFieldMultiPoint)
    addable(squared_field_point_a, multi_field_multi_point, classes.MultiFieldMultiPoint)
    not_addable(squared_field_point_a, fields)
    not_addable(squared_field_point_a, cost_fields)
    not_addable(squared_field_point_a, cost_field_points)

    # Test of other morphing
    mult(squared_field_point_a, classes.SquaredCostFieldPoint)
    bind(squared_field_point_a, classes.SquaredFieldPoint)
    sub(squared_field_point_a, classes.SquaredFieldPoint)

    # Test misc
    str(squared_field_point_a)


def test_squared_cost_field():
    # Test for addition
    addable(squared_cost_field, cost_field, classes.MultiCostField)
    addable(squared_cost_field, squared_cost_field, classes.MultiCostField)
    addable(squared_cost_field, multi_cost_field, classes.MultiCostField)
    not_addable(squared_cost_field, fields)
    not_addable(squared_cost_field, field_points)
    not_addable(squared_cost_field, cost_field_points)

    # Test of other morphing
    mult(squared_cost_field, classes.SquaredCostField)
    bind(squared_cost_field, classes.SquaredCostFieldPoint)
    sub(squared_cost_field, classes.SquaredCostField)

    # Test misc
    str(squared_cost_field)


def test_squared_cost_field_point():
    # Test for addition
    addable(squared_cost_field_point_a, cost_field_a, classes.MultiCostFieldPoint)
    addable(squared_cost_field_point_a, cost_field_b, classes.MultiCostFieldMultiPoint)
    addable(squared_cost_field_point_a, squared_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(squared_cost_field_point_a, squared_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(squared_cost_field_point_a, multi_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(squared_cost_field_point_a, multi_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(squared_cost_field_point_a, multi_cost_field_multi_point, classes.MultiCostFieldMultiPoint)
    not_addable(squared_cost_field_point_a, fields)
    not_addable(squared_cost_field_point_a, cost_fields)
    not_addable(squared_cost_field_point_a, field_points)

    # Test of other morphing
    mult(squared_cost_field_point_a, classes.SquaredCostFieldPoint)
    bind(squared_cost_field_point_a, classes.SquaredCostFieldPoint)
    sub(squared_cost_field_point_a, classes.SquaredCostFieldPoint)

    # Test misc
    str(squared_cost_field_point_a)


def test_multi_field():
    # Test for addition
    addable(multi_field, field, classes.MultiField)
    addable(multi_field, squared_field, classes.MultiField)
    addable(multi_field, multi_field, classes.MultiField)
    not_addable(multi_field, field_points)
    not_addable(multi_field, cost_fields)
    not_addable(multi_field, cost_field_points)

    # Test of other morphing
    mult(multi_field, classes.MultiCostField)
    bind(multi_field, classes.MultiFieldPoint)
    sub(multi_field, classes.MultiField)

    # Test misc
    str(multi_field)


def test_multi_field_point():
    # Test for addition
    addable(multi_field_point_a, field_point_a, classes.MultiFieldPoint)
    addable(multi_field_point_a, field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_point_a, squared_field_point_a, classes.MultiFieldPoint)
    addable(multi_field_point_a, squared_field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_point_a, multi_field_point_a, classes.MultiFieldPoint)
    addable(multi_field_point_a, multi_field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_point_a, multi_field_multi_point, classes.MultiFieldMultiPoint)
    not_addable(multi_field_point_a, fields)
    not_addable(multi_field_point_a, cost_fields)
    not_addable(multi_field_point_a, cost_field_points)

    # Test of other morphing
    mult(multi_field_point_a, classes.MultiCostFieldPoint)
    bind(multi_field_point_a, classes.MultiFieldPoint)
    sub(multi_field_point_a, classes.MultiFieldPoint)

    # Test misc
    str(multi_field_point_a)


def test_multi_cost_field():
    # Test for addition
    addable(multi_cost_field, cost_field, classes.MultiCostField)
    addable(multi_cost_field, squared_cost_field, classes.MultiCostField)
    addable(multi_cost_field, multi_cost_field, classes.MultiCostField)
    not_addable(multi_cost_field, fields)
    not_addable(multi_cost_field, field_points)
    not_addable(multi_cost_field, cost_field_points)

    # Test of other morphing
    mult(multi_cost_field, classes.MultiCostField)
    bind(multi_cost_field, classes.MultiCostFieldPoint)
    sub(multi_cost_field, classes.MultiCostField)

    # Test misc
    str(multi_cost_field)


def test_multi_cost_field_point():
    # Test for addition
    addable(multi_cost_field_point_a, cost_field_a, classes.MultiCostFieldPoint)
    addable(multi_cost_field_point_a, cost_field_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_point_a, squared_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(multi_cost_field_point_a, squared_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_point_a, multi_cost_field_point_a, classes.MultiCostFieldPoint)
    addable(multi_cost_field_point_a, multi_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_point_a, multi_cost_field_multi_point, classes.MultiCostFieldMultiPoint)
    not_addable(multi_cost_field_point_a, fields)
    not_addable(multi_cost_field_point_a, cost_fields)
    not_addable(multi_cost_field_point_a, field_points)

    # Test of other morphing
    mult(multi_cost_field_point_a, classes.MultiCostFieldPoint)
    bind(multi_cost_field_point_a, classes.MultiCostFieldPoint)
    sub(multi_cost_field_point_a, classes.MultiCostFieldPoint)

    # Test misc
    str(multi_cost_field_point_a)


def test_multi_field_multi_point():
    # Test for addition
    addable(multi_field_multi_point, field_point_a, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, squared_field_point_a, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, squared_field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, multi_field_point_a, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, multi_field_point_b, classes.MultiFieldMultiPoint)
    addable(multi_field_multi_point, multi_field_multi_point, classes.MultiFieldMultiPoint)
    not_addable(multi_field_multi_point, fields)
    not_addable(multi_field_multi_point, cost_fields)
    not_addable(multi_field_multi_point, cost_field_points)

    mult(multi_field_multi_point, classes.MultiCostFieldMultiPoint)

    # Test misc
    str(multi_field_multi_point)


def test_multi_cost_field_multi_point():
    # Test for addition
    addable(multi_cost_field_multi_point, cost_field_a, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, cost_field_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, squared_cost_field_point_a, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, squared_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, multi_cost_field_point_a, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, multi_cost_field_point_b, classes.MultiCostFieldMultiPoint)
    addable(multi_cost_field_multi_point, multi_cost_field_multi_point, classes.MultiCostFieldMultiPoint)
    not_addable(multi_cost_field_multi_point, fields)
    not_addable(multi_cost_field_multi_point, cost_fields)
    not_addable(multi_cost_field_multi_point, field_points)

    mult(multi_cost_field_multi_point, classes.MultiCostFieldMultiPoint)

    # Test misc
    str(multi_cost_field_multi_point)
