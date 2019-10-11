.. _field_wrappers:
.. default-role:: py:obj

##############
Field Wrappers
##############

.. automodule:: levitate._field_wrappers

Class list
==========

Public API
----------
These are the only classes and functions regarded as part of the public API,
but they will only be used directly when implementing new algorithm types.

.. autoclass:: levitate._field_wrappers.FieldImplementation
    :members:

Basic Types
-----------
.. autoclass:: levitate._field_wrappers.Field
    :members:

.. autoclass:: levitate._field_wrappers.FieldPoint
    :members:

.. autoclass:: levitate._field_wrappers.CostField
    :members:

.. autoclass:: levitate._field_wrappers.CostFieldPoint
    :members:

Magnitude Squared Types
-----------------------
.. autoclass:: levitate._field_wrappers.SquaredFieldBase
    :members:

.. autoclass:: levitate._field_wrappers.SquaredField
    :members:

.. autoclass:: levitate._field_wrappers.SquaredFieldPoint
    :members:

.. autoclass:: levitate._field_wrappers.SquaredCostField
    :members:

.. autoclass:: levitate._field_wrappers.SquaredCostFieldPoint
    :members:

MultiFields
-----------
.. autoclass:: levitate._field_wrappers.MultiField
    :members:

.. autoclass:: levitate._field_wrappers.MultiFieldPoint
    :members:

.. autoclass:: levitate._field_wrappers.MultiCostField
    :members:

.. autoclass:: levitate._field_wrappers.MultiCostFieldPoint
    :members:

MultiPoints
-----------
.. autoclass:: levitate._field_wrappers.MultiFieldMultiPoint
    :members:

.. autoclass:: levitate._field_wrappers.MultiCostFieldMultiPoint
    :members:

Private Classes
---------------
These classes are not considered part of the public API, and should not appear
other than as superclasses.

.. autoclass:: levitate._field_wrappers.FieldBase
    :members:
    :private-members:

.. autoclass:: levitate._field_wrappers.FieldImplementationMeta
    :members:

.. autoclass:: levitate._field_wrappers.FieldMeta
    :members:
    :private-members:
