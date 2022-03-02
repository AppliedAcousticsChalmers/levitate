.. _field_wrappers:
.. default-role:: py:obj

##############
Field Wrappers
##############

.. automodule:: levitate.fields._wrappers

Class list
==========

Public API
----------
These are the only classes and functions regarded as part of the public API,
but they will only be used directly when implementing new algorithm types.

.. autoclass:: levitate.fields._wrappers.FieldImplementation
    :members:

Basic Types
-----------
.. autoclass:: levitate.fields._wrappers.Field
    :members:

.. autoclass:: levitate.fields._wrappers.FieldPoint
    :members:

.. autoclass:: levitate.fields._wrappers.CostField
    :members:

.. autoclass:: levitate.fields._wrappers.CostFieldPoint
    :members:

Magnitude Squared Types
-----------------------
.. autoclass:: levitate.fields._wrappers.SquaredFieldBase
    :members:

.. autoclass:: levitate.fields._wrappers.SquaredField
    :members:

.. autoclass:: levitate.fields._wrappers.SquaredFieldPoint
    :members:

.. autoclass:: levitate.fields._wrappers.SquaredCostField
    :members:

.. autoclass:: levitate.fields._wrappers.SquaredCostFieldPoint
    :members:

MultiFields
-----------
.. autoclass:: levitate.fields._wrappers.MultiField
    :members:

.. autoclass:: levitate.fields._wrappers.MultiFieldPoint
    :members:

.. autoclass:: levitate.fields._wrappers.MultiCostField
    :members:

.. autoclass:: levitate.fields._wrappers.MultiCostFieldPoint
    :members:

MultiPoints
-----------
.. autoclass:: levitate.fields._wrappers.MultiFieldMultiPoint
    :members:

.. autoclass:: levitate.fields._wrappers.MultiCostFieldMultiPoint
    :members:

Private Classes
---------------
These classes are not considered part of the public API, and should not appear
other than as superclasses.

.. autoclass:: levitate.fields._wrappers.FieldBase
    :members:
    :private-members:

.. autoclass:: levitate.fields._wrappers.FieldImplementationMeta
    :members:

.. autoclass:: levitate.fields._wrappers.FieldMeta
    :members:
    :private-members:
