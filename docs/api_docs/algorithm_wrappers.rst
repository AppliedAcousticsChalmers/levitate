.. _algorithm_wrappers:
.. default-role:: py:obj

##################
Algorithm Wrappers
##################

.. automodule:: levitate._algorithm

Class list
==========

Public API
----------
These are the only classes and functions regarded as part of the public API,
but they will only be used directly when implementing new algorithm types.

.. autoclass:: levitate._algorithm.AlgorithmImplementation
    :members:

Basic Types
-----------
.. autoclass:: levitate._algorithm.Algorithm
    :members:

.. autoclass:: levitate._algorithm.BoundAlgorithm
    :members:

.. autoclass:: levitate._algorithm.UnboundCostFunction
    :members:

.. autoclass:: levitate._algorithm.CostFunction
    :members:

Magnitude Squared Types
-----------------------
.. autoclass:: levitate._algorithm.MagnitudeSquaredBase
    :members:

.. autoclass:: levitate._algorithm.MagnitudeSquaredAlgorithm
    :members:

.. autoclass:: levitate._algorithm.MagnitudeSquaredBoundAlgorithm
    :members:

.. autoclass:: levitate._algorithm.MagnitudeSquaredUnboundCostFunction
    :members:

.. autoclass:: levitate._algorithm.MagnitudeSquaredCostFunction
    :members:

Points
------
.. autoclass:: levitate._algorithm.AlgorithmPoint
    :members:

.. autoclass:: levitate._algorithm.BoundAlgorithmPoint
    :members:

.. autoclass:: levitate._algorithm.UnboundCostFunctionPoint
    :members:

.. autoclass:: levitate._algorithm.CostFunctionPoint
    :members:

Collections
-----------
.. autoclass:: levitate._algorithm.AlgorithmCollection
    :members:

.. autoclass:: levitate._algorithm.CostFunctionCollection
    :members:

Private Classes
---------------
These classes are not considered part of the public API, and should not appear
other than as superclasses.

.. autoclass:: levitate._algorithm.AlgorithmBase
    :members:
    :private-members:

.. autoclass:: levitate._algorithm.AlgorithmImplementationMeta
    :members:

.. autoclass:: levitate._algorithm.AlgorithmMeta
    :members:
    :private-members:
