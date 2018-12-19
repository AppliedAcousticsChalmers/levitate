Introduction
============

This is the documentation for the Levitate research project python toolbox.
The toolbox is distributed as an open source python package, hosted on the Chalmers Applied Acoustics GitHub (`<https://github.com/AppliedAcousticsChalmers/levitate>`_).
The primary goal of this toolbox is to provide a collection of algorithms and design patterns to aid researchers working with acoustic levitation and related topics, e.g. mid-air haptic feedback or parametric audio.
Included are both basic building blocks for simulating ultrasonic transducer arrays, and beamforming algorithms to design sound fields for specific purposes.

The package targets two major groups: Researchers who primarily focus on developing new algorithms used to design the sound fields, and researchers who use the existing algorithms to investigate areas of application, e.g. within human-computer interaction.
The first group requires the possibility of fast prototyping of new algorithms or schemes.
The inherent transparency in the python language together with the flexible and extensible design of the toolbox fulfills this requirement.
The second group needs simple and reliable tools to quickly design a sound field according to the needs of the application.
This is covered by the variety of algorithms existing in the toolbox, and the ease at which they can be applied in varying configurations.
Not considered at this point are end-users. 
The tools still require significant knowledge of the operator and, to a certain degree, understanding of the physical limitations.
