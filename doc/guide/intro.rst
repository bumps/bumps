.. _intro-guide:

***********
Using Bumps
***********

.. contents:: :local:

The Bumps library is organized into modules.  Specific functions and
classes can be imported from a module, such as::

    >>> from bumps.parameter import Parameter

The most common imports have been gathered together in bumps.names.  This
allows you to use names like :class:`Parameter <bumps.parameter.Parameter>` directly::

    >>> from bumps.names import *
    >>> p = Parameter(name="P1", value=3.5, range=(2,6))

This pattern of importing all names from a file,  while convenient for
simple scripts, makes the code more difficult to understand later, and
can lead to unexpected results when the same name is used in multiple
modules.  A safer, though slightly more verbose pattern is to use:

    >>> import bumps.names as bmp
    >>> s = bmp.Parameter(name="P1", value=3.5, range=(2,6))

This documents to the reader unfamiliar with your code (such as you dear
reader when looking at your model files two years from now) exactly where 
the name comes from.

